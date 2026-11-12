"""CPU unit tests for checkpoint contracts and extracted training functions."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import decider_checkpoints as checkpoints
import train_decider_full as train
from decider_training_step import (
    DropoutSeed,
    StepConfig,
    prepare_batch,
    train_step,
    validate_and_clip_gradients,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def forward(self, input_ids, attention_mask, labels):
        prediction = (input_ids * attention_mask * self.weight).sum(dim=-1)
        return {"loss": ((prediction - labels) ** 2).mean()}


def collate(rows):
    return {
        "input_ids": torch.tensor([row[0] for row in rows]),
        "attention_mask": torch.ones(len(rows), 2),
        "labels": torch.tensor([row[1] for row in rows]),
    }


def capture_cpu_rng(collator):
    return dict(
        hardware_rng=torch.zeros(1, dtype=torch.int32),
        neuron_rng=torch.zeros(1, dtype=torch.uint8),
        cpu_rng=torch.get_rng_state(),
        collator_rng=collator.rng.getstate(),
    )


def restore_cpu_rng(state, collator):
    torch.set_rng_state(state["cpu_rng"])
    collator.rng.setstate(state["collator_rng"])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch("decider_checkpoints.dist.barrier"))
        self.model = TinyModel()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-6,
            betas=(0.9, 0.95),
            weight_decay=0,
            foreach=False,
        )
        self.collator = SimpleNamespace(rng=random.Random(42))
        self.signature = {"source": {"train_decider_full.py": "test-hash"}, "world": 1}

    def update(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(**collate([([1.0, 2.0], 0.4)]))["loss"].backward()
        optimizer.step()

    def save(self, path):
        checkpoints.save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.collator,
            step=1,
            rank=0,
            world=1,
            data_sha="dataset",
            signature=self.signature,
            capture_rng=capture_cpu_rng,
        )

    def test_adamw_save_resume_and_export_parity(self):
        self.update(self.model, self.optimizer)
        path = self.directory / "step-1"
        self.save(path)
        checkpoints.verify_checkpoint_export(path, self.model)
        expected_random = self.collator.rng.random()
        self.update(self.model, self.optimizer)
        expected = self.model.weight.detach().clone()
        restored = TinyModel()
        optimizer = torch.optim.AdamW(
            restored.parameters(),
            lr=5e-6,
            betas=(0.9, 0.95),
            weight_decay=0,
            foreach=False,
        )
        collator = SimpleNamespace(rng=random.Random(0))
        step = checkpoints.restore_checkpoint(
            path,
            restored,
            optimizer,
            collator,
            rank=0,
            world=1,
            data_sha="dataset",
            signature=self.signature,
            restore_rng=restore_cpu_rng,
        )
        self.assertEqual(step, 1)
        self.assertEqual(collator.rng.random(), expected_random)
        self.update(restored, optimizer)
        torch.testing.assert_close(restored.weight, expected, rtol=0, atol=0)
        with self.assertRaises(FileExistsError):
            self.save(path)

    def test_incomplete_checkpoint_never_gets_completion_marker(self):
        path = self.directory / "partial"
        with patch("decider_checkpoints.torch.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.save(path)
        self.assertFalse((path / checkpoints.COMPLETION_FILE).exists())
        with self.assertRaisesRegex(ValueError, "missing ranks"):
            checkpoints.write_completion(path, 1, 1)
        with self.assertRaises(FileNotFoundError):
            checkpoints.read_completion(path)

    def test_resume_rejects_identity_and_shape_changes_before_mutation(self):
        path = self.directory / "step-1"
        self.save(path)
        original = checkpoints.load_rank_checkpoint(path, 0)
        for field, replacement in (
            ("data_sha", "other"),
            ("signature", {}),
            ("dropout_seed_policy", "wrong"),
            ("world", 2),
            ("step", 2),
            ("parameters", {"weight": torch.zeros(3)}),
        ):
            with self.subTest(field=field):
                state = copy.deepcopy(original)
                state[field] = replacement
                torch.save(state, checkpoints.rank_path(path, 0))
                before = self.model.weight.detach().clone()
                with self.assertRaises(ValueError):
                    checkpoints.restore_checkpoint(
                        path,
                        self.model,
                        self.optimizer,
                        self.collator,
                        rank=0,
                        world=1,
                        data_sha="dataset",
                        signature=self.signature,
                        restore_rng=restore_cpu_rng,
                    )
                torch.testing.assert_close(self.model.weight, before, rtol=0, atol=0)

    def test_export_uses_rank_order_and_rejects_missing_or_mixed_shards(self):
        path = self.directory / "two-ranks"
        path.mkdir()
        states = [
            dict(
                parameters={"weight": torch.tensor([value])},
                step=1,
                world=2,
                data_sha="dataset",
                signature={},
                dropout_seed_policy=checkpoints.DROPOUT_SEED_POLICY,
            )
            for value in (0.3, -0.2)
        ]
        for rank, state in enumerate(states):
            torch.save(state, checkpoints.rank_path(path, rank))
        checkpoints.write_completion(path, 1, 2)
        checkpoints.verify_checkpoint_export(path, self.model)
        states[1]["signature"] = {"different": True}
        torch.save(states[1], checkpoints.rank_path(path, 1))
        with self.assertRaisesRegex(ValueError, "provenance"):
            checkpoints.verify_checkpoint_export(path, self.model)
        checkpoints.rank_path(path, 1).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            checkpoints.read_completion(path)

    def test_dtensor_optimizer_rebuild_preserves_layout_and_cpu_step(self):
        calls = []

        class FakeDTensor:
            def __init__(self, tensor):
                self.tensor = tensor
                self.device_mesh = "mesh"
                self.placements = ("shard(0)",)
                self.shape = torch.Size([4])

            def to_local(self):
                return self.tensor

            def stride(self):
                return (1,)

            @classmethod
            def from_local(cls, tensor, mesh, placements, *, shape, stride):
                calls.append((tensor, mesh, placements, shape, stride))
                return cls(tensor)

        path = self.directory / "dtensor"
        path.mkdir()
        parameter = FakeDTensor(torch.zeros(2))
        model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
        optimizer = SimpleNamespace(state={})
        state = dict(
            parameters={"weight": torch.ones(2)},
            optimizer={
                "weight": {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.ones(2),
                    "exp_avg_sq": torch.full((2,), 2.0),
                }
            },
            step=1,
            world=1,
            data_sha="dataset",
            signature=self.signature,
            dropout_seed_policy=checkpoints.DROPOUT_SEED_POLICY,
            **capture_cpu_rng(self.collator),
        )
        torch.save(state, checkpoints.rank_path(path, 0))
        checkpoints.write_completion(path, 1, 1)
        with patch("decider_checkpoints.DTensor", FakeDTensor):
            checkpoints.restore_checkpoint(
                path,
                model,
                optimizer,
                self.collator,
                rank=0,
                world=1,
                data_sha="dataset",
                signature=self.signature,
                restore_rng=restore_cpu_rng,
            )
        self.assertEqual(len(calls), 2)
        for _, mesh, placements, shape, stride in calls:
            self.assertEqual(
                (mesh, placements, shape, stride),
                ("mesh", ("shard(0)",), torch.Size([4]), (1,)),
            )
        self.assertEqual(optimizer.state[parameter]["step"].device.type, "cpu")
        torch.testing.assert_close(parameter.tensor, torch.ones(2))


class TrainingStepTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("decider_training_step.dist.all_reduce"))
        self.rows = [([1.0, 2.0], 4.0), ([3.0, 1.0], -2.0), ([2.0, 4.0], 1.0)]

    def test_step_matches_mean_loss_and_global_norm_reference(self):
        model, reference = TinyModel(), TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        reference(**collate(self.rows))["loss"].backward()
        norm_sq = sum(
            parameter.grad.float().square().sum()
            for parameter in reference.parameters()
        )
        scale = (1.0 / (norm_sq.sqrt() + 1e-6)).clamp(max=1.0)
        for parameter in reference.parameters():
            parameter.grad.mul_(scale)
        expected_optimizer.step()
        log = Mock()
        train_step(
            model,
            optimizer,
            collate,
            self.rows,
            step=0,
            config=StepConfig(0, 1, 2, 42, device="cpu"),
            dropout_seed=DropoutSeed(),
            log=log,
        )
        torch.testing.assert_close(model.weight, reference.weight)
        self.assertEqual(log.call_args_list[-1].args[0], "optimizer_step")

    def test_dummy_microbatch_has_zero_loss_and_position_seed(self):
        rows = self.rows + self.rows[:2]
        model = TinyModel()
        log, seed = Mock(), DropoutSeed()
        train_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0),
            collate,
            rows,
            step=3,
            config=StepConfig(3, 4, 2, 42, device="cpu"),
            dropout_seed=seed,
            log=log,
        )
        microbatches = [
            c.kwargs for c in log.call_args_list if c.args[0] == "microbatch"
        ]
        self.assertEqual([c["real"] for c in microbatches], [True, False])
        self.assertEqual(microbatches[1]["loss"], 0.0)
        self.assertEqual(seed.value, 42 + 3_000_000 + 7)
        expected_loss = float(model(**collate([rows[3]]))["loss"].detach()) * 4 / 5
        self.assertAlmostEqual(microbatches[0]["loss"], expected_loss, places=5)

    def test_clip_uses_reduced_global_norm_and_rejects_invalid_gradients(self):
        model = TinyModel()
        model.weight.grad = torch.tensor([3.0, 4.0])
        with patch(
            "decider_training_step.dist.all_reduce",
            side_effect=lambda value: value.mul_(4),
        ):
            validate_and_clip_gradients(model, device="cpu")
        torch.testing.assert_close(
            model.weight.grad, torch.tensor([3.0, 4.0]) / (10.0 + 1e-6)
        )
        for grad in (
            None,
            torch.tensor([float("nan"), 1.0]),
            torch.tensor([float("inf"), 1.0]),
        ):
            model.weight.grad = grad
            with self.assertRaisesRegex(ValueError, "weight"):
                validate_and_clip_gradients(model, device="cpu")
        model.weight.grad = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "weight"):
            validate_and_clip_gradients(model, device="meta")

    def test_padding_rejects_truncation_and_preserves_mask(self):
        batch = prepare_batch(
            collate(self.rows[:1]), pad_to=4, pad_token_id=9, device="cpu"
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1.0, 2.0, 9.0, 9.0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1.0, 1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            prepare_batch(
                collate(self.rows[:1]), pad_to=1, pad_token_id=0, device="cpu"
            )


class TrainerSetupTests(unittest.TestCase):
    def test_orchestration_keeps_partial_batch_and_final_checkpoint(self):
        rows = list(range(7))
        args = SimpleNamespace(
            upstream=Path("unused"),
            data=Path("unused"),
            merged=Path("merged"),
            output=Path("output"),
            resume=None,
            batch_size=5,
            steps=5,
            epochs=1,
            pad_to=8,
            seed=42,
            checkpoint_every=100,
        )
        model, optimizer, collator, seed = TinyModel(), Mock(), Mock(), DropoutSeed()
        with (
            patch.object(sys, "path", sys.path.copy()),
            patch(
                "train_decider_full.load_training_data", return_value=(rows, "dataset")
            ),
            patch("train_decider_full.training_signature", return_value={}),
            patch(
                "train_decider_full.create_training_components",
                return_value=(model, optimizer, collator, seed),
            ),
            patch("train_decider_full.train_step") as step,
            patch("train_decider_full.save_checkpoint") as save,
            patch("train_decider_full.validate_parameter_changes"),
            patch("train_decider_full.export_final_model") as export,
        ):
            train.run_training(args, 0, 4, Mock())
        self.assertEqual(
            [call.args[3] for call in step.call_args_list], [rows[:5], rows[5:]]
        )
        self.assertEqual([call.kwargs["step"] for call in step.call_args_list], [0, 1])
        self.assertEqual(step.call_args_list[0].kwargs["config"].device, "neuron")
        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["step"], 2)
        self.assertEqual(export.call_args.kwargs["total_steps"], 2)

    def test_export_restores_full_state_strictly_and_writes_run_metadata(self):
        model = Mock()
        exported = SimpleNamespace(config=SimpleNamespace(), load_state_dict=Mock())
        full_state = {"weight": torch.ones(2)}
        model_io = SimpleNamespace(
            load_full=Mock(return_value=exported), save_full=Mock()
        )
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                merged=Path("merged"), output=Path(directory), epochs=1
            )
            with (
                patch.dict(sys.modules, {"decider_model_io": model_io}),
                patch(
                    "train_decider_full.gather_full_parameters", return_value=full_state
                ),
                patch("train_decider_full.dist.barrier") as barrier,
            ):
                train.export_final_model(
                    model,
                    args,
                    rank=0,
                    world=4,
                    signature={"test": True},
                    total_steps=2,
                    row_count=7,
                    log=Mock(),
                )
            exported.load_state_dict.assert_called_once_with(full_state, strict=True)
            model_io.save_full.assert_called_once_with(exported, args.output / "final")
            self.assertEqual(exported.config.temperature, 1.0)
            self.assertEqual(exported.config.temperature_by_kind, {})
            record = json.loads((args.output / "run.json").read_text())
            self.assertEqual(
                (record["steps"], record["rows"], record["world"]), (2, 7, 4)
            )
            self.assertEqual(record["selection"], "last_step")
            barrier.assert_called_once()

    def test_compile_precedes_sharding_and_head_remains_seeded_fp32(self):
        events = []
        layers = [
            SimpleNamespace(forward=Mock(name=f"layer-{index}")) for index in range(2)
        ]
        model = SimpleNamespace(
            torso=SimpleNamespace(
                layers=layers,
                embed_tokens=object(),
                gradient_checkpointing_enable=Mock(),
            ),
            head=SimpleNamespace(forward=Mock(return_value="head-result")),
            training=True,
        )
        seed, neuron = DropoutSeed(), SimpleNamespace(manual_seed=Mock())
        with (
            patch("train_decider_full.init_device_mesh", return_value="mesh"),
            patch(
                "train_decider_full.torch.compile",
                side_effect=lambda forward, **kw: events.append(("compile", forward))
                or forward,
            ),
            patch(
                "train_decider_full.fully_shard",
                side_effect=lambda module, **kw: events.append(("shard", module)),
            ),
            patch.object(torch, "neuron", neuron, create=True),
            patch(
                "train_decider_full.torch.autocast",
                side_effect=lambda *a, **k: nullcontext(),
            ) as autocast,
        ):
            train.configure_fsdp_model(model, 4, seed)
            self.assertEqual(
                [kind for kind, _ in events],
                ["compile", "shard", "compile", "shard", "shard", "shard", "shard"],
            )
            self.assertIs(events[-1][1], model)
            seed.value = 123
            self.assertEqual(model.head.forward(), "head-result")
            neuron.manual_seed.assert_called_once_with(123)
            autocast.assert_called_with("neuron", enabled=False)
            model.training = False
            model.head.forward()
            neuron.manual_seed.assert_called_once()
        model.torso.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def test_global_update_validation_accepts_remote_embedding_change(self):
        model = torch.nn.Module()
        model.head = torch.nn.Linear(1, 1, bias=False)
        model.torso = torch.nn.Module()
        model.torso.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])
        model.torso.embed_tokens = torch.nn.Embedding(2, 1)
        before = train.snapshot_parameters(model)
        with torch.no_grad():
            model.head.weight.add_(1)
            model.torso.layers[0].weight.add_(1)
        index = list(before).index("torso.embed_tokens.weight")
        with patch(
            "train_decider_full.dist.all_reduce",
            side_effect=lambda flags: flags.__setitem__(index, 1),
        ):
            changed = train.validate_parameter_changes(
                model, before, Mock(), device="cpu"
            )
        self.assertIn("torso.embed_tokens.weight", changed)
        with patch("train_decider_full.dist.all_reduce"):
            with self.assertRaisesRegex(ValueError, "embed_tokens"):
                train.validate_parameter_changes(model, before, Mock(), device="cpu")

    def test_every_rank_gathers_but_only_rank_zero_retains(self):
        parameter = SimpleNamespace(full_tensor=Mock(return_value=torch.ones(2)))
        model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
        self.assertEqual(train.gather_full_parameters(model, 1), {})
        self.assertEqual(parameter.full_tensor.call_count, 1)
        result = train.gather_full_parameters(model, 0)
        torch.testing.assert_close(result["weight"], torch.ones(2))
        self.assertEqual(parameter.full_tensor.call_count, 2)

    def test_process_group_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(seed=42, output=Path(directory) / "run")
            with (
                patch("train_decider_full.initialize_distributed", return_value=(0, 4)),
                patch("train_decider_full.dist.barrier"),
                patch("train_decider_full.dist.is_initialized", return_value=True),
                patch("train_decider_full.dist.destroy_process_group") as destroy,
                patch(
                    "train_decider_full.run_training",
                    side_effect=ValueError("test failure"),
                ),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(ValueError, "test failure"):
                    train.main(args)
                destroy.assert_called_once()
            record = json.loads((args.output / "rank-0.jsonl").read_text())
            self.assertEqual(record["event"], "error")


class CommandLineTests(unittest.TestCase):
    def test_documented_data_option_survives_module_renames(self):
        for script in (
            "prepare_decider_model.py",
            "train_decider_full.py",
            "verify_decider_training.py",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, "-B", str(PACKAGE_ROOT / script), "--help"],
                    cwd="/tmp",
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("--data DATA", result.stdout)
                self.assertNotIn("--prepare_decider_dataset", result.stdout)


if __name__ == "__main__":
    unittest.main()
