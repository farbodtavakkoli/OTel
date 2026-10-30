import json
import os
import stat
import threading
import time

import pytest
import torch
import yaml
from torch import nn

from cray_infra.util.default_job_config import JobConfig
from cray_megatron.megatron import persistent_grpo as pg
from cray_megatron.megatron.dataset import load_grpo_dataset as lgd
from cray_megatron.megatron.training_loop import TrainingState

REQUIRED_FIELDS = {
    "job_directory": "/tmp/job",
    "training_data_path": "/tmp/dataset.jsonlines",
    "dataset_hash": "abc",
}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(2, 2, bias=False)
        nn.init.ones_(self.lora_A.weight)

    def backward_sync(self):
        pass

    def unwrap_model(self):
        return {"layer.lora_A.weight": self.lora_A.weight.detach().clone()}


class FakeLoop:
    def __init__(self):
        self.training_state = TrainingState()
        self.training_state.model_info = {"model": TinyModel(), "tokenizer": None,
                                          "init_adapter_checkpoint": {"big": 1}}
        self._grpo_step_stats = []
        self.micro_batches = []

    def training_step_accumulate(self, batch, accum_step, ga):
        model = self.training_state.model_info["model"]
        loss = model.lora_A(torch.ones(1, 2)).sum() * batch["advantage"]
        (loss / ga).backward()
        self.micro_batches.append((accum_step, ga))
        self._grpo_step_stats.append({"loss": float(loss), "mean_ratio": 1.0, "clip_frac": 0.0,
                                      "tokens": 3, "sequences": 1, "time_s": 0.01})
        return loss.detach()

    def optimizer_step(self):
        self.training_state.optimizer.step()
        self.training_state.scheduler.step()
        return True

    def sync_loss(self, loss):
        return loss, float(loss)

    def print_training_step_info(self, loss, step_time):
        pass

    def _log_grpo_step(self):
        self._grpo_step_stats = []

    def _stop_requested_on_any_rank(self, local_stop):
        return bool(local_stop)


class FakeDataLoader:
    def __init__(self, model, tokenizer, starting_epoch=0):
        with open(lgd._path_override) as f:
            self.rows = [json.loads(line) for line in f]
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        row = self.rows[self.i % len(self.rows)]
        self.i += 1
        return {"advantage": torch.tensor(float(row["advantage"]))}


@pytest.fixture
def session(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    job_dir = jobs / "session"
    job_dir.mkdir(parents=True)
    config_path = job_dir / "config.yaml"
    config = {"job_directory": str(job_dir), "training_mode": "grpo", "grpo": {"persistent": True},
              "adapter_type": "lora", "lora_config": {"lora_alpha": 32}}
    config_path.write_text(yaml.safe_dump(config))
    (job_dir / "status.json").write_text(json.dumps({"status": "TRAINING", "history": []}))

    monkeypatch.setattr(pg, "get_job_config", lambda: dict(config))
    monkeypatch.setattr(pg, "get_job_config_path", lambda: str(config_path))
    monkeypatch.setattr(pg, "is_main_rank", lambda: True)
    monkeypatch.setattr(pg, "DataLoader", FakeDataLoader)
    monkeypatch.setattr(pg, "get_optimizer", lambda m: torch.optim.SGD(m.parameters(), lr=0.1))
    monkeypatch.setattr(pg, "get_scheduler", lambda o, n: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0))
    monkeypatch.setattr(pg, "build_adapter_metadata", lambda: {"lora_alpha": 32})
    monkeypatch.setattr(pg, "POLL_S", 0.01)
    monkeypatch.setattr(lgd, "_path_override", None)
    return jobs, job_dir


def _start(loop):
    errors = []

    def target():
        try:
            pg.PersistentGrpo(loop).run()
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, errors


def _request(job_dir, seq, rows, model_name, ga=1, steps=1):
    inbox = job_dir / "inbox"
    deadline = time.time() + 10
    while not inbox.is_dir():
        assert time.time() < deadline, "trainer never created its inbox"
        time.sleep(0.01)
    data = f"{seq:06d}.jsonl"
    (inbox / data).write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp = inbox / f"{seq:06d}.json.tmp"
    tmp.write_text(json.dumps({"model_name": model_name, "data": data, "steps": steps,
                               "gradient_accumulation_steps": ga}))
    os.replace(tmp, inbox / f"{seq:06d}.json")


def _stop(job_dir, seq):
    (job_dir / "inbox" / f"{seq:06d}.json").write_text(json.dumps({"stop": True}))


def _result(job_dir, seq):
    path = job_dir / "outbox" / f"{seq:06d}.json"
    deadline = time.time() + 10
    while not path.exists():
        assert time.time() < deadline, f"no result for request {seq}"
        time.sleep(0.01)
    return json.loads(path.read_text())


def test_job_config_keeps_grpo_persistent():
    assert JobConfig(**REQUIRED_FIELDS, grpo={"clip_eps": 0.2}).dict()["grpo"]["persistent"] is False
    assert JobConfig(**REQUIRED_FIELDS, grpo={"persistent": True}).dict()["grpo"]["persistent"] is True


def test_is_persistent_grpo():
    assert pg.is_persistent_grpo({"training_mode": "grpo", "grpo": {"persistent": True}})
    assert not pg.is_persistent_grpo({"training_mode": "grpo", "grpo": {"persistent": False}})
    assert not pg.is_persistent_grpo({"training_mode": "grpo", "grpo": None})
    assert not pg.is_persistent_grpo({"training_mode": "language_model", "grpo": {"persistent": True}})


def test_dataset_path_override(monkeypatch):
    monkeypatch.setattr(lgd, "get_dataset_path", lambda: "/job/dataset.jsonlines")
    monkeypatch.setattr(lgd, "_path_override", None)
    seen = []
    monkeypatch.setattr(lgd, "_read_rows", lambda p: seen.append(p) or [])
    monkeypatch.setattr(lgd, "get_job_config", lambda: {"max_token_block_size": 16})
    monkeypatch.setattr(lgd, "get_max_position_embeddings", lambda c: 32)
    monkeypatch.setattr(lgd, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(lgd, "get_data_parallel_world_size", lambda: 1)

    class M:
        config = None

    tok = type("T", (), {"pad_token_id": 0})()
    lgd.load_grpo_dataset(M(), tok, 0)
    lgd.set_dataset_path_override("/inbox/000001.jsonl")
    lgd.load_grpo_dataset(M(), tok, 0)
    assert seen == ["/job/dataset.jsonlines", "/inbox/000001.jsonl"]


def test_serves_requests_publishes_adapters_and_stops(session):
    jobs, job_dir = session
    loop = FakeLoop()
    thread, errors = _start(loop)

    _request(job_dir, 1, [{"advantage": 1.0}, {"advantage": 1.0}], "a" * 64, ga=2)
    r1 = _result(job_dir, 1)
    _request(job_dir, 2, [{"advantage": -1.0}], "b" * 64)
    r2 = _result(job_dir, 2)
    _stop(job_dir, 3)
    thread.join(10)

    assert not thread.is_alive() and not errors
    assert "init_adapter_checkpoint" not in loop.training_state.model_info
    for d in ("inbox", "outbox"):
        assert stat.S_IMODE(os.stat(job_dir / d).st_mode) == 0o777
    assert loop.micro_batches == [(0, 2), (1, 2), (0, 1)]
    assert r1["model_name"] == "a" * 64 and r1["seq"] == 1 and len(r1["losses"]) == 1
    assert r1["grpo_steps"][0]["micro_batches"] == 2 and r1["grpo_steps"][0]["tokens"] == 6
    assert r2["seq"] == 2 and r2["losses"][0] < 0

    w1 = torch.load(jobs / ("a" * 64) / "checkpoint_0.pt")
    w2 = torch.load(jobs / ("b" * 64) / "checkpoint_0.pt")
    assert set(w1) == {"model_state_dict", "metadata", "step", "epoch"}
    assert w1["metadata"] == {"lora_alpha": 32} and (w1["step"], w2["step"]) == (0, 1)
    a1 = w1["model_state_dict"]["layer.lora_A.weight"]
    a2 = w2["model_state_dict"]["layer.lora_A.weight"]
    assert torch.allclose(a1, torch.full((2, 2), 0.9))
    assert torch.allclose(a2, torch.full((2, 2), 1.0))
    assert torch.equal(a2, loop.training_state.model_info["model"].lora_A.weight.detach())

    status = json.loads((jobs / ("a" * 64) / "status.json").read_text())
    assert status["status"] == "COMPLETED" and status["history"][0]["step"] == 0
    assert status["persistent_session"] == "session" and status["start_time"] > 0
    cfg = yaml.safe_load((jobs / ("a" * 64) / "config.yaml").read_text())
    assert cfg["job_directory"] == str(jobs / ("a" * 64))
    assert not list((jobs / ("a" * 64)).glob("*.tmp"))
    session_status = json.loads((job_dir / "status.json").read_text())
    assert session_status["status"] == "TRAINING" and session_status["persistent_cycles"] == 2


def test_idle_timeout_ends_session(session, monkeypatch):
    jobs, job_dir = session
    monkeypatch.setattr(pg, "IDLE_TIMEOUT_S", 0.2)
    thread, errors = _start(FakeLoop())
    thread.join(10)
    assert not thread.is_alive() and not errors
    assert not os.listdir(job_dir / "outbox")


def test_restarted_session_refuses_to_serve(session):
    jobs, job_dir = session
    (job_dir / "outbox").mkdir()
    (job_dir / "outbox" / "000001.json").write_text("{}")
    with pytest.raises(RuntimeError, match="restarted session"):
        pg.PersistentGrpo(FakeLoop()).run()
