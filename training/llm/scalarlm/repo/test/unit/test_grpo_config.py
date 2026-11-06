import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


HERE = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    def load(name, filename):
        if not (HERE / filename).is_file():
            pytest.skip(f"Client {filename} is not shipped in the server image")
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    utils = load("scalarlm_outer_utils_under_test", "utils.py")
    monkeypatch.setitem(sys.modules, "utils", utils)
    driver = load("scalarlm_outer_grpo_under_test", "train_llm_grpo_scalarlm.py")
    return utils, driver


@pytest.fixture
def cli(monkeypatch, tmp_path):
    def parse(utils, *extra):
        monkeypatch.setattr(sys, "argv", ["train_llm_grpo_scalarlm.py", "--model", str(tmp_path / "model"),
                                          "--data", str(tmp_path / "data.jsonl"),
                                          "--output-dir", str(tmp_path / "output"), *extra])
        return utils.parse_args()

    return parse


@pytest.mark.parametrize("selected", [None, "grpo_judge_api"])
def test_loop_selects_reward_module_and_records_its_rubric(modules, cli, monkeypatch, selected):
    utils, driver = modules
    args = cli(utils, "--steps", "0", "--warmup-steps", "0")
    if selected is None:
        monkeypatch.delenv("GRPO_JUDGE_MODULE", raising=False)
    else:
        monkeypatch.setenv("GRPO_JUDGE_MODULE", selected)

    def forbidden(*args, **kwargs):
        pytest.fail("zero-cycle reward selection must not score, roll out, or train")

    for name in ("grpo_judge_local", "grpo_judge_api"):
        reward = types.ModuleType(name)
        reward.RUBRIC_VERSION = f"test-{name}"
        reward.score_batch = forbidden
        monkeypatch.setitem(sys.modules, name, reward)
    for name in ("rollout", "submit_training_job", "sync_adapter"):
        monkeypatch.setattr(utils, name, forbidden)
    output = Path(args.output_dir)
    output.mkdir()
    summary = driver.run_loop(args, [], "", driver.base_train_args(args, "targets"), False, str(output))
    assert summary["rubric"] == f"test-{selected or 'grpo_judge_local'}"
    assert summary["n_cycles"] == 0
    assert summary["models"] == []
    assert json.loads((output / "summary.json").read_text())["rubric"] == summary["rubric"]


def test_unknown_reward_module_fails_before_rollout(modules, cli, monkeypatch):
    utils, driver = modules
    args = cli(utils, "--steps", "0", "--warmup-steps", "0")
    monkeypatch.setenv("GRPO_JUDGE_MODULE", "scalarlm_nonexistent_reward_for_test")
    with pytest.raises(ModuleNotFoundError, match="scalarlm_nonexistent_reward_for_test"):
        driver.run_loop(args, [], "", {}, False, args.output_dir)
    assert not Path(args.output_dir).exists()


def test_cli_defaults_select_lora_grpo_and_adamw_bf16(modules, cli):
    utils, driver = modules
    args = cli(utils)
    assert args.optimizer_type == "adamw"
    config = driver.base_train_args(args, "targets")
    assert args.objective == "grpo"
    assert config["adapter_type"] == "lora"
    assert config["training_storage_dtype"] == "bfloat16"
    assert config["optimizer_type"] == "adamw"
    assert config["lora_config"] == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
                                     "target_modules": "targets", "use_rslora": False}
    assert config["training_mode"] == "grpo"
    assert config["grpo"] == {"clip_eps": 0.2, "kl_beta": 0.0}
    assert config["init_optimizer_from_adapter"] is True


@pytest.mark.parametrize("optimizer", ["adamw", "sgd", "rmsprop"])
@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
def test_cli_optimizer_and_storage_overrides_round_trip(modules, cli, optimizer, storage):
    utils, driver = modules
    args = cli(utils, "--training-storage-dtype", storage, "--optimizer-type", optimizer)
    config = driver.base_train_args(args, "targets")
    assert config["optimizer_type"] == optimizer
    assert config["training_storage_dtype"] == storage


@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
def test_storage_defaults_to_adamw(modules, cli, storage):
    utils, driver = modules
    args = cli(utils, "--training-storage-dtype", storage)
    assert args.optimizer_type == "adamw"
    config = driver.base_train_args(args, "targets")
    assert config["training_storage_dtype"] == storage
    assert config["optimizer_type"] == "adamw"


def test_rft_lora_config_has_only_language_model_fields(modules, cli):
    utils, driver = modules
    args = cli(utils, "--objective", "rft", "--rft-max-steps", "3", "--rft-learning-rate", "0.002",
               "--rft-batch-size", "4", "--rft-grad-accum", "5")
    config = driver.base_train_args(args, "targets")
    assert config["adapter_type"] == "lora"
    assert config["lora_config"]["target_modules"] == "targets"
    assert config["training_mode"] == "language_model"
    assert config["max_steps"] == 3
    assert config["learning_rate"] == 0.002
    assert config["batch_size"] == 4
    assert config["gradient_accumulation_steps"] == 5
    assert "grpo" not in config
    assert "init_optimizer_from_adapter" not in config


@pytest.mark.parametrize("flag", ["--adapter-type", "--training-storage-dtype", "--optimizer-type"])
def test_cli_rejects_invalid_choices(modules, cli, flag, capsys):
    utils, _ = modules
    with pytest.raises(SystemExit) as error:
        cli(utils, flag, "invalid")
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("objective", ["grpo", "rft"])
def test_full_parameter_config_omits_lora_fields(modules, cli, objective):
    utils, driver = modules
    args = cli(utils, "--objective", objective, "--adapter-type", "none")
    config = driver.base_train_args(args, None)
    assert config["adapter_type"] == "none"
    assert "lora_config" not in config
    assert config["training_mode"] == ("grpo" if objective == "grpo" else "language_model")
    if objective == "grpo":
        assert config["init_optimizer_from_adapter"] is False
    else:
        assert "grpo" not in config
        assert "init_optimizer_from_adapter" not in config
        assert config["max_steps"] == args.rft_max_steps
        assert config["learning_rate"] == args.rft_learning_rate
        assert config["gradient_accumulation_steps"] == args.rft_grad_accum


def test_lora_cli_overrides_and_optimizer_carry(modules, cli):
    utils, driver = modules
    args = cli(utils, "--lora-r", "8", "--lora-alpha", "64", "--lora-regex", "custom",
               "--no-carry-optimizer")
    config = driver.base_train_args(args, args.lora_regex)
    assert config["lora_config"]["r"] == 8
    assert config["lora_config"]["lora_alpha"] == 64
    assert config["lora_config"]["target_modules"] == "custom"
    assert config["init_optimizer_from_adapter"] is False


@pytest.mark.parametrize("objective", ["grpo", "rft"])
@pytest.mark.parametrize("persistent", [False, True])
def test_full_parameter_main_fails_before_model_reads_or_launch(modules, cli, monkeypatch, objective, persistent):
    utils, driver = modules
    extra = ["--objective", objective, "--adapter-type", "none", "--start-server", "--render-nodes", "0"]
    if persistent:
        extra.append("--persistent-trainer")
    args = cli(utils, *extra)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported full-parameter loop reached a side effect")

    for name in ("load_model_config", "load_rows", "start_server", "health", "submit_training_job",
                 "start_persistent_trainer"):
        monkeypatch.setattr(utils, name, forbidden)
    with pytest.raises(SystemExit, match=f"Full-parameter {objective.upper()} policy synchronization is unsupported"):
        driver.main()
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("objective", ["grpo", "rft"])
def test_direct_loop_rejects_full_parameter_mode(modules, cli, objective):
    utils, driver = modules
    args = cli(utils, "--adapter-type", "none", "--objective", objective)
    with pytest.raises(SystemExit, match="policy synchronization is unsupported"):
        driver.run_loop(args, [], "", {}, False, args.output_dir)


def test_full_parameter_dry_run_prints_config_without_lora_derivation(modules, cli, monkeypatch, capsys):
    utils, driver = modules
    cli(utils, "--adapter-type", "none", "--dry-run")

    def forbidden(*args, **kwargs):
        pytest.fail("full-parameter config should not derive LoRA targets")

    monkeypatch.setattr(utils, "load_model_config", forbidden)
    monkeypatch.setattr(utils, "default_lora_regex", forbidden)
    driver.main()
    output = json.loads(capsys.readouterr().out)
    config = output["train_args"]
    assert output["policy_sync_supported"] is False
    assert config["adapter_type"] == "none"
    assert "lora_config" not in config


@pytest.mark.parametrize("multimodal", [False, True])
def test_lora_dry_run_derives_model_specific_targets(modules, cli, monkeypatch, capsys, multimodal):
    utils, driver = modules
    args = cli(utils, "--dry-run")
    model = Path(args.model)
    model.mkdir()
    architectures = ["ExampleForConditionalGeneration" if multimodal else "ExampleForCausalLM"]
    (model / "config.json").write_text(json.dumps({"architectures": architectures}))
    driver.main()
    config = json.loads(capsys.readouterr().out)["train_args"]
    assert config["lora_config"]["target_modules"] == (
        (r".*language_model.*" if multimodal else r".*") + utils.LORA_PROJECTIONS)
