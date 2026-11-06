import importlib.util
import pathlib
import sys
import types
from argparse import Namespace

import pytest

FAMILIES = ["qwen3", "llama3", "gemma3", "gemma4", "rnj-1", "olmo3", "mistral", "lfm", "phi4",
            "gpt-oss_reasoning", "gpt-oss_it", "other"]


@pytest.fixture(scope="module")
def train(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setitem(sys.modules, "scalarlm", types.ModuleType("scalarlm"))
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch_module.setitem(sys.modules, "dotenv", dotenv)
    path = pathlib.Path(__file__).resolve().parents[3] / "train.py"
    if not path.is_file():
        pytest.skip("Client train.py is not shipped in the server image")
    spec = importlib.util.spec_from_file_location("scalarlm_train_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


EXAMPLE = {"prompt": "What is 5G?", "completion": "A mobile standard.", "reasoning": "Recall telecom basics."}


@pytest.mark.parametrize("family", FAMILIES)
def test_output_tokens_is_the_full_conversation_split_once(train, family):
    config = Namespace(gradient_calculation="output_tokens", model_type=family)
    row = train.format_for_scalarlm(EXAMPLE, config)
    assert row["input"] + row["output"] == train.format_conversation(EXAMPLE, model_type=family)
    assert EXAMPLE["prompt"] in row["input"]
    assert EXAMPLE["completion"] not in row["input"]
    assert row["output"].count(EXAMPLE["completion"]) == 1


@pytest.mark.parametrize("family", ["qwen3", "llama3", "gemma3", "olmo3", "lfm", "phi4"])
def test_output_tokens_has_one_user_and_one_assistant_turn(train, family):
    config = Namespace(gradient_calculation="output_tokens", model_type=family)
    row = train.format_for_scalarlm(EXAMPLE, config)
    full = train.format_conversation(EXAMPLE, model_type=family)
    empty = train.format_conversation({"prompt": "", "completion": "", "reasoning": ""}, model_type=family)
    assert empty not in row["input"] + row["output"] or empty == ""
    assert (row["input"] + row["output"]).count(EXAMPLE["prompt"]) == full.count(EXAMPLE["prompt"]) == 1


def test_reasoning_is_trained_for_gpt_oss_reasoning(train):
    config = Namespace(gradient_calculation="output_tokens", model_type="gpt-oss_reasoning")
    row = train.format_for_scalarlm(EXAMPLE, config)
    assert row["input"].endswith("<|start|>assistant<|channel|>analysis<|message|>")
    assert row["output"].startswith(EXAMPLE["reasoning"])
    assert row["output"].endswith(EXAMPLE["completion"] + "<|return|>")


def test_gemma4_split_is_unchanged(train):
    config = Namespace(gradient_calculation="output_tokens", model_type="gemma4")
    row = train.format_for_scalarlm(EXAMPLE, config)
    assert row["input"] == "<bos><|turn>user\nWhat is 5G?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    assert row["output"] == "A mobile standard.<turn|>"


def test_qwen3_split_point(train):
    config = Namespace(gradient_calculation="output_tokens", model_type="qwen3")
    row = train.format_for_scalarlm(EXAMPLE, config)
    assert row["input"] == "<|im_start|>user\nWhat is 5G?<|im_end|>\n<|im_start|>assistant\n"
    assert row["output"] == "A mobile standard.<|im_end|>"


def test_schedule_client_round_trips(train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--warmup_steps", "2", "--scheduler_type", "cosine", "--dtype", "float32"])
    args = train.build_train_args(train.parse_args())
    assert args["warmup_steps"] == 2
    assert args["scheduler_type"] == "cosine"
    assert args["dtype"] == "float32"


def test_schedule_client_defaults_remain_linear(train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train.build_train_args(train.parse_args())
    assert args["warmup_steps"] == 0
    assert args["scheduler_type"] == "linear"
    assert args["dtype"] == "auto"


def test_schedule_client_rejects_unknown_scheduler(train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--scheduler_type", "bad"])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_sft_defaults_to_full_parameter_adamw_bf16(train, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    config = train.parse_args()
    assert config.optimizer_type == "adamw"
    args = train.build_train_args(config)
    assert args["adapter_type"] == "none"
    assert args["training_storage_dtype"] == "bfloat16"
    assert args["optimizer_type"] == "adamw"
    assert args["dtype"] == "auto"


@pytest.mark.parametrize("storage,optimizer", [("float32", "adamw"), ("bfloat16", "adamw")])
def test_sft_storage_override_defaults_to_adamw(train, monkeypatch, storage, optimizer):
    monkeypatch.setattr(sys, "argv", ["train.py", "--training_storage_dtype", storage])
    args = train.build_train_args(train.parse_args())
    assert args["training_storage_dtype"] == storage
    assert args["optimizer_type"] == optimizer


@pytest.mark.parametrize("storage", ["float32", "bfloat16"])
@pytest.mark.parametrize("optimizer", ["adamw", "sgd", "rmsprop"])
def test_sft_explicit_optimizer_is_preserved(train, monkeypatch, storage, optimizer):
    monkeypatch.setattr(sys, "argv", ["train.py", "--training_storage_dtype", storage,
                                      "--optimizer_type", optimizer, "--adapter_type", "lora"])
    args = train.build_train_args(train.parse_args())
    assert args["adapter_type"] == "lora"
    assert args["training_storage_dtype"] == storage
    assert args["optimizer_type"] == optimizer


@pytest.mark.parametrize("mode", ["embedding", "classification"])
def test_nonlanguage_modes_keep_float32_adamw_defaults(train, monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["train.py", "--training_mode", mode])
    args = train.build_train_args(train.parse_args())
    assert args["training_storage_dtype"] == "float32"
    assert args["optimizer_type"] == "adamw"


@pytest.mark.parametrize("optimizer", [None, "adamw", "sgd", "rmsprop"])
def test_older_namespace_without_storage_dtype_remains_supported(train, monkeypatch, optimizer):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    config = train.parse_args()
    del config.training_storage_dtype
    config.optimizer_type = optimizer
    args = train.build_train_args(config)
    assert args["training_storage_dtype"] == "bfloat16"
    assert args["optimizer_type"] == (optimizer or "adamw")


@pytest.mark.parametrize("flag", ["--training_storage_dtype", "--optimizer_type"])
def test_sft_precision_cli_rejects_unknown_choices(train, monkeypatch, capsys, flag):
    monkeypatch.setattr(sys, "argv", ["train.py", flag, "invalid"])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_entire_sequence_unchanged(train):
    config = Namespace(gradient_calculation="entire_sequence", model_type="qwen3")
    row = train.format_for_scalarlm(EXAMPLE, config)
    assert row == {"input": "", "output": train.format_conversation(EXAMPLE, model_type="qwen3")}
