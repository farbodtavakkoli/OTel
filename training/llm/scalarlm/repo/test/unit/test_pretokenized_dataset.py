import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=["repo", "outer"])
def loader(request, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    root = repo if request.param == "repo" else repo.parent
    path = root / "ml/cray_megatron/megatron/dataset/load_language_model_dataset.py"
    spec = importlib.util.spec_from_file_location(f"dataset_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_data_parallel_world_size", lambda: 1)
    return module


@pytest.fixture
def model():
    return SimpleNamespace(
        config=SimpleNamespace(vocab_size=128, max_position_embeddings=16, pad_token_id=0),
        generation_config=SimpleNamespace(eos_token_id=99),
    )


@pytest.fixture
def row():
    return {
        "input_ids": [10, 11, 12, 13, 14, 15, 16, 99],
        "labels": [-100, -100, 12, 13, -100, -100, 16, -100],
        "attention_mask": [1] * 8,
    }


def configure(loader, monkeypatch, tmp_path, rows, **settings):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {"training_data_path": str(path), "max_token_block_size": 16, **settings}
    monkeypatch.setattr(loader, "get_job_config", lambda: config)
    return config


class ForbiddenTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text):
        raise AssertionError("Pretokenized rows must not be tokenized")


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text):
        if isinstance(text, list):
            ids = [[ord(char) for char in item] for item in text]
            return {"input_ids": ids, "attention_mask": [[1] * len(item) for item in ids]}
        return {"input_ids": [ord(char) for char in text]}


@pytest.mark.parametrize("ends_in_eos", [False, True])
def test_explicit_turn_masks_survive_loading_without_eos_changes(
    loader, model, row, monkeypatch, tmp_path, ends_in_eos
):
    if not ends_in_eos:
        row["input_ids"][-1] = 17
    original = copy.deepcopy(row)
    configure(loader, monkeypatch, tmp_path, [row], pretokenized_dataset=True)
    items = list(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0))
    assert len(items) == 1
    item = items[0]
    assert item["input_ids"].tolist() == row["input_ids"] + [0] * 8
    assert item["labels"].tolist() == row["labels"] + [-100] * 8
    assert item["attention_mask"].tolist() == [1] * 8 + [0] * 8
    assert item["position_ids"].tolist() == list(range(8)) + [0] * 8
    assert item["document_ids"].tolist() == [0] * 8 + [1] * 8
    assert item["labels"].dtype == torch.int64
    assert row == original


def test_packing_masks_document_starts_and_preserves_interior_turn_masks(
    loader, model, row, monkeypatch, tmp_path
):
    row["labels"][0] = row["input_ids"][0]
    configure(loader, monkeypatch, tmp_path, [row, row], pretokenized_dataset=True)
    item = next(iter(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0)))
    assert item["labels"].tolist() == ([-100] + row["labels"][1:]) * 2
    assert item["document_ids"].tolist() == [0] * 8 + [1] * 8
    assert item["position_ids"].tolist() == list(range(8)) * 2


def test_ten_plus_ten_rows_are_not_split_at_budget_sixteen(
    loader, model, monkeypatch, tmp_path
):
    row = {
        "input_ids": list(range(10, 20)),
        "labels": list(range(10, 20)),
        "attention_mask": [1] * 10,
    }
    configure(loader, monkeypatch, tmp_path, [row, row], pretokenized_dataset=True)
    items = list(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0))
    assert len(items) == 2
    for item in items:
        assert item["input_ids"].tolist() == row["input_ids"] + [0] * 6
        assert item["labels"].tolist() == [-100] + row["labels"][1:] + [-100] * 6
        assert item["attention_mask"].tolist() == [1] * 10 + [0] * 6
        assert item["position_ids"].tolist() == list(range(10)) + [0] * 6
        assert item["document_ids"].tolist() == [0] * 10 + [1] * 6


@pytest.mark.parametrize(
    "lengths,groups",
    [
        ([], []),
        ([2], [[0]]),
        ([16], [[0]]),
        ([16, 16], [[0], [1]]),
        ([14, 2], [[0, 1]]),
        ([2, 14], [[0, 1]]),
        ([15, 2], [[0], [1]]),
        ([2, 15], [[0], [1]]),
        ([10, 10], [[0], [1]]),
        ([8, 8, 2], [[0, 1], [2]]),
        ([10, 6, 16, 2], [[0, 1], [2], [3]]),
        ([10, 10, 6], [[0], [1, 2]]),
        ([7, 7, 3, 13, 2], [[0, 1], [2, 3], [4]]),
    ],
)
def test_whole_row_greedy_packing_boundaries(
    loader, model, monkeypatch, tmp_path, lengths, groups
):
    rows = [
        {
            "input_ids": [10 + i] * length,
            "labels": [10 + i] * length,
            "attention_mask": [1] * length,
            "auxiliary_mask": [2 + i] * length,
        }
        for i, length in enumerate(lengths)
    ]
    configure(loader, monkeypatch, tmp_path, rows, pretokenized_dataset=True)
    dataset = {
        key: [row[key] for row in rows]
        for key in ("input_ids", "labels", "attention_mask", "auxiliary_mask")
    }
    original = copy.deepcopy(dataset)
    pack = loader.get_pack_function(model, ForbiddenTokenizer())
    result = pack(dataset)
    expected = {key: [] for key in (*dataset, "position_ids", "document_ids")}
    for group in groups:
        ids, labels, positions, documents, auxiliary = [], [], [], [], []
        for doc_idx, row_idx in enumerate(group):
            length = lengths[row_idx]
            ids.extend(rows[row_idx]["input_ids"])
            labels.extend([-100] + rows[row_idx]["labels"][1:])
            positions.extend(range(length))
            documents.extend([doc_idx] * length)
            auxiliary.extend(rows[row_idx]["auxiliary_mask"])
        real_length = len(ids)
        pad_count = 16 - real_length
        expected["input_ids"].append(ids + [0] * pad_count)
        expected["labels"].append(labels + [-100] * pad_count)
        expected["attention_mask"].append([1] * real_length + [0] * pad_count)
        expected["auxiliary_mask"].append(auxiliary + [0] * pad_count)
        expected["position_ids"].append(positions + [0] * pad_count)
        expected["document_ids"].append(documents + [len(group)] * pad_count)
    assert result == expected
    assert dataset == original
    assert pack(dataset) == expected


def test_whole_row_packing_rejects_misaligned_auxiliary_stream(
    loader, model, row, monkeypatch, tmp_path
):
    configure(loader, monkeypatch, tmp_path, [row], pretokenized_dataset=True)
    dataset = {key: [value] for key, value in row.items()}
    dataset["auxiliary_mask"] = [[1]]
    with pytest.raises(ValueError, match="misaligned auxiliary_mask"):
        loader.get_pack_function(model, ForbiddenTokenizer())(dataset)


@pytest.mark.parametrize("setting", [None, False])
def test_legacy_packing_keeps_concatenate_then_split_behavior(
    loader, model, monkeypatch, tmp_path, setting
):
    settings = {} if setting is None else {"pretokenized_dataset": setting}
    configure(loader, monkeypatch, tmp_path, [], **settings)
    dataset = {
        "input_ids": [list(range(10, 20)), list(range(20, 30))],
        "labels": [list(range(10, 20)), list(range(20, 30))],
        "attention_mask": [[1] * 10, [1] * 10],
    }
    result = loader.get_pack_function(model, CharacterTokenizer())(dataset)
    assert result["input_ids"] == [list(range(10, 26)), list(range(26, 30)) + [0] * 12]
    assert result["position_ids"] == [list(range(10)) + list(range(6)), list(range(6, 10)) + [0] * 12]
    assert result["document_ids"] == [[0] * 10 + [1] * 6, [1] * 4 + [2] * 12]


@pytest.mark.parametrize("setting", [None, False])
def test_legacy_input_output_still_tokenizes_and_supervises_appended_eos(
    loader, model, monkeypatch, tmp_path, setting
):
    settings = {} if setting is None else {"pretokenized_dataset": setting}
    configure(loader, monkeypatch, tmp_path, [{"input": "ab", "output": "de"}], **settings)
    item = next(iter(loader.load_language_model_dataset(model, CharacterTokenizer(), epoch=0)))
    assert item["input_ids"].tolist() == [97, 98, 100, 101, 99] + [0] * 11
    assert item["labels"].tolist() == [-100, -100, 100, 101, 99] + [-100] * 11
    assert item["attention_mask"].tolist() == [1] * 5 + [0] * 11


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("input_ids", None, "require"),
        ("labels", None, "require"),
        ("attention_mask", None, "require"),
        ("input_ids", [], "length"),
        ("input_ids", [10], "length"),
        ("input_ids", [10] * 17, "length"),
        ("labels", [-100], "misaligned"),
        ("attention_mask", [1], "misaligned"),
        ("input_ids", [-1] + [11, 12, 13, 14, 15, 16, 99], "vocabulary"),
        ("input_ids", [128] + [11, 12, 13, 14, 15, 16, 99], "vocabulary"),
        ("input_ids", [True] + [11, 12, 13, 14, 15, 16, 99], "vocabulary"),
        ("input_ids", [10.0] + [11, 12, 13, 14, 15, 16, 99], "vocabulary"),
        ("labels", [-100, -100, 11, 13, -100, -100, 16, -100], "corresponding"),
        ("labels", [-100, -100, 12.0, 13, -100, -100, 16, -100], "corresponding"),
        ("labels", [-101] * 8, "corresponding"),
        ("attention_mask", [2] * 8, "integer 0 or 1"),
        ("attention_mask", [True] * 8, "integer 0 or 1"),
        ("attention_mask", [1.0] * 8, "integer 0 or 1"),
        ("attention_mask", [1, 1, 0, 1, 1, 1, 1, 1], "unattended"),
        ("labels", [-100] * 8, "supervised shifted target"),
        ("labels", [10] + [-100] * 7, "supervised shifted target"),
    ],
)
def test_malformed_rows_fail_before_schema_can_coerce_values(
    loader, model, row, monkeypatch, tmp_path, field, value, error
):
    row[field] = value
    configure(loader, monkeypatch, tmp_path, [row], pretokenized_dataset=True)
    with pytest.raises(ValueError, match=error):
        list(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0))


def test_nested_vocabulary_length_boundaries_and_masked_attention(
    loader, model, monkeypatch, tmp_path
):
    model.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=128, max_position_embeddings=4))
    row = {"input_ids": [0, 127, 0, 0], "labels": [-100, 127, -100, -100], "attention_mask": [1, 1, 0, 0]}
    configure(loader, monkeypatch, tmp_path, [row], pretokenized_dataset=True, max_token_block_size=8)
    item = next(iter(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0)))
    assert item["input_ids"].tolist() == row["input_ids"]
    assert item["labels"].tolist() == row["labels"]
    assert item["attention_mask"].tolist() == row["attention_mask"]


def test_target_without_attended_predecessor_is_not_shifted_supervision(
    loader, model, monkeypatch, tmp_path
):
    row = {"input_ids": [10, 11], "labels": [-100, 11], "attention_mask": [0, 1]}
    configure(loader, monkeypatch, tmp_path, [row], pretokenized_dataset=True)
    with pytest.raises(ValueError, match="supervised shifted target"):
        list(loader.load_language_model_dataset(model, ForbiddenTokenizer(), epoch=0))
