"""Dependency-free data, path and batching contract tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from fractions import Fraction

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from decider_recipe_utils import (
    load_rows,
    StrictTokenizer,
    write_json,
    digest,
    verify_manifest,
)
from prepare_decider_dataset import example


class RecipeTests(unittest.TestCase):
    def test_valid_and_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "rows.jsonl"
            row = example("A synthetic sentence.", 0)
            p.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(load_rows(p)), 1)
            p.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_rows(p)

    def test_bad_inputs(self):
        for changes in (
            {"label": True},
            {"label": 4},
            {"state": ""},
            {"kind": "score"},
            {"weight": 2},
            {"options": [["same", "a"], ["same", "b"]]},
        ):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as t:
                p = Path(t) / "rows.jsonl"
                p.write_text(json.dumps(dict(example("text", 0), **changes)))
                with self.assertRaises(ValueError):
                    load_rows(p)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "report.json"
            write_json(p, {"ok": 1})
            with self.assertRaises(FileExistsError):
                write_json(p, {"ok": 2})

    def test_distributed_loss_weight(self):
        for n in range(1, 14):
            for world in (1, 2, 4):
                rank_sums = []
                for rank in range(world):
                    rank_sums.append(
                        sum(
                            (
                                Fraction(world, n) * (i + 1)
                                for i in range(rank, n, world)
                            ),
                            Fraction(0),
                        )
                    )
                self.assertEqual(
                    sum(rank_sums) / world, Fraction(sum(range(1, n + 1)), n)
                )

    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "weights").write_bytes(b"original")
            write_json(
                root / "manifest.json", {"files": {"weights": digest(root / "weights")}}
            )
            verify_manifest(root)
            (root / "weights").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_manifest(root)

    def test_architecture_detects_same_shape_changes(self):
        from prepare_decider_model import architecture
        from types import SimpleNamespace

        class Config:
            def to_json(self):
                return json.dumps(
                    {
                        "temperature": 1.0,
                        "temperature_by_kind": {},
                        "head_type": "pointer",
                    }
                )

        class Dropout:
            p = 0.05

        class Tokenizer:
            special_tokens_map = {"eos_token": "end"}

            def get_vocab(self):
                return self.vocab

            vocab = {"a": 0, "end": 1}

        drop = Dropout()
        tok = Tokenizer()
        model = SimpleNamespace(
            config=Config(),
            tokenizer=tok,
            head=drop,
            torso=SimpleNamespace(
                config=SimpleNamespace(
                    to_dict=lambda: {"hidden_size": 8, "id2label": {0: "LABEL_0"}}
                )
            ),
            named_parameters=lambda: [],
            parameters=lambda: [],
            named_modules=lambda: [("head.dropout", drop)],
        )
        before = architecture(model)
        self.assertEqual(before, json.loads(json.dumps(before)))
        drop.p = 0.1
        self.assertNotEqual(before["dropout"], architecture(model)["dropout"])
        drop.p = 0.05
        tok.vocab = {"a": 1, "end": 0}
        self.assertNotEqual(
            before["tokenizer_vocab_sha256"],
            architecture(model)["tokenizer_vocab_sha256"],
        )

    def test_no_truncation(self):
        class Fake:
            def __call__(self, *args, **kwargs):
                assert kwargs["truncation"] is False and "max_length" not in kwargs

                class IDs:
                    shape = (1, 9)

                return {"input_ids": IDs()}

        with self.assertRaisesRegex(ValueError, "exceeds"):
            StrictTokenizer(Fake(), 8)(["text"], truncation=True, max_length=8)


if __name__ == "__main__":
    unittest.main()
