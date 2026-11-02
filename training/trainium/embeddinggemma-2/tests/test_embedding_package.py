"""Lightweight failure-path tests; no model download or accelerator required."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_embedding_data import prepare_data
from embedding_training_config import COMPILE_CACHE_DIRECTORIES

spec = importlib.util.spec_from_file_location(
    "recipe", ROOT / "train_embedding_standalone.py"
)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class PackageTests(unittest.TestCase):
    def run_launcher(self, runtime_dir):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / 'venv'
            tools = root / 'bin'
            (venv / 'bin').mkdir(parents=True)
            tools.mkdir()
            python = venv / 'bin/python'
            python.write_text('#!/bin/sh\nexit 0\n')
            python.chmod(0o755)
            flock = tools / 'flock'
            flock.write_text('#!/bin/sh\nexit 0\n')
            flock.chmod(0o755)
            stat = tools / 'stat'
            stat.write_text(f'#!/bin/sh\nprintf "%s 700\\n" "{os.getuid()}"\n')
            stat.chmod(0o755)
            env = dict(os.environ, RECIPE_VENV=str(venv),
                       TRAINIUM_RECIPE_RUNTIME_DIR=str(runtime_dir),
                       PATH=str(tools) + os.pathsep + os.environ.get('PATH', ''))
            return subprocess.run(
                ['bash', str(ROOT / 'run_embedding_training.sh')], env=env,
                capture_output=True, text=True)

    def test_launcher_rejects_symlink_lock_without_truncating_target(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / 'runtime'
            runtime.mkdir(mode=0o700)
            target = Path(directory) / 'target'
            target.write_text('preserve-me')
            (runtime / 'embeddinggemma-2.lock').symlink_to(target)
            result = self.run_launcher(runtime)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Refusing symbolic-link lock file', result.stderr)
            self.assertEqual(target.read_text(), 'preserve-me')

    def test_launcher_accepts_private_runtime_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / 'runtime'
            result = self.run_launcher(runtime)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(runtime.stat().st_mode & 0o777, 0o700)
            self.assertTrue((runtime / 'embeddinggemma-2.lock').is_file())

    def test_missing_compiler_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'prefix', directory), patch.object(
                sys, 'executable', directory + '/bin/python'
            ), patch.dict(os.environ, {'PATH': ''}):
                with self.assertRaisesRegex(RuntimeError, 'neuronx-cc is not on PATH'):
                    recipe.runtime_preflight()

    def test_neuron_backend_workspace_is_run_scoped(self):
        variables = dict(COMPILE_CACHE_DIRECTORIES)
        self.assertEqual(variables['TORCH_NEURONX_DEBUG_DIR'], 'debug')

    def test_preflight_cannot_silently_become_preparation(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'train_embedding_standalone.py'),
             '--preflight', '--prepare_only'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot be combined', result.stderr)

    def test_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'keep.txt'
            sentinel.write_text('preserved')
            result = subprocess.run(
                [sys.executable, str(ROOT / 'train_embedding_standalone.py'),
                 '--prepare_only', '--output_dir', directory],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'preserved')

    def test_data_module_preserves_sample_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                train_file=ROOT / 'OTel_embedding_sample_100.jsonl',
                seed=42,
                num_negatives=1,
                eval_corpus_size=20000,
                eval_max_seq_length=1024,
                matryoshka_dims=[768, 512, 256, 128],
                query_prompt='task: search result | query: ',
                document_prompt='title: none | text: ',
            )
            split_rows, _, audit = prepare_data(
                args, Path(directory) / 'data'
            )
            self.assertEqual(
                {name: len(split_rows[name]) for name in split_rows},
                {'train': 93, 'validation': 3, 'test': 4},
            )
            self.assertEqual(audit['evaluation_documents'], 599)


if __name__ == '__main__':
    unittest.main()
