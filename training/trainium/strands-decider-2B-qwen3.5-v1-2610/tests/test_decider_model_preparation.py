"""Offline Hub failure tests; requires the pinned huggingface-hub client."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    LocalTokenNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prepare_decider_model import download_snapshot_for_preparation
from decider_recipe_utils import MODEL_ID, MODEL_REVISION


class PreparationDownloadTests(unittest.TestCase):
    def test_failures_are_actionable_and_leave_output_uncreated(self):
        def response(status):
            return httpx.Response(
                status, request=httpx.Request("GET", "https://huggingface.co")
            )

        errors = (
            (
                RevisionNotFoundError("private error body", response=response(404)),
                "pinned revision",
            ),
            (
                GatedRepoError("private error body", response=response(403)),
                "hf auth login",
            ),
            (
                RepositoryNotFoundError("private error body", response=response(404)),
                "model ID",
            ),
            (LocalTokenNotFoundError("private error body"), "hf auth login"),
            (
                HfHubHTTPError("private error body", response=response(401)),
                "hf auth login",
            ),
            (
                HfHubHTTPError("private error body", response=response(403)),
                "hf auth login",
            ),
            (HfHubHTTPError("private error body", response=response(503)), "HTTP 503"),
            (httpx.ConnectError("private error body"), "connectivity"),
            (httpx.ReadTimeout("private error body"), "connectivity"),
            (httpx.RemoteProtocolError("private error body"), "connectivity"),
            (LocalEntryNotFoundError("private error body"), "cached snapshot"),
            (
                IncompleteSnapshotError("private error body", "/unused/snapshot"),
                "cached snapshot",
            ),
            (OfflineModeIsEnabled("private error body"), "HF_HUB_OFFLINE"),
            (OSError("private error body"), "free disk space"),
            (RuntimeError("private error body"), "snapshot transfer"),
        )
        for error, expected in errors:
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "new-parent" / "prepare_decider_model"
                cache = root / "cache"
                cache.mkdir()
                cached = cache / "partial"
                cached.write_text("reusable download state")
                with patch("huggingface_hub.snapshot_download", side_effect=error):
                    with self.assertRaises(SystemExit) as caught:
                        download_snapshot_for_preparation(output, cache)
                message = str(caught.exception)
                self.assertIn(MODEL_ID, message)
                self.assertIn(MODEL_REVISION, message)
                self.assertIn(expected, message)
                self.assertNotIn("private error body", message)
                self.assertFalse(output.parent.exists())
                self.assertEqual(cached.read_text(), "reusable download state")

    def test_failure_can_be_rerun_with_identical_output_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, cache = root / "prepare_decider_model", root / "cache"
            snapshot = str(cache / "snapshots" / MODEL_REVISION)
            attempts = []

            def download(*args, **kwargs):
                self.assertFalse(output.exists())
                attempts.append((args, kwargs))
                if len(attempts) == 1:
                    raise httpx.ConnectTimeout("timed out")
                return snapshot

            with patch("huggingface_hub.snapshot_download", side_effect=download):
                with self.assertRaises(SystemExit):
                    download_snapshot_for_preparation(output, cache)
                self.assertEqual(
                    download_snapshot_for_preparation(output, cache), snapshot
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(
                attempts,
                [((MODEL_ID,), {"revision": MODEL_REVISION, "cache_dir": cache})] * 2,
            )

    def test_existing_directory_is_preserved_without_network_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepare_decider_model"
            output.mkdir()
            marker = output / "user-file"
            marker.write_text("keep")
            with patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    download_snapshot_for_preparation(output, Path(directory) / "cache")
            download.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_dangling_output_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepare_decider_model"
            output.symlink_to(root / "absent", target_is_directory=True)
            with patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    download_snapshot_for_preparation(output, root / "cache")
            download.assert_not_called()
            self.assertTrue(output.is_symlink())

    def test_failure_exits_nonzero_without_traceback(self):
        script = (
            "import sys; from pathlib import Path; from unittest.mock import patch; "
            "import httpx; from prepare_decider_model import download_snapshot_for_preparation\n"
            "with patch('huggingface_hub.snapshot_download', "
            "side_effect=httpx.ConnectError('private error body')):\n"
            "    download_snapshot_for_preparation(Path(sys.argv[1]), Path(sys.argv[2]))\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(root / "prepare_decider_model"),
                    str(root / "cache"),
                ],
                cwd=PACKAGE_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(MODEL_ID, result.stderr)
            self.assertIn("connectivity", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("private error body", result.stderr)
            self.assertFalse((root / "prepare_decider_model").exists())


if __name__ == "__main__":
    unittest.main()
