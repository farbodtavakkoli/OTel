"""Offline retry-adapter tests; run with the application venv (urllib3 required)."""

import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from urllib3.exceptions import NewConnectionError, ReadTimeoutError, SSLError
from urllib3.response import HTTPResponse

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prepare_decider_dataset import OTEL_URL, download_otel, main


class DownloadTests(unittest.TestCase):
    def setUp(self):
        # Mock only the transport. The real connection pool and Retry adapter run.
        self.transport = Mock()

        def request(pool, *args, **kwargs):
            response = self.transport(*args, **kwargs)
            # _make_request normally attaches the current retry state.
            response.retries = kwargs["retries"]
            return response

        self.enterContext(
            patch("urllib3.connectionpool.HTTPSConnectionPool._make_request", request)
        )
        self.sleep = self.enterContext(patch("urllib3.util.retry.time.sleep"))
        self.enterContext(patch("urllib3.util.retry.random.random", return_value=0.5))
        self.bodies = []

    def response(self, status=200, body=b"complete", headers=None):
        stream = io.BytesIO(body)
        self.bodies.append(stream)
        return HTTPResponse(
            body=stream, status=status, headers=headers, preload_content=False
        )

    def broken_response(self):
        class BrokenBody(io.BytesIO):
            def read(self, *args, **kwargs):
                raise ConnectionResetError("connection reset while reading")

        stream = BrokenBody()
        self.bodies.append(stream)
        return HTTPResponse(body=stream, status=200, preload_content=False)

    def test_transient_statuses_then_success(self):
        self.transport.side_effect = [
            self.response(503),
            self.response(429),
            self.response(),
        ]
        self.assertEqual(download_otel(), b"complete")
        self.assertEqual(self.transport.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2.25])
        self.assertTrue(all(body.closed for body in self.bodies))
        for call in self.transport.call_args_list:
            self.assertEqual(call.args[1], "GET")

    def test_exhaustion_is_four_attempts_with_bounded_jitter(self):
        self.transport.side_effect = [self.response(503) for _ in range(4)]
        with self.assertRaises(SystemExit) as caught:
            download_otel()
        self.assertIn(OTEL_URL, str(caught.exception))
        self.assertIn("maximum 4 GET attempts", str(caught.exception))
        self.assertEqual(self.transport.call_count, 4)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2.25, 4.25])

    def test_permanent_statuses_and_redirects_do_not_retry(self):
        for status in (301, 302, 400, 401, 403, 404, 413):
            with self.subTest(status=status):
                self.transport.reset_mock()
                self.transport.side_effect = [
                    self.response(status, headers={"Retry-After": "3600"})
                ]
                with self.assertRaises(SystemExit) as caught:
                    download_otel()
                self.assertIn(f"HTTP {status}", str(caught.exception))
                self.assertEqual(self.transport.call_count, 1)

    def test_connection_timeout_and_body_failure_then_success(self):
        self.transport.side_effect = [
            NewConnectionError(None, "connection refused"),
            ReadTimeoutError(None, OTEL_URL, "timed out"),
            self.broken_response(),
            self.response(),
        ]
        self.assertEqual(download_otel(), b"complete")
        self.assertEqual(self.transport.call_count, 4)
        self.assertTrue(all(body.closed for body in self.bodies))

    def test_mixed_failures_share_one_budget(self):
        self.transport.side_effect = [
            self.response(503),
            self.broken_response(),
            ReadTimeoutError(None, OTEL_URL, "timed out"),
            self.response(503),
        ]
        with self.assertRaises(SystemExit):
            download_otel()
        self.assertEqual(self.transport.call_count, 4)

    def test_certificate_failure_does_not_retry(self):
        self.transport.side_effect = SSLError("certificate verify failed")
        with self.assertRaises(SystemExit) as caught:
            download_otel()
        self.assertIn("certificate verify failed", str(caught.exception))
        self.assertEqual(self.transport.call_count, 1)

    def test_checksum_and_size_failures_do_not_retry_or_write(self):
        for body in (b"wrong checksum", b"x" * (1024 * 1024 + 2)):
            with (
                self.subTest(size=len(body)),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.transport.reset_mock()
                self.transport.side_effect = [self.response(body=body)]
                output = (
                    Path(directory) / "new-directory" / "prepare_decider_dataset.jsonl"
                )
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    main(SimpleNamespace(mode="otel", output=output))
                self.assertEqual(self.transport.call_count, 1)
                self.assertFalse(output.parent.exists())
                self.assertTrue(self.bodies[-1].closed)

    def test_retry_success_writes_only_complete_csv(self):
        body = b"text,label\nA sample document.,doc_direct\n"
        self.transport.side_effect = [self.broken_response(), self.response(body=body)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepare_decider_dataset.jsonl"
            with patch(
                "prepare_decider_dataset.OTEL_SHA", hashlib.sha256(body).hexdigest()
            ):
                main(SimpleNamespace(mode="otel", output=output))
            self.assertIn("A sample document.", output.read_text())
        self.assertEqual(self.transport.call_count, 2)

    def test_cli_exhaustion_preserves_existing_output_without_traceback(self):
        script = (
            "import runpy, sys; from unittest.mock import patch; "
            "from urllib3.exceptions import NewConnectionError; "
            "sys.argv = [sys.argv[1], 'otel', '--output', sys.argv[2]]\n"
            "with patch('urllib3.connectionpool.HTTPSConnectionPool._make_request', "
            "side_effect=NewConnectionError(None, 'offline')) as request, "
            "patch('urllib3.util.retry.time.sleep'):\n"
            "    try:\n"
            "        runpy.run_path(sys.argv[0], run_name='__main__')\n"
            "    finally:\n"
            "        assert request.call_count == 4\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.jsonl"
            output.write_text("existing data")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(PACKAGE_ROOT / "prepare_decider_dataset.py"),
                    str(output),
                ],
                cwd=PACKAGE_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(OTEL_URL, result.stderr)
            self.assertIn("offline", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(output.read_text(), "existing data")


if __name__ == "__main__":
    unittest.main()
