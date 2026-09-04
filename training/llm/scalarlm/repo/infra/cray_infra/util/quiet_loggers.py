"""
Turn down the volume on chatty third-party loggers.

We run the root logger at DEBUG on purpose — our own code benefits — but
libraries like httpcore log every HTTP state transition at DEBUG and httpx
logs every request at INFO, which drowns out anything useful on boot and
during HF Hub downloads. Downgrading these specific loggers to WARNING keeps
our own DEBUG output visible while silencing the noise.

Single source of truth: every process that configures logging (the API
server, the training job) calls quiet_noisy_loggers() immediately after
basicConfig.
"""

import logging

# Loggers we actively want silenced to WARNING.
_NOISY_WARNING = (
    # HTTP stack used by httpx / huggingface_hub / transformers downloads
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpx",
    # HF Hub / transformers download chatter — per-file progress + cache hits
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub.hf_api",
    "filelock",
    # Generic
    "urllib3",
    "urllib3.connectionpool",
    "asyncio",
    # aiohttp access logs are on every request, separate from our own logging
    "aiohttp.access",
    "aiohttp.client",
    # Matplotlib's font manager enumerates thousands of fonts at DEBUG
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    # multipart parsing in FastAPI's upload path
    "multipart",
    "python_multipart",
)


def quiet_noisy_loggers() -> None:
    """Clamp the level on each noisy logger so they only surface warnings+."""
    for name in _NOISY_WARNING:
        logging.getLogger(name).setLevel(logging.WARNING)
