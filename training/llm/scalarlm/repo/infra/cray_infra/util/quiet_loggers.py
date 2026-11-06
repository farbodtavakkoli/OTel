import logging

_NOISY_WARNING = (
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpx",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub.hf_api",
    "filelock",
    "urllib3",
    "urllib3.connectionpool",
    "asyncio",
    "aiohttp.access",
    "aiohttp.client",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "multipart",
    "python_multipart",
)


def quiet_noisy_loggers() -> None:
    for name in _NOISY_WARNING:
        logging.getLogger(name).setLevel(logging.WARNING)
