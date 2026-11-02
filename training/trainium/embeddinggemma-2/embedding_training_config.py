"""Shared constants for the validated EmbeddingGemma-2 Trainium recipe."""

DEFAULT_MODEL = "google/embeddinggemma-2"
DEFAULT_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

# Stable hash buckets keep related examples together while targeting a
# deterministic 90% train, 5% validation, and 5% test split.
SPLIT_BUCKET_COUNT = 1000
TRAIN_SPLIT_CUTOFF = 900
VALIDATION_SPLIT_CUTOFF = 950
SPLIT_HASH_PREFIX_LENGTH = 8
MAX_EVALUATION_QUERIES_PER_SPLIT = 512

DEPENDENCY_NAMES = (
    "torch",
    "torch-neuronx",
    "neuronx-cc",
    "nki",
    "transformers",
    "sentence-transformers",
    "datasets",
    "accelerate",
)

COMPILE_CACHE_DIRECTORIES = (
    ("TORCH_NEURONX_DEBUG_DIR", "debug"),
    ("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
    ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
    ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
)

QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "fsspec",
)
