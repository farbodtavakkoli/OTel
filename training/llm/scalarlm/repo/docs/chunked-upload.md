# Chunked Upload Protocol

## Overview
To bypass infrastructure-level request body limits (e.g., Cloudflare's 100MB cap) when uploading large training datasets, ScalarLM implements a session-based chunked upload protocol. This allows the Python SDK to split large tar archives into small, verifiable chunks across multiple HTTP requests.

## Protocol Sequence

### 1. Initialization (`POST /v1/megatron/upload/init`)
The client initiates a session by providing metadata about the full archive.
- **Request**: Total size, total SHA-256 hash, expected chunk size, and `train_args`.
- **Server**: Generates a unique `upload_id`, creates a staging directory, and writes a `manifest.json`.
- **Response**: Returns the `upload_id` and a list of any chunks already received (for resumption).

### 2. Chunk Upload (`POST /v1/megatron/upload/chunk`)
The client streams the archive in slices.
- **Request**: Raw bytes of a single chunk.
- **Headers**: `X-Upload-Id`, `X-Chunk-Index`, and `X-Chunk-Hash` (SHA-256 of the slice).
- **Server**:
    - Validates the `upload_id` and chunk size.
    - Verifies the `X-Chunk-Hash` against the received bytes.
    - Writes the chunk atomically to `/app/cray/upload_sessions/{upload_id}/chunks/{index}`.
- **Response**: Acknowledgment of the received chunk index.

### 3. Finalization (`POST /v1/megatron/upload/finalize`)
The client signals that all chunks have been sent.
- **Request**: The `upload_id`.
- **Server**:
    1. **Reconstruction**: Concatenates all chunks in index order into a temporary tar archive.
    2. **Verification**: Validates the total size and total SHA-256 hash against the manifest.
    3. **Job Preparation**: Replicates the standard training data pipeline:
       - Computes `dataset_hash`.
       - Injects hash into `train_args`.
       - Extracts the tar into the job directory.
       - Copies the `ml/` directory if missing.
    4. **Launch**: Calls the SLURM dispatcher to start the training job.
- **Response**: The standard `TrainResponse` containing `job_status` and `job_config`.

## Implementation Details

### State Management
Session state is stored on the local filesystem to remain consistent with ScalarLM's architecture:
- **Staging Root**: `/app/cray/upload_sessions/`
- **Session Folder**: `.../upload_sessions/{upload_id}/`
  - `manifest.json`: Stores metadata and timestamps.
  - `chunks/`: Directory containing zero-padded index files (e.g., `000001`).

### Reliability & Efficiency
- **Atomicity**: Chunks are written to `.tmp` files and renamed upon successful hash verification to prevent partial writes.
- **Retries**: The SDK implements exponential backoff with jitter for individual chunk failures.
- **Compression**: The archive is created as a gzipped tar (`.tar.gz`) by default to minimize the number of required HTTP requests.
- **Cleanup**: A periodic background task (reaper) removes abandoned upload sessions older than a configured TTL (e.g., 6 hours).

## Comparison: Single-POST vs. Chunked

| Feature | Legacy `/train` | Chunked Protocol |
| :--- | :--- | :--- |
| **Payload** | Single Multipart POST | Multiple Raw Body POSTs |
| **Limit** | Proxy Hard Cap (e.g. 100MB) | App-level Cap (e.g. 10GB) |
| **Resilience** | All-or-nothing | Per-chunk retries / resumption |
| **Complexity** | Simple | Session-based state tracking |
