# `inference/mlx/llm` — `mlx_lm.server` serving (OTel-LLM-E4B-IT)

## Overview & when to use

Serves **`OTel-LLM-E4B-IT`** as an MLX 4-bit checkpoint via `mlx_lm.server` (MLX's OpenAI-compatible HTTP server). The client `infer_llm_mlx.py` interacts with `POST /v1/chat/completions` and polls `GET /health`.

Use this directory to run the OTel context-grounded generation model locally on Apple silicon (retrieval, reranking, and generation on a single machine).

> **Format note:** The published `OTel-LLM-E4B-IT` checkpoint contains PyTorch `.bin` shards (31 GB). [`convert_otel_e4b.py`](convert_otel_e4b.py) rewrites these into bf16 safetensors, quantizes to 4-bit (3.9 GB), and retains only the text model. 

> **Model note:** This is a **context-grounded** model trained to answer from retrieved passages. Without a passage, it will either abstain or answer from memory (test ungrounded behavior via `--context ''`).

## Convert (once)

```bash
cd inference/mlx/llm && source ../.env_mlx/bin/activate
pip install torch safetensors                      # conversion only; not in requirements.txt
python convert_otel_e4b.py --mlx_path ./OTel-LLM-E4B-IT-4bit
```

This step downloads the source checkpoint to `$HF_HOME` (~31 GB, requires ~11 GB peak RAM for the largest shard), converts, quantizes, and cleans up temp files. Use `--no_quantize` to output bf16 MLX weights (8 GB).

## Environment & secrets

`dev.env` is symlinked to the repo-root. `OTel-LLM-E4B-IT` is ungated.

```bash
export HF_HOME=/path/to/hf_cache       # source checkpoint cache (31 GB) and mlx-community models
export OUTPUT_DIR=/path/to/outputs     # inference artifacts
```

## Serve

```bash
bash serve.sh ./OTel-LLM-E4B-IT-4bit 8080
# Equivalent: python -m mlx_lm server --model ./OTel-LLM-E4B-IT-4bit --host 127.0.0.1 --port 8080 --chat-template-args '{"enable_thinking": false}'
```

`enable_thinking: false` prevents Gemma 4's `<|think|>` token from entering the system turn. The server process requires ~4.3 GB for the weights and prompt cache. The `--model` requested by the client **must** match the server's starting argument.

## Client / smoke command

```bash
python infer_llm_mlx.py --out $OUTPUT_DIR/inference_llm_mlx/chat_e4b.json
python infer_llm_mlx.py --context ''                              # ungrounded question
python infer_llm_mlx.py --prompt "..." --context "$(cat passage.txt)"
```

## Expected output (Apple M4, 24 GB)

```text
endpoint      : http://127.0.0.1:8080/v1/chat/completions
model         : ./OTel-LLM-E4B-IT-4bit
prompt_tokens : 127
output_tokens : 51
latency       : 1.89s (~26.9 tok/s end-to-end, warm prompt cache)
--- generated text ---
In Mode 2, the sensing window is used to monitor the availability of radio resources. If the
number of candidate resources within the window falls below 20%, the threshold for excluding
resources is increased by 3 dB and the procedure is repeated.
```

**Memory Profiling:** The `mlx_lm.server` process idles around 4.3–5.4 GB RSS (unified memory, no separate VRAM).

## Measured — Apple M4, 24 GB

| Model | Disk | Peak Memory | Prompt tok/s | Generation tok/s | Notes |
|---|---|---|---|---|---|
| `OTel-LLM-E4B-IT` 4-bit (in-process) | 3.9 GB | 4.5 GB | 456 | 35.2 | Fast and lightweight. |
| `OTel-LLM-E4B-IT` 4-bit (server) | - | 5.4 GB | - | ~27 (end-to-end) | Default for this leaf. |
| `mlx-community/Qwen3.8-27B-4bit` | 15 GB | 15.5 GB | 4 | 6.3 | Parity model. Use single concurrent request. |

## Arguments

### `serve.sh` / `mlx_lm.server`

| Flag | Default | Meaning |
|---|---|---|
| positional 1 | `./OTel-LLM-E4B-IT-4bit` | MLX model dir or Hub id |
| positional 2 | `8080` | port |
| `--chat-template-args` | `{"enable_thinking": false}` | Drop to allow thinking |
| `--prompt-cache-size` | server default | Lower for long contexts on 16 GB machines |

### `infer_llm_mlx.py`

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8080` | server address |
| `--model` | `./OTel-LLM-E4B-IT-4bit` | **Must equal the server's model argument** |
| `--prompt` | built-in | user question |
| `--context` | built-in passage | Grounding text. Use `''` for none |
| `--system` | `None` | Optional system prompt |
| `--max_tokens` | `128` | Generation limit |
| `--temperature` / `--top_p` | `0.0` / `1.0` | Greedy by default |
| `--out` | `None` | Path to write the JSON reference artifact |

## Output

Prints endpoint, token usage, latency, and the generated text.
`--out` writes `{"request": ..., "response": <raw chat.completion>, "latency_seconds": ...}`:

```
$OUTPUT_DIR/inference_llm_mlx/chat_e4b.json
```

## Hardware support

- **Apple M4, 24 GB (macOS 26.6, Metal)**: Verified. Any M-series Mac with ≥ 8 GB can run the E4B model. The 27B model requires ≥ 24 GB.

## Notes & quirks

- **Autonomous thinking:** Even with `enable_thinking: false`, the model may open a `<|channel>thought` block for hard queries. The server returns this in a separate `reasoning` field.
- **Abstention behavior:** OTel-Safety tuning prompts abstention on insufficient context, but without context it often answers from memory. Treat it as a RAG model.
- **Prompt cache:** The prompt cache operates per server process, so latency stats on the client refer to the warm second request.
- **LoRA Support:** Pass `--adapter-path` to `mlx_lm.server` to load a LoRA adapter dynamically.
