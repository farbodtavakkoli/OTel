#!/usr/bin/env bash
# Serve an MLX checkpoint over an OpenAI-compatible HTTP API with mlx_lm.server.
#   bash serve.sh [model_dir_or_hf_id] [port]
# Default model: the OTel-LLM-E4B-IT 4-bit conversion produced by convert_otel_e4b.py.
# Gemma 4 emits a <|channel>thought block unless thinking is disabled through the chat
# template, so it is turned off here; drop --chat-template-args to get it back.
set -euo pipefail
MODEL=${1:-./OTel-LLM-E4B-IT-4bit}
PORT=${2:-8080}
exec python -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --chat-template-args '{"enable_thinking": false}' --log-level INFO
