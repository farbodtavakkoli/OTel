#!/bin/bash
# Shared judge for every stack: Qwen3.8-27B-FP8 on GPU 0 via vLLM. Usage: judge_server.sh start|stop|health
set -euo pipefail

IMAGE="${JUDGE_IMAGE:-grpo-base:mi355}"
GRPO_PLATFORM="${GRPO_PLATFORM:-rocm}"
GRPO_SHARED_DIR="${GRPO_SHARED_DIR:-./shared}"
MODEL_DIR="${JUDGE_MODEL_DIR:-$GRPO_SHARED_DIR/models/unsloth--Qwen3.8-27B-FP8}"
MODEL_DIR="$(readlink -m "$MODEL_DIR")"   # docker -v needs an absolute host path
PORT="${JUDGE_PORT:-8100}"
JUDGE_QUANT="${JUDGE_QUANT-fp8}"
GPU="${JUDGE_GPU:-0}"
NAME=grpo-judge
LOG_DIR="$GRPO_SHARED_DIR/benchmarks/logs"
mkdir -p "$LOG_DIR"

case "${1:-start}" in
  stop)   docker rm -f $NAME >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; exit 0 ;;
  health) curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" && echo || { echo "judge not up"; exit 1; }; exit 0 ;;
  start)  ;;
  *) echo "usage: $0 [start|stop|health]"; exit 2 ;;
esac

docker rm -f $NAME >/dev/null 2>&1 || true
# rocm passes /dev/dri whole and restricts with the visibility vars; cuda exposes one GPU directly.
if [ "$GRPO_PLATFORM" = cuda ]; then
  GPU_ARGS=(--gpus "device=$GPU")
else
  GPU_ARGS=(--device /dev/kfd --device /dev/dri --group-add video
            -e ROCR_VISIBLE_DEVICES=$GPU)
fi
docker run -d --name $NAME \
  "${GPU_ARGS[@]}" \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE --ipc host --shm-size 32g \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_V1=1 \
  -v "$MODEL_DIR":/models/judge:ro \
  -p 127.0.0.1:$PORT:8000 \
  --entrypoint python "$IMAGE" -m vllm.entrypoints.openai.api_server \
    --model /models/judge --served-model-name judge \
    --dtype auto ${JUDGE_QUANT:+--quantization $JUDGE_QUANT} \
    --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 --limit-mm-per-prompt '{"image":0,"video":0}' \
    --port 8000 > /dev/null
echo "started $NAME on GPU $GPU, waiting for /v1/models on port $PORT ..."
for i in $(seq 1 120); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -q '"judge"'; then
    echo "judge healthy after ~$((i*5))s"; docker logs $NAME 2>&1 | grep -iE 'kv cache|Maximum concurrency' | tail -2; exit 0
  fi
  if ! docker ps -q -f name=$NAME | grep -q .; then echo "judge container exited:"; docker logs $NAME 2>&1 | tail -30; exit 1; fi
  sleep 5
done
echo "judge did not become healthy in 600s"; docker logs $NAME 2>&1 | tail -30; exit 1
