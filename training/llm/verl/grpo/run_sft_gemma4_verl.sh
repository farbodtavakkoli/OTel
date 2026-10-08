#!/usr/bin/env bash
# LoRA SFT warm-up launcher for verl (docs/FINDINGS.md §4, experiments E3/E4/E5), single GPU, runs
# INSIDE the verl-grpo-bench container. Same engine as the GRPO actor (FSDP2, PEFT LoRA r=16/alpha=32 on the
# text-layer q/k/v/o/gate/up/down projections, fp32 master weights, bf16 compute, sdpa, padded to the
# micro-batch max), so the exported adapter has exactly the module names the GRPO worker expects.
#
# Data: the verl GRPO parquet (the schema train_llm_grpo_verl.py prepares); utils.GrpoSFTDataset builds
#   <rollout prompt incl. empty thought channel> + reference + <turn|>   with loss on the reference only.
#
# Paths: GRPO_SHARED_DIR is the shared storage root, mounted at an identical path inside the container
# (see train_llm_grpo_verl.py). <container> below is the verl container name.
#
# Usage (from the host):
#   docker exec -e SFT_NAME=E3_sft -e SOURCES='[identity]' -e EPOCHS=2 -e TRAIN_BS=3 \
#       <container> bash /workspace/verl_folder/run_sft_gemma4_verl.sh
#   docker exec -e SFT_NAME=E4_sft -e EPOCHS=1 -e TRAIN_BS=6 <container> bash /workspace/verl_folder/run_sft_gemma4_verl.sh
#   docker exec -e SFT_NAME=E5_sft2 -e EPOCHS=1 -e LR=5e-5 -e ADAPTER_PATH=/path/to/lora_adapter \
#       <container> bash /workspace/verl_folder/run_sft_gemma4_verl.sh
# Afterwards: train_llm_grpo_verl.py --export-only --output-dir $SFT_DIR --model <base> exports $SFT_DIR/lora_adapter
set -euo pipefail

SFT_NAME=${SFT_NAME:-sft_smoke}
GRPO_SHARED_DIR=${GRPO_SHARED_DIR:-./shared}
MODEL_PATH=${MODEL_PATH:-${GRPO_SHARED_DIR}/models/google--gemma-4-31B-it}
TRAIN_PARQUET=${TRAIN_PARQUET:-${GRPO_SHARED_DIR}/runs/verl/data/grpo_train_810.parquet}
VAL_PARQUET=${VAL_PARQUET:-${GRPO_SHARED_DIR}/runs/verl/data/grpo_eval_90.parquet}   # "" = no val loss
RUNS_ROOT=${RUNS_ROOT:-${GRPO_SHARED_DIR}/runs/verl}
SOURCES=${SOURCES:-}                 # Hydra list, e.g. '[identity]'; empty = all sources
EPOCHS=${EPOCHS:-1}
TRAIN_BS=${TRAIN_BS:-6}              # global batch (sequences) per optimizer step; 810 = 135 x 6, 90 = 30 x 3
MICRO_BS=${MICRO_BS:-1}              # sequences per forward/backward (grad accumulation = TRAIN_BS / MICRO_BS)
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-0}
MAX_LEN=${MAX_LEN:-4096}
SEED=${SEED:-42}
ADAPTER_PATH=${ADAPTER_PATH:-}       # PEFT adapter dir to continue from (E5 second SFT); empty = fresh LoRA
N_GPUS=${N_GPUS:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

export SFT_DIR="${RUNS_ROOT}/${SFT_NAME}"
mkdir -p "${SFT_DIR}"
export PYTHONPATH="/workspace/verl_folder:/workspace/grpo:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${SFT_DIR}"

LORA_TARGET='.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'

CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" -m verl.trainer.sft_trainer
  data.train_files="${TRAIN_PARQUET}"
  data.val_files="${VAL_PARQUET:-null}"
  data.train_batch_size="${TRAIN_BS}"
  data.micro_batch_size_per_gpu="${MICRO_BS}"
  data.use_dynamic_bsz=False
  data.max_length="${MAX_LEN}"
  data.pad_mode=no_padding
  data.truncation=error
  data.num_workers=2
  data.custom_cls.path=${HERE:-$(dirname "$0")}/utils.py
  data.custom_cls.name=GrpoSFTDataset
  model.path="${MODEL_PATH}"
  model.lora_rank=16
  model.lora_alpha=32
  "model.target_modules='${LORA_TARGET}'"
  model.use_remove_padding=False
  +model.override_config.attn_implementation=sdpa
  model.enable_gradient_checkpointing=True
  engine.strategy=fsdp2
  optim.lr="${LR}"
  optim.lr_warmup_steps="${WARMUP_STEPS}"
  optim.lr_scheduler_type=constant
  optim.clip_grad=1.0
  checkpoint.save_contents='[model]'
  +checkpoint.save_lora_only=True
  trainer.total_epochs="${EPOCHS}"
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.resume_mode=disable
  trainer.seed="${SEED}"
  trainer.logger='[console]'
  trainer.project_name=grpo_sft
  trainer.experiment_name="sft_gemma4_31b_${SFT_NAME}"
  trainer.default_local_dir="${SFT_DIR}/checkpoints"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.nnodes=1
)
if [ -n "${SOURCES}" ]; then CMD+=("+data.sources=${SOURCES}"); fi
if [ -n "${ADAPTER_PATH}" ]; then CMD+=("model.lora_adapter_path=${ADAPTER_PATH}"); fi
if [ -n "${EXTRA_ARGS}" ]; then CMD+=(${EXTRA_ARGS}); fi

echo "=== $(date -Is) SFT_NAME=${SFT_NAME} SFT_DIR=${SFT_DIR}" | tee -a "${SFT_DIR}/command.txt"
printf '%q ' "${CMD[@]}" | tee -a "${SFT_DIR}/command.txt"; echo | tee -a "${SFT_DIR}/command.txt"
python3 -c "import torch, verl, transformers, peft; print('versions torch', torch.__version__, 'verl', verl.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" 2>/dev/null | tee "${SFT_DIR}/versions.txt"

T0=$(date +%s)
"${CMD[@]}" 2>&1 | tee "${SFT_DIR}/train.log"
RC=${PIPESTATUS[0]}
echo "=== SFT EXIT rc=${RC} wall_s=$(( $(date +%s) - T0 )) $(date -Is)" | tee -a "${SFT_DIR}/train.log"
exit "${RC}"
