"""OpenRLHF launcher — assembles and execs the right deepspeed / ray job submit command line (see readme_openrlhf.md)."""

import argparse
import logging
import os
import shlex
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("openrlhf-launcher")

# GRPO/RLOO/REINFORCE++ are all train_ppo_ray with a different advantage estimator.
ESTIMATORS = {"grpo": "group_norm", "dr_grpo": "dr_grpo", "reinforce": "reinforce",
              "reinforce_baseline": "reinforce_baseline", "rloo": "rloo"}


def parse_args():
    p = argparse.ArgumentParser(description="Launcher for OpenRLHF SFT/DPO/PPO/GRPO training")
    p.add_argument("--mode", choices=["sft", "dpo", "ppo", "grpo"], default="grpo")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Policy/actor model: HF repo id or local path.")
    p.add_argument("--dataset", default="data/OTel_LLM_sample_10.jsonl",
                   help="HF dataset id or local JSONL path (OpenRLHF loads both). "
                        "Default is the shipped chat sample, ready for --mode sft.")
    p.add_argument("--output_dir", default="./checkpoint/openrlhf-run")
    p.add_argument("--num_gpus", type=int, default=8, help="GPUs on this node.")
    p.add_argument("--input_key", default="messages",
                   help="SFT/PPO prompt field. Default `messages` matches OTel_LLM_sample_10.jsonl.")
    p.add_argument("--output_key", default=None, help="SFT target field (unset with chat templates).")
    p.add_argument("--label_key", default="answer",
                   help="PPO/GRPO ground-truth field. Matches data/otel_rl.jsonl from "
                        "prepare_data_openrlhf.py.")
    p.add_argument("--apply_chat_template", action="store_true", default=True)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--max_samples", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=128, help="Global train batch size.")
    p.add_argument("--micro_batch_size", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6, help="SFT/DPO: adam.lr; RL: actor lr.")
    p.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--packing_samples", action="store_true", default=True)
    p.add_argument("--flash_attn", action="store_true", default=True)
    p.add_argument("--reward_model", default=None, help="PPO: reward model repo id/path.")
    p.add_argument("--reward_func", default=None,
                   help="Path to a reward_func.py (--reward.remote_url). Required for GRPO "
                        "unless --reward_model is set.")
    p.add_argument("--critic_lr", type=float, default=9e-6)
    p.add_argument("--kl_coef", type=float, default=0.01, help="0 disables the reference model.")
    p.add_argument("--rollout_batch_size", type=int, default=1024)
    p.add_argument("--n_samples_per_prompt", type=int, default=8,
                   help="Must be >1 for GRPO group normalization to mean anything.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--vllm_num_engines", type=int, default=4)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--colocate_all", action="store_true", default=True,
                   help="Hybrid Engine: share GPUs across actor/ref/critic/vLLM.")
    p.add_argument("--ray_address", default="http://127.0.0.1:8265", help="Ray dashboard address.")
    p.add_argument("--working_dir", default=".", help="Shipped to Ray workers as the runtime env.")
    p.add_argument("--dry_run", action="store_true", help="Print the command, do not run it.")
    return p.parse_args()


def build_deepspeed_cmd(args):
    """SFT and DPO run through `deepspeed --module`, using the --model.* arg namespace."""
    module = "openrlhf.cli.train_sft" if args.mode == "sft" else "openrlhf.cli.train_dpo"
    cmd = [
        "deepspeed", "--module", module,
        "--model.model_name_or_path", args.model_name,
        "--data.dataset", args.dataset,
        "--data.max_len", str(args.max_len),
        "--data.max_samples", str(args.max_samples),
        "--train.batch_size", str(args.batch_size),
        "--train.micro_batch_size", str(args.micro_batch_size),
        "--train.max_epochs", str(args.max_epochs),
        "--adam.lr", str(args.learning_rate),
        "--ds.zero_stage", str(args.zero_stage),
        "--ds.param_dtype", "bf16",
        "--ckpt.output_dir", args.output_dir,
        "--ckpt.save_steps", "-1",
        "--logger.logging_steps", "1",
        "--eval.steps", "-1",
        "--model.gradient_checkpointing_enable",
    ]
    if args.mode == "sft":
        cmd += ["--data.input_key", args.input_key]
        if args.output_key:
            cmd += ["--data.output_key", args.output_key]
    else:
        cmd += ["--data.chosen_key", "chosen", "--data.rejected_key", "rejected"]
    if args.apply_chat_template:
        cmd += ["--data.apply_chat_template"]
    if args.packing_samples:
        cmd += ["--ds.packing_samples"]
    if args.flash_attn:
        cmd += ["--ds.attn_implementation", "flash_attention_2"]
    return cmd


def build_ray_cmd(args):
    """PPO and GRPO run through `ray job submit`, using the --actor.*/--critic.* namespace."""
    train = [
        "python3", "-m", "openrlhf.cli.train_ppo_ray",
        "--actor.model_name_or_path", args.model_name,
        "--actor.num_nodes", "1",
        "--actor.num_gpus_per_node", str(args.num_gpus),
        "--ref.num_nodes", "1",
        "--ref.num_gpus_per_node", str(args.num_gpus),
        "--data.prompt_dataset", args.dataset,
        "--data.input_key", args.input_key,
        "--data.label_key", args.label_key,
        "--data.max_len", str(args.max_len),
        "--data.max_samples", str(args.max_samples),
        "--rollout.batch_size", str(args.rollout_batch_size),
        "--rollout.n_samples_per_prompt", str(args.n_samples_per_prompt),
        "--rollout.max_new_tokens", str(args.max_new_tokens),
        "--train.batch_size", str(args.batch_size),
        "--train.micro_batch_size", str(args.micro_batch_size),
        "--train.max_epochs", str(args.max_epochs),
        "--actor.adam.lr", str(args.learning_rate),
        "--algo.kl.init_coef", str(args.kl_coef),
        "--ds.zero_stage", str(args.zero_stage),
        "--ds.param_dtype", "bf16",
        "--vllm.num_engines", str(args.vllm_num_engines),
        "--vllm.tensor_parallel_size", str(args.vllm_tensor_parallel_size),
        "--vllm.gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm.sync_backend", "nccl",
        "--vllm.enforce_eager",
        "--ckpt.output_dir", args.output_dir,
        "--ckpt.save_hf",
        "--logger.logging_steps", "1",
        "--eval.steps", "-1",
        "--actor.gradient_checkpointing_enable",
    ]
    # PPO learns a value function, so it needs a critic; GRPO estimates the baseline from
    # the sample group instead and must not be given one.
    if args.mode == "ppo":
        train += ["--critic.num_nodes", "1",
                  "--critic.num_gpus_per_node", str(args.num_gpus),
                  "--critic.adam.lr", str(args.critic_lr)]
    else:
        train += ["--algo.advantage.estimator", ESTIMATORS["grpo"]]

    if args.reward_model:
        train += ["--reward.model_name_or_path", args.reward_model,
                  "--reward.num_nodes", "1",
                  "--reward.num_gpus_per_node", str(args.num_gpus),
                  "--reward.normalize_enable"]
    elif args.reward_func:
        train += ["--reward.remote_url", args.reward_func]

    if args.apply_chat_template:
        train += ["--data.apply_chat_template"]
    if args.packing_samples:
        train += ["--ds.packing_samples"]
    if args.flash_attn:
        train += ["--ds.attn_implementation", "flash_attention_2"]
    if args.colocate_all:
        train += ["--train.colocate_all", "--vllm.enable_sleep", "--ds.enable_sleep"]

    runtime_env = '{"working_dir": "%s"}' % args.working_dir
    return ["ray", "job", "submit", "--address", args.ray_address,
            "--runtime-env-json", runtime_env, "--"] + train


def main():
    args = parse_args()

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN not set (dev.env missing?); gated models will fail to download.")

    if args.mode in ("ppo", "grpo") and not (args.reward_model or args.reward_func):
        logger.error("%s needs a reward signal: pass --reward_model or --reward_func.", args.mode)
        return 2
    if args.mode == "grpo" and args.n_samples_per_prompt < 2:
        logger.error("GRPO needs --n_samples_per_prompt > 1 to form a comparison group.")
        return 2

    build = build_deepspeed_cmd if args.mode in ("sft", "dpo") else build_ray_cmd
    cmd = build(args)

    logger.info("Mode: %s", args.mode)
    logger.info("Command:\n%s", " \\\n  ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        logger.info("Dry run; nothing executed.")
        return 0

    logger.info("Launching. Ensure the Ray head node is up first for ppo/grpo.")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
