"""GRPO/PPO launcher for verl — builds the Hydra override list for verl.trainer.main_ppo and execs it (see readme_verl.md)."""

import argparse
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="verl GRPO/PPO launcher")

    parser.add_argument("--algo", default="grpo", choices=["grpo", "ppo"],
                        help="grpo = group-relative, no critic; ppo = with a critic")
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-file", default="data/otel_train.parquet",
                        help="Parquet from prepare_data_verl.py (see readme_verl.md)")
    parser.add_argument("--val-file", default="data/otel_train.parquet",
                        help="Held-out parquet. The 10-row sample is reused for smoke eval.")

    parser.add_argument("--reward-file", default="reward_verl.py",
                        help="python file holding the reward function")
    parser.add_argument("--reward-fn", default="compute_score")

    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--strategy", default="fsdp2", choices=["fsdp", "fsdp2", "megatron"])
    parser.add_argument("--rollout-backend", default="vllm", choices=["vllm", "sglang", "hf"])
    parser.add_argument("--rollout-tp", type=int, default=2,
                        help="tensor-parallel size for the ROLLOUT engine only")

    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--rollout-n", type=int, default=5,
                        help="rollouts sampled per prompt; GRPO needs >1 to form a group")

    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--gpu-mem-util", type=float, default=0.6,
                        help="fraction of HBM the rollout engine may hold")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--test-freq", type=int, default=5)
    parser.add_argument("--project-name", default="verl_local")
    parser.add_argument("--experiment-name", default="grpo_run")
    parser.add_argument("--logger", default="console",
                        help="comma-separated: console,wandb,tensorboard,mlflow")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command without running it")
    return parser.parse_args()


def build_overrides(args):
    """Assemble the Hydra override list for verl.trainer.main_ppo."""
    loggers = ",".join(f'"{name.strip()}"' for name in args.logger.split(","))

    overrides = [
        f"algorithm.adv_estimator={args.algo}",
        f"data.train_files={args.train_file}",
        f"data.val_files={args.val_file}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.strategy={args.strategy}",
        f"actor_rollout_ref.actor.optim.lr={args.actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_loss_coef}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.entropy_coeff={args.entropy_coeff}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.name={args.rollout_backend}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_mem_util}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.strategy={args.strategy}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"custom_reward_function.path={args.reward_file}",
        f"custom_reward_function.name={args.reward_fn}",
        f"trainer.n_gpus_per_node={args.gpus_per_node}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.total_epochs={args.epochs}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.logger=[{loggers}]",
        "trainer.critic_warmup=0",
    ]

    # PPO trains a separate value network; GRPO replaces it with group statistics.
    if args.algo == "ppo":
        overrides += [
            f"critic.strategy={args.strategy}",
            f"critic.model.path={args.model_path}",
            f"critic.optim.lr={args.actor_lr * 10}",
            f"critic.ppo_micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        ]

    return overrides


def main():
    args = parse_args()

    if not os.getenv("HF_TOKEN"):
        logger.warning("HF_TOKEN not set (dev.env missing?); gated models will fail to download")
    if args.algo == "grpo" and args.rollout_n < 2:
        raise SystemExit("GRPO needs --rollout-n >= 2 to form a comparison group")

    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + build_overrides(args)

    logger.info("algo=%s model=%s strategy=%s rollout=%s",
                args.algo, args.model_path, args.strategy, args.rollout_backend)
    logger.info("launching: %s", " ".join(cmd))

    if args.dry_run:
        return

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
