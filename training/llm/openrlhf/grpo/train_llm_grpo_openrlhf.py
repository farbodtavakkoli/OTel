import os
import sys
import time

import utils


def openrlhf_command(args, run_dir: str, train_file: str, lora_regex: str, reward_url: str) -> list:
    save_freq = args.steps if args.save_freq is None else args.save_freq
    cmd = [
        sys.executable, "-m", "openrlhf.cli.train_ppo_ray",
        "--actor.model_name_or_path", args.model,
        "--actor.num_nodes", "1", "--actor.num_gpus_per_node", str(args.actor_gpus),
        "--ref.num_nodes", "1", "--ref.num_gpus_per_node", str(args.actor_gpus),
        "--data.prompt_dataset", train_file, "--data.input_key", "prompt", "--data.label_key", "label",
        "--data.apply_chat_template",
        "--data.max_len", str(args.max_len), "--data.max_samples", str(args.prompt_batch_size * args.steps),
        "--rollout.batch_size", str(args.prompt_batch_size), "--rollout.n_samples_per_prompt", str(args.group_size),
        "--rollout.max_new_tokens", str(args.max_tokens), "--rollout.temperature", str(args.temperature),
        "--rollout.top_p", "1.0", "--rollout.micro_batch_size", str(args.micro_rollout_bs),
        "--train.batch_size", str(args.prompt_batch_size * args.group_size),  # one optimizer step per rollout batch
        "--train.micro_batch_size", str(args.micro_train_bs), "--train.max_epochs", "1", "--train.num_episodes", "1",
        "--train.seed", str(args.seed),
        "--actor.adam.lr", str(args.learning_rate), "--algo.kl.init_coef", "0", "--algo.advantage.estimator", "group_norm",
        "--ds.zero_stage", "2", "--ds.param_dtype", "bf16", "--ds.attn_implementation", args.attn,
        "--ds.lora.rank", str(args.lora_r), "--ds.lora.alpha", str(args.lora_alpha),
        "--ds.lora.target_modules", f"re:{lora_regex}",  # `re:` = regex targets (patch_openrlhf_lora_sync.py)
        "--actor.gradient_checkpointing_enable", "--actor.freeze_visual_encoder",
        "--vllm.num_engines", str(args.vllm_engines), "--vllm.tensor_parallel_size", str(args.vllm_tp),
        "--vllm.gpu_memory_utilization", str(args.gpu_memory_utilization), "--vllm.sync_backend", "nccl",
        "--reward.remote_url", reward_url,
        "--ckpt.output_dir", os.path.join(run_dir, "lora_adapter"),  # final PEFT save
        "--ckpt.path", os.path.join(run_dir, "ds_ckpt"), "--ckpt.save_hf", "--ckpt.disable_ds",
        "--ckpt.save_steps", str(save_freq), "--ckpt.max_num", "1",
        "--logger.logging_steps", "1", "--eval.steps", "-1",
    ]
    if args.enforce_eager:
        cmd.append("--vllm.enforce_eager")
    return utils.apply_overrides(cmd, args.override)


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    os.makedirs(run_dir, exist_ok=True)
    if args.judge_rubric:
        os.environ["GRPO_JUDGE_RUBRIC"] = args.judge_rubric  # read by judge_reward at import

    cfg = utils.load_model_config(args.model)
    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    train_file = utils.data_file(args.data, run_dir, args.dry_run or args.export_only)
    reward = utils.RewardServer(args.reward_port, os.path.join(run_dir, "reward_requests.jsonl"), args.reward_concurrency)
    cmd = openrlhf_command(args, run_dir, train_file, lora_regex, reward.url)
    env = utils.subprocess_env(args)
    n_gpus = args.actor_gpus + args.vllm_engines * args.vllm_tp

    if args.dry_run:
        utils.print_dry_run(cmd, env, run_dir, args.ray_port, n_gpus, reward)
        return

    if not args.export_only:
        utils.record_command(cmd, run_dir)
        print("[grpo] versions:", utils.write_versions(os.path.join(run_dir, "versions.txt")), flush=True)
        started_ray = utils.start_ray(args.ray_port, n_gpus)
        reward.start()
        t0 = time.time()
        try:
            rc = utils.run_openrlhf(cmd, env, run_dir)
        finally:
            reward.stop()
            if started_ray:
                utils.stop_ray()
        print(f"[grpo] openrlhf exited rc={rc} after {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            raise SystemExit(rc)

    if args.no_export:
        return
    adapter_dir = os.path.join(run_dir, "lora_adapter")
    utils.export_lora_adapter(utils.latest_adapter(run_dir), adapter_dir)
    if args.verify:
        prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
        proof = utils.verify_adapter(args.model, adapter_dir, prompts)
        print(f"[verify] proof written to {proof}", flush=True)


if __name__ == "__main__":
    main()
