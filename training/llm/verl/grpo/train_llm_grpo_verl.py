import os
import sys
import time

import utils


def verl_command(args, run_dir: str, train_file: str, val_file: str, lora_regex: str) -> list:
    save_freq = args.steps if args.save_freq is None else args.save_freq
    if args.remove_padding:
        pad = ["actor_rollout_ref.model.use_remove_padding=True",
               "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"]
    else:
        pad = ["actor_rollout_ref.model.use_remove_padding=False",
               "+actor_rollout_ref.model.override_config.attn_implementation=sdpa"]
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={args.prompt_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_tokens}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.shuffle=True",
        f"data.seed={args.seed}",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={args.model}",
        f"actor_rollout_ref.model.lora_rank={args.lora_r}",
        f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
        f"actor_rollout_ref.model.target_modules='{lora_regex}'",
        *pad,
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.prompt_batch_size}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_bs}",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={args.group_size}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.seed={args.seed}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_tokens}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=16384",
        "actor_rollout_ref.rollout.max_num_seqs=256",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={args.weight_bucket_mb}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.logprob_micro_bs}",
        # one batched judge call per step instead of verl's streaming per-sample reward
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=utils.BatchedJudgeAgentLoopManager",
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "reward.num_workers=1",
        "trainer.use_v1=False",  # the batched judge needs the classic trainer
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={args.steps}",
        "trainer.total_epochs=100",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.val_before_train={args.val_before_train}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "trainer.critic_warmup=0",
        "trainer.logger=[console]",
        "trainer.project_name=grpo_bench",
        f"trainer.experiment_name={os.path.basename(run_dir)}",
        f"trainer.default_local_dir={run_dir}/checkpoints",
    ]
    if args.lora_adapter_path:
        cmd.append(f"actor_rollout_ref.model.lora_adapter_path={os.path.abspath(args.lora_adapter_path)}")
    return utils.apply_overrides(cmd, args.override)


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    os.makedirs(run_dir, exist_ok=True)
    skip_data = args.dry_run or args.export_only

    cfg = utils.load_model_config(args.model)
    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    train_file = utils.data_file(args.data, run_dir, "train", skip_data)
    val_file = utils.data_file(args.val_data, run_dir, "val", skip_data) if args.val_data else train_file
    cmd = verl_command(args, run_dir, train_file, val_file, lora_regex)
    env = utils.subprocess_env(args, run_dir)

    if args.dry_run:
        utils.print_dry_run(cmd, env, run_dir)
        return

    if not args.export_only:
        utils.record_command(cmd, run_dir)
        print("[grpo] versions:", utils.write_versions(os.path.join(run_dir, "versions.txt")), flush=True)
        t0 = time.time()
        rc = utils.run_verl(cmd, env, run_dir)
        print(f"[grpo] verl exited rc={rc} after {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            raise SystemExit(rc)

    if args.no_export or (args.save_freq is not None and args.save_freq < 0 and not args.export_only):
        return
    adapter_dir = os.path.join(run_dir, "lora_adapter")
    utils.export_lora_adapter(utils.latest_actor_checkpoint(run_dir), adapter_dir, args.model,
                              args.lora_r, args.lora_alpha, lora_regex)
    if args.verify:
        prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
        proof = utils.verify_adapter(args.model, adapter_dir, prompts)
        print(f"[verify] proof written to {proof}", flush=True)


if __name__ == "__main__":
    main()
