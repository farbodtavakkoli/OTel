import json
import os
import time

import utils

HERE = os.path.dirname(os.path.abspath(__file__))


def verl_overrides(args, lora_regex: str, tied_embeddings: bool, run_dir: str) -> list:
    ov = [
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192",
        f"actor_rollout_ref.model.target_modules='{lora_regex}'",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.rollout.seed={args.seed}",
        f"data.seed={args.seed}",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        "reward.reward_manager.source=importlib",
        f"reward.reward_manager.module.path={HERE}/utils.py",
        "reward.reward_manager.name=TruncationAwareRewardManager",
        # verl's v1 trainer imports `transfer_queue`, which is not published on PyPI; the classic
        # trainer is also what the custom reward manager above hooks into.
        "trainer.use_v1=False",
        f"trainer.rollout_data_dir={run_dir}/rollouts",
        f"trainer.experiment_name={os.path.basename(run_dir)}",
        f"hydra.run.dir={run_dir}/hydra",
    ]
    if not tied_embeddings:
        # FSDP2 shards an untied lm_head as its own unit; the fused log-prob kernel then
        # sees a DTensor and fails ("mixed torch.Tensor and DTensor").
        ov.append("actor_rollout_ref.model.use_fused_kernels=False")
    return ov + args.override


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    os.makedirs(run_dir, exist_ok=True)

    cfg = utils.load_model_config(args.model)
    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    overrides = verl_overrides(args, lora_regex, bool(cfg.get("tie_word_embeddings", False)), run_dir)

    kwargs = dict(
        backend="verl",
        model_path=args.model,
        data_path=args.data,
        ckpt_output_dir=run_dir,
        n_gpus=args.n_gpus,
        group_size=args.group_size,
        prompt_batch_size=args.prompt_batch_size,
        num_iterations=1,
        n_train=args.steps * args.prompt_batch_size,
        n_val=0,
        max_tokens=args.max_tokens,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        reward_fn=os.path.join(HERE, "utils.py"),
        reward_type="custom",
        data_source="telecom",
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_dr_grpo=True,
    )
    with open(os.path.join(run_dir, "lora_grpo_kwargs.json"), "w") as f:
        json.dump(kwargs, f, indent=1)
    if args.dry_run:
        print(json.dumps({"kwargs": kwargs, "overrides": overrides}, indent=1))
        return

    if not args.export_only:
        utils.write_versions(os.path.join(run_dir, "versions.txt"))
        utils.start_ray(args.ray_port, args.n_gpus)
        utils.patch_training_hub_popen(overrides, os.path.join(run_dir, "command.txt"))
        from training_hub import lora_grpo

        t0 = time.time()
        result = lora_grpo(**kwargs)
        wall = time.time() - t0
        with open(os.path.join(run_dir, "lora_grpo_result.json"), "w") as f:
            json.dump(dict(result, wall_s=wall), f, indent=1, default=str)
        print(f"[grpo] finished in {wall:.0f}s; reward_history={result.get('reward_history')}", flush=True)

    if args.no_export:
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
