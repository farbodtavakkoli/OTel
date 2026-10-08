import json
import os
import time

import utils


def grpo_config_kwargs(args, grad_accum=None, ds_config=None, dry=False):
    grad_accum = grad_accum or (args.prompt_batch_size * args.group_size) // args.per_device_batch
    return dict(
        output_dir=os.path.abspath(args.output_dir),
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        steps_per_generation=grad_accum,          # one generation batch == one optimizer step
        num_generations=args.group_size,
        num_iterations=1,
        max_completion_length=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=args.steps,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        log_completions=False,
        shuffle_dataset=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.server_host,
        vllm_server_port=args.server_port,
        vllm_server_timeout=600.0,
        deepspeed=ds_config if not dry else (f"build_deepspeed_config(zero_stage={args.zero_stage})" if args.zero_stage > 0 else None),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


def train(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    import utils
    import judge_reward

    run_dir = os.path.abspath(args.output_dir)
    adapter_dir = os.path.join(run_dir, "adapter")
    prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
    if args.verify_only:
        print(f"[verify] proof written to {utils.verify_adapter(args.model, adapter_dir, prompts)}", flush=True)
        return

    utils.patch_trl_vllm_client()
    assert torch.cuda.device_count() >= 1, "no GPU visible to the trainer"
    n_completions = args.prompt_batch_size * args.group_size
    grad_accum = n_completions // args.per_device_batch
    assert grad_accum * args.per_device_batch == n_completions, "prompt-batch-size * group-size must be a multiple of per-device-batch"
    assert 0 <= args.warmup_steps < args.steps, "need warmup-steps < steps"

    ds = utils.load_grpo_dataset(args.data)
    print(f"[grpo] dataset rows={len(ds)} columns={ds.column_names}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template, "tokenizer has no chat template"
    cfg = utils.load_model_config(args.model)
    t0 = time.perf_counter()
    model = utils.auto_model_class(cfg).from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    print(f"[grpo] model class={type(model).__name__} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = True

    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                             task_type="CAUSAL_LM", target_modules=lora_regex)
    eos_ids = utils.eos_id_set(tokenizer, model)
    timers = utils.StepTimers(os.path.join(run_dir, "steps.jsonl"))
    reward_fn = utils.make_judge_reward(args.max_tokens, eos_ids, timers.timers)

    ds_config = utils.build_deepspeed_config(zero_stage=args.zero_stage) if args.zero_stage > 0 else None
    grpo_args = GRPOConfig(**grpo_config_kwargs(args, grad_accum, ds_config))
    trainer = GRPOTrainer(model=model, args=grpo_args, train_dataset=ds, processing_class=tokenizer,
                          reward_funcs=[reward_fn], peft_config=peft_config, callbacks=timers.callbacks())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[grpo] trainable params={trainable / 1e6:.1f}M of {total / 1e9:.2f}B", flush=True)
    timers.install(trainer)

    torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    trainer.train()
    train_wall = time.perf_counter() - t_train
    print(f"[grpo] train() wall={train_wall:.1f}s steps={trainer.state.global_step}", flush=True)

    base, n_match, n_lora, nonzero = utils.save_and_reload_adapter(trainer, adapter_dir)
    config = {
        "model": args.model, "dataset": args.data, "output_dir": run_dir, "label": args.label,
        "num_generations": args.group_size, "prompts_per_step": args.prompt_batch_size,
        "per_device_batch": args.per_device_batch, "max_completion_length": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "beta": args.beta, "learning_rate": args.learning_rate,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps - args.warmup_steps, "seed": args.seed, "zero_stage": args.zero_stage,
        "gradient_checkpointing": not args.no_gradient_checkpointing, "vllm_mode": "server",
        "vllm_server_host": args.server_host, "vllm_server_port": args.server_port, "server_mode": args.server_mode,
        "vllm_server_args": utils.server_args_record(args), "gradient_accumulation_steps": grad_accum,
        "generation_batch_size": n_completions, "lora_target_modules": lora_regex,
        "model_class": type(base).__name__, "eos_ids": sorted(eos_ids), "total_params": total,
        "deepspeed": ds_config, "trl_defaults_used": {
            "loss_type": grpo_args.loss_type, "epsilon": grpo_args.epsilon,
            "vllm_importance_sampling_correction": grpo_args.vllm_importance_sampling_correction,
            "vllm_importance_sampling_cap": grpo_args.vllm_importance_sampling_cap,
            "scale_rewards": grpo_args.scale_rewards, "mask_truncated_completions": grpo_args.mask_truncated_completions,
            "num_iterations": grpo_args.num_iterations, "optim": grpo_args.optim},
        "adapter_reload": {"matched": n_match, "lora_tensors": n_lora, "lora_B_nonzero": nonzero},
    }
    summary = utils.write_grpo_summary(os.path.join(run_dir, "summary.json"), config,
                                       utils.grpo_versions(ds_config is not None), timers, args.warmup_steps,
                                       trainer.state.global_step, train_wall, adapter_dir, trainable,
                                       judge_reward.RUBRIC_VERSION)
    print("[grpo] DONE", json.dumps({k: summary[k] for k in ("steps_total", "steps_measured", "peak_mem_gb",
                                                            "reward_first", "reward_last")}), flush=True)
    if args.verify:
        del trainer, model, base
        torch.cuda.empty_cache()
        print(f"[verify] proof written to {utils.verify_adapter(args.model, adapter_dir, prompts)}", flush=True)


def main():
    args = utils.parse_args()
    if args.server_mode == "docker":
        utils.run_docker(args)
    else:
        utils.run_local(args, train, grpo_config_kwargs)


if __name__ == "__main__":
    main()
