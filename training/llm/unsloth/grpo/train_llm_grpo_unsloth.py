import gc
import json
import os
import time

import utils


def train(args, run_dir: str, derived: dict):
    # Unsloth must be imported before torch/transformers/trl: it patches them at import time.
    from unsloth import FastModel
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if not args.no_fast_inference:
        utils.patch_vllm_fork_lora_manager()
        if args.allow_gemma4_vllm:
            utils.patch_gemma4_fast_inference(args.max_num_seqs)

    t0 = time.time()
    load_kwargs = dict(model_name=args.model, max_seq_length=args.max_seq_length, dtype=torch.bfloat16,
                       load_in_4bit=args.load_in_4bit, fast_inference=not args.no_fast_inference)
    if not args.no_fast_inference:
        load_kwargs.update(max_lora_rank=args.lora_r, gpu_memory_utilization=args.gpu_memory_utilization,
                           max_num_seqs=args.max_num_seqs)
    model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    load_s = time.time() - t0
    utils.log.info(f"model loaded in {load_s:.1f}s; class={type(model).__name__}; "
                   f"vllm_engine={type(getattr(model, 'vllm_engine', None)).__name__}; "
                   f"mem_alloc_gb={torch.cuda.memory_allocated() / 1e9:.1f}")

    model = FastModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=utils.LORA_TARGET_MODULES,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    utils.log.info(f"LoRA attached: trainable={n_train:,} / {n_all:,}")
    utils.time_load_lora(model)

    ds = utils.load_grpo_dataset(args.data, args.seed)
    utils.log.info(f"dataset: {len(ds)} rows, columns={ds.column_names}")

    cfg = GRPOConfig(
        output_dir=run_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=derived["gradient_accumulation_steps"],
        steps_per_generation=derived["steps_per_generation"],   # one optimizer step per generation batch
        num_generations=args.group_size,
        max_completion_length=args.max_tokens,
        temperature=args.temperature, top_p=1.0, top_k=0,
        beta=args.beta,
        max_steps=args.steps,
        learning_rate=args.learning_rate, lr_scheduler_type="constant", warmup_ratio=0.0,
        weight_decay=0.0, optim=args.optim,
        logging_steps=1, save_strategy="no", seed=args.seed, bf16=True, report_to="none",
        use_vllm=not args.no_fast_inference,
        remove_unused_columns=False,
        log_completions=False,
        # Pin Unsloth's old-logprob chunking: left to autotune it collapses to one full-batch forward
        # from step 2 (OOM on a 31B model) and recompiles on every shape change.
        unsloth_grpo_mini_batch=args.logprob_chunks,
        unsloth_logit_chunk_multiplier=4,
    )
    reward_fn = utils.make_judge_reward(args.max_tokens, os.path.join(run_dir, "rollouts"))
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward_fn],
                          args=cfg, train_dataset=ds, callbacks=[utils.step_clock()])
    utils.time_trainer(trainer, model)

    torch.cuda.reset_peak_memory_stats()
    mem_before_gb = torch.cuda.memory_allocated() / 1e9
    t_train = time.time()
    trainer.train()
    train_wall = time.time() - t_train

    adapter_dir = os.path.join(run_dir, "final_model")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    utils.log.info(f"adapter saved to {adapter_dir}")

    from judge_reward import RUBRIC_VERSION
    result = {
        "args": vars(args), "generation_batch_size": derived["generation_batch_size"],
        "grad_accumulation_steps": derived["gradient_accumulation_steps"],
        "load_s": load_s, "train_wall_s": train_wall, "mem_alloc_before_train_gb": mem_before_gb,
        "peak_alloc_overall_gb": torch.cuda.max_memory_allocated() / 1e9,
        "trainable_params": n_train, "total_params": n_all, "rubric": RUBRIC_VERSION,
        "versions": utils.versions(), "counts": {k: len(v) for k, v in utils.BENCH.items()},
        "steps": utils.assemble_steps(), "trl_logs": utils.BENCH["trl_logs"],
        "raw_load_lora_s": utils.BENCH["sync_s"], "raw_add_adapter_s": utils.BENCH["lora_add_s"],
    }
    with open(os.path.join(run_dir, "bench_steps.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    utils.log.info(f"wrote {os.path.join(run_dir, 'bench_steps.json')}")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    derived = utils.resolve(args)
    adapter_dir = os.path.join(run_dir, "final_model")
    if args.dry_run:
        print(json.dumps({"args": vars(args), "derived": derived, "adapter_dir": adapter_dir}, indent=1))
        return
    os.makedirs(run_dir, exist_ok=True)
    utils.setup_logging()

    if args.verify_only:
        prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
        print(f"[verify] proof written to {utils.verify_adapter(args.model, adapter_dir, prompts)}", flush=True)
        return

    train(args, run_dir, derived)
    if args.verify:
        utils.run_verify(args, __file__)
    utils.log.info("DONE")


if __name__ == "__main__":
    main()
