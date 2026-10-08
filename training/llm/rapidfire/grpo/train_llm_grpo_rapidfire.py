import json
import os
import sys
import time

import utils

# RapidFire's own ports (its default MLflow port is 8852); off the defaults so two stacks can share a node.
RF_PORTS = {"RF_API_PORT": "8861", "RF_FRONTEND_PORT": "8863", "RF_RAY_PORT": "8865"}


def grpo_config_kwargs(args):
    grad_accum = (args.prompt_batch_size * args.group_size) // args.per_device_batch
    return dict(
        output_dir=os.path.join(os.path.abspath(args.output_dir), "trainer_output"),
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
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


def train(args):
    run_dir = os.path.abspath(args.output_dir)
    adapter_dir = os.path.join(run_dir, "adapter")
    prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
    # RapidFire reads these at import time; the spawned worker inherits the whole environment.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RF_EXPERIMENT_PATH", os.path.join(run_dir, "rf_experiments"))
    os.environ.setdefault("RF_LOG_PATH", os.path.join(run_dir, "rf_logs"))
    os.environ["RF_MLFLOW_PORT"] = str(args.mlflow_port)
    for k, v in RF_PORTS.items():
        os.environ.setdefault(k, v)
    os.environ["RF_BENCH_OUT"] = run_dir
    os.environ["RF_BENCH_MAX_COMPLETION"] = str(args.max_tokens)
    os.environ["RF_BENCH_DUMP_COMPLETIONS"] = str(args.dump_completions)

    import torch
    import utils
    import judge_reward

    if args.verify_only:
        proof = utils.verify_adapter(args.model, adapter_dir, prompts)
        print(f"[verify] proof written to {proof['proof_file']}", flush=True)
        return

    n_dev = torch.cuda.device_count()
    assert n_dev == 1, f"RapidFire spawns one worker per visible GPU; expected exactly one, got {n_dev}"
    n_completions = args.prompt_batch_size * args.group_size
    grad_accum = n_completions // args.per_device_batch
    assert grad_accum * args.per_device_batch == n_completions, "prompt-batch-size * group-size must be a multiple of per-device-batch"
    assert 0 <= args.warmup_steps < args.steps, "need warmup-steps < steps"
    for name in ("steps.jsonl", "events.jsonl", "judge_calls.jsonl", "completions_sample.jsonl", "worker.log"):
        open(os.path.join(run_dir, name), "w").close()

    eos_ids = utils.eos_id_set(args.model)
    os.environ["RF_BENCH_EOS_IDS"] = ",".join(str(i) for i in sorted(eos_ids))
    cfg = utils.load_model_config(args.model)
    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    ds = utils.load_grpo_dataset(args.data)
    print(f"[grpo] dataset rows={len(ds)} columns={ds.column_names} eos_ids={sorted(eos_ids)} lora_regex={lora_regex}", flush=True)
    utils.event("controller_start", argv=sys.argv,
                env={k: v for k, v in os.environ.items() if k.startswith(("RF_", "ROCR", "HIP_", "CUDA_"))})

    from rapidfireai import Experiment
    from rapidfireai.automl import List, RFGridSearch, RFGRPOConfig, RFLoraConfig, RFModelConfig

    peft_cfg = RFLoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                            task_type="CAUSAL_LM", target_modules=lora_regex)
    training_args = RFGRPOConfig(**grpo_config_kwargs(args))
    model_cfg = RFModelConfig(
        model_name=args.model, peft_config=peft_cfg, training_args=training_args,
        model_type="image_text_to_text" if utils.is_multimodal(cfg) else "causal_lm",   # informational in the fit path
        model_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},               # RapidFire adds device_map
        tokenizer_kwargs={"use_fast": True},
        reward_funcs=[utils.judge_reward],
    )
    config_group = RFGridSearch(configs=List([model_cfg]), trainer_type="GRPO")
    print(f"[grpo] RFGridSearch of size 1, num_chunks={args.num_chunks}, steps={args.steps}, generation batch={n_completions}", flush=True)

    mlflow = utils.start_mlflow(args.mlflow_port, run_dir)
    run_fit_wall = None
    try:
        t_exp = time.perf_counter()
        experiment = Experiment(experiment_name=args.experiment_name, mode="fit")
        utils.event("experiment_created", experiment_id=experiment.experiment_id,
                    experiment_path=str(experiment.experiment_path), init_s=time.perf_counter() - t_exp)
        try:
            t0 = time.perf_counter()
            utils.event("run_fit_start")
            experiment.run_fit(config_group, utils.create_model, ds, None, num_chunks=args.num_chunks, seed=args.seed)
            run_fit_wall = time.perf_counter() - t0
            utils.event("run_fit_end", run_fit_s=run_fit_wall)
            print(f"[grpo] run_fit wall={run_fit_wall:.1f}s", flush=True)
            try:
                experiment.get_results().to_csv(os.path.join(run_dir, "rf_get_results.csv"))
            except Exception as e:  # noqa: BLE001
                print(f"[grpo] get_results failed: {e}", flush=True)
        finally:
            t1 = time.perf_counter()
            experiment.end()
            utils.event("experiment_end", end_s=time.perf_counter() - t1)
    finally:
        utils.stop_mlflow(mlflow, run_dir)

    adapter_src = utils.collect_adapter(experiment, adapter_dir)
    config = {
        "model": args.model, "dataset": args.data, "output_dir": run_dir, "label": args.label,
        "num_generations": args.group_size, "prompts_per_step": args.prompt_batch_size,
        "per_device_batch": args.per_device_batch, "max_completion_length": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "beta": args.beta, "learning_rate": args.learning_rate,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps - args.warmup_steps, "seed": args.seed,
        "num_chunks": args.num_chunks, "experiment_name": args.experiment_name,
        "gradient_checkpointing": not args.no_gradient_checkpointing, "vllm_mode": "server",
        "vllm_server_host": args.server_host, "vllm_server_port": args.server_port, "server_mode": args.server_mode,
        "vllm_server_args": utils.server_args_record(args), "gradient_accumulation_steps": grad_accum,
        "generation_batch_size": n_completions, "lora_target_modules": lora_regex, "eos_ids": sorted(eos_ids),
        "rf_grpo_config_user_params": training_args._user_params, "rf_lora_config_user_params": peft_cfg._user_params,
        "trl_defaults_used": {
            "loss_type": training_args.loss_type, "epsilon": training_args.epsilon,
            "vllm_importance_sampling_correction": training_args.vllm_importance_sampling_correction,
            "vllm_importance_sampling_cap": training_args.vllm_importance_sampling_cap,
            "scale_rewards": training_args.scale_rewards, "mask_truncated_completions": training_args.mask_truncated_completions,
            "num_iterations": training_args.num_iterations, "optim": training_args.optim},
    }
    summary_path = os.path.join(run_dir, "summary.json")
    summary = utils.write_grpo_summary(summary_path, run_dir, config, args.warmup_steps, run_fit_wall, adapter_dir,
                                       adapter_src, judge_reward.RUBRIC_VERSION)
    print("[grpo] DONE", json.dumps({k: summary[k] for k in ("steps_total", "steps_measured", "peak_mem_gb",
                                                            "reward_first", "reward_last", "rf_overhead")}, default=str), flush=True)
    # Runs only after the worker has exited and released the GPU.
    summary["adapter_proof"] = utils.verify_adapter(args.model, adapter_dir, prompts if args.verify else [])
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    if args.verify:
        print(f"[verify] proof written to {summary['adapter_proof']['proof_file']}", flush=True)

def main():
    args = utils.parse_args()
    if args.server_mode == "docker":
        utils.run_docker(args, grpo_config_kwargs)
    else:
        utils.run_local(args, train, grpo_config_kwargs)


if __name__ == "__main__":
    main()
