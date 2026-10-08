import os
import time

import utils


def config_kwargs(args, run_dir: str, flavor: str) -> dict:
    return {
        "flavor": flavor, "hf_dir": os.path.abspath(args.model), "num_training_steps": args.steps,
        "prompts_per_step": args.prompt_batch_size, "group_size": args.group_size,
        "max_response_tokens": args.max_tokens, "max_prompt_tokens": args.max_prompt_length,
        "num_generators": args.num_generators, "generator_tp": args.generator_tp,
        "trainer_dp_shard": args.trainer_dp_shard, "trainer_tp": args.trainer_tp,
        "lr": args.learning_rate, "save_final": not args.no_save, "compile_enable": not args.no_compile,
        "data": os.path.abspath(args.data), "temperature": args.temperature, "seed": args.seed,
        "dump_folder": run_dir,
    }


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    titan_repo = utils.titan_repo_path(args)
    gpus = utils.gpu_list(args)
    flavor = args.flavor or utils.qwen3_flavor(utils.load_model_config(args.model))
    cfg = config_kwargs(args, run_dir, flavor)
    cmd = utils.titan_command(args, run_dir)
    env = utils.subprocess_env(args, run_dir, gpus, cfg)
    if args.config != "rl_telecom":
        print(f"[grpo] preset {args.config}: the shape/model arguments are ignored, only --data, --output-dir "
              f"and --gpus apply", flush=True)

    if args.dry_run:
        utils.print_dry_run(cmd, env, titan_repo, args.gpu_env_var)
        return

    if not args.export_only:
        os.makedirs(run_dir, exist_ok=True)
        print(f"[data] {utils.count_rows(cfg['data'], titan_repo)} rows in {cfg['data']}", flush=True)
        utils.record_command(cmd, run_dir, args.gpu_env_var, gpus, env["TITAN_GRPO_CONFIG"])
        print("[grpo]", utils.write_versions(os.path.join(run_dir, "versions.txt"), titan_repo), flush=True)
        t0 = time.time()
        rc = utils.run_titan(cmd, env, titan_repo, run_dir)
        print(f"[grpo] torchtitan exited rc={rc} after {time.time() - t0:.0f}s", flush=True)
        utils.write_metrics(run_dir, args.warmup_steps, args.steps)
        if rc != 0:
            raise SystemExit(rc)

    if args.no_save and not args.export_only:
        return
    ckpt = utils.latest_hf_checkpoint(run_dir)
    copied = utils.finalize_hf_checkpoint(ckpt, os.path.abspath(args.model))
    print(f"[export] HF checkpoint {ckpt} (copied from base: {copied or 'nothing'})", flush=True)
    if args.verify:
        prompts = args.verify_prompt or ["Which 3GPP release introduced 5G NR Phase 1?"]
        rc = utils.run_verify(args, gpus[0], os.path.abspath(args.model), ckpt, prompts)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
