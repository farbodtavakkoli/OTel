import hashlib
import json
import math
import os
import random
import shlex
import statistics
import sys
import time

import utils

TIMED = ("rollout_s", "judge_s", "train_step_s", "sync_s", "cycle_s", "tokens_per_s")


def base_train_args(args, lora_regex: str) -> dict:
    common = {
        "gpus": args.trainer_gpus, "nodes": 1, "max_token_block_size": args.block_size,
        "steps_per_checkpoint": 100000, "adapter_type": "lora",
        "lora_config": {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": 0.0,
                        "target_modules": lora_regex, "use_rslora": False},
        "seed": args.seed, "optimizer_type": "adamw", "gradient_checkpointing": True,
        "distribution_strategy": "pytorch_fsdp", "attn_implementation": "sdpa",
        "freeze_layer_keywords": utils.FROZEN_TOWERS,
        "cuda_memory_log_interval": 1, "timeout": 3600,
        "upload_to_hf": False, "hf_repo_id": "", "hf_upload_token": "", "custom_data_path": None,
    }
    if args.objective == "grpo":
        return dict(common, training_mode="grpo", grpo={"clip_eps": args.clip_eps, "kl_beta": 0.0},
                    max_steps=args.ppo_epochs, learning_rate=args.learning_rate, batch_size=args.micro_batch,
                    init_optimizer_from_adapter=not args.no_carry_optimizer)
    return dict(common, training_mode="language_model", max_steps=args.rft_max_steps,
                learning_rate=args.rft_learning_rate, batch_size=args.rft_batch_size,
                gradient_accumulation_steps=args.rft_grad_accum)


class PersistentTrainer:
    def __init__(self, args, train_args_base):
        self.args, self.train_args_base = args, train_args_base
        self.tag = f"{os.path.basename(os.path.abspath(args.output_dir))}-{time.time():.0f}"
        self.session, self.seq, self.starts = None, 0, 0

    def start(self, init_adapter_from=None):
        self.starts += 1
        self.session = utils.start_persistent_trainer(self.args.api_url, self.train_args_base,
                                                      f"{self.tag}-{self.starts}", init_adapter_from)
        self.seq = 0
        print(f"[persistent] trainer session {self.session[:12]} submitted "
              f"(init_adapter_from={str(init_adapter_from)[:12]})", flush=True)

    def train(self, dataset, train_args, previous_job):
        for attempt in (1, 2):
            if self.session is None:
                self.start(previous_job)
            self.seq += 1
            request = {"model_name": hashlib.sha256(f"{self.session}:{self.seq}".encode()).hexdigest(),
                       "steps": train_args["max_steps"],
                       "gradient_accumulation_steps": train_args["gradient_accumulation_steps"],
                       "sampling_seed": train_args["sampling_seed"]}
            try:
                result, t_done = utils.persistent_train_step(self.args.api_url, self.args.jobs_dir, self.session,
                                                             self.seq, dataset, request)
                return result, t_done, self.session, self.seq
            except utils.PersistentSessionEnded as e:
                print(f"  {e}; starting a new trainer session", flush=True)
                self.session = None
                if attempt == 2:
                    raise

    def stop(self):
        if self.session is None:
            return
        state = utils.stop_persistent_trainer(self.args.api_url, self.args.jobs_dir, self.session, self.seq + 1)
        print(f"[persistent] trainer session {self.session[:12]} stopped: {state}", flush=True)
        self.session = None


def persistent_cycle(persistent, cyc, dataset, train_args, previous_job, t_submit):
    result, t_done, session, seq = persistent.train(dataset, train_args, previous_job)
    train_step_s = t_done - t_submit
    gs, losses = result["grpo_steps"], result["losses"]
    cyc.update(train_step_s=train_step_s, model_name=result["model_name"],
               train_args={k: v for k, v in train_args.items() if k in ("gradient_accumulation_steps", "sampling_seed")},
               train_loss_first=losses[0] if losses else None, train_loss_last=losses[-1] if losses else None,
               grpo_mean_ratio=gs[0].get("mean_ratio") if gs else None, grpo_clip_frac=gs[0].get("clip_frac") if gs else None,
               grpo_tokens=gs[0].get("tokens") if gs else None, grpo_steps=gs,
               trainer_step_times_s=result["step_times_s"], trainer_microbatch_times_s=result["microbatch_times_s"],
               trainer_from_pretrained_s=None, init_adapter={"persistent_session": session, "request": seq},
               init_optimizer=None, trainer_on_policy_warning=None, peak_mem_gib=result["peak_mem_gib"],
               trainer_gpu_env=None, trainer_device=None, trainer_log=None, trainer_wait_ready_s=result["wait_ready_s"],
               trainer_train_s=result["train_s"], trainer_save_s=result["save_s"])
    print(f"  train {train_step_s:.1f}s (persistent session {session[:12]} request {seq}; waited "
          f"{result['wait_ready_s']:.1f}s for the trainer, train {result['train_s']:.1f}s, step times "
          f"{[round(s, 2) for s in result['step_times_s']]}, save {result['save_s']:.1f}s, peak {result['peak_mem_gib']} GiB, "
          f"loss {cyc['train_loss_first']} -> {cyc['train_loss_last']}, mean_ratio {cyc['grpo_mean_ratio']} "
          f"clip_frac {cyc['grpo_clip_frac']})", flush=True)
    return result["model_name"], train_step_s


def run_loop(args, rows: list, turn_end: str, train_args_base: dict, want_lp: bool, run_dir: str,
             persistent=None) -> dict:
    from judge_reward import RUBRIC_VERSION, score_batch

    G, n_cycles = args.group_size, args.warmup_steps + args.steps
    cycles_path = os.path.join(run_dir, "cycles.jsonl")
    current_model, previous_job, winners_all, cycles = args.model_name, None, [], []
    t_run0 = time.time()

    for c in range(n_cycles):
        measured = c >= args.warmup_steps
        batch = rows[c * args.prompt_batch_size:(c + 1) * args.prompt_batch_size]
        print(f"\n=== cycle {c + 1}/{n_cycles} ({'measured' if measured else 'warm-up'}) policy={current_model[:16]} ===", flush=True)
        cyc = {"cycle": c + 1, "measured": measured, "policy": current_model}

        gens, rollout_s = utils.rollout(args.api_url, current_model, [r["rendered"] for r in batch], G, args.max_tokens,
                                        args.seed * 100000 + c * 10000, args.concurrency, want_lp, args.temperature)
        gen_tokens = sum(g["completion_tokens"] for g in gens)
        cyc.update(rollout_s=rollout_s, gen_tokens=gen_tokens, tokens_per_s=gen_tokens / rollout_s,
                   rollout_errors=sum(1 for g in gens if g.get("error")),
                   completions_with_old_logprobs=sum(1 for g in gens if g["old_logprobs"] is not None),
                   mean_completion_tokens=gen_tokens / len(gens),
                   finish_length_frac=sum(g["finish_reason"] == "length" for g in gens) / len(gens))
        print(f"  rollout {rollout_s:.1f}s  {gen_tokens} tokens  {gen_tokens / rollout_s:.0f} tok/s  "
              f"len-truncated {cyc['finish_length_frac']:.2f}  errors {cyc['rollout_errors']}  "
              f"with old_logprobs {cyc['completions_with_old_logprobs']}/{len(gens)}", flush=True)

        questions = [r["question"] or r["messages"] for r in batch for _ in range(G)]
        refs = [r["reference"] for r in batch for _ in range(G)]
        srcs = [r["source"] for r in batch for _ in range(G)]
        scored = score_batch(questions, [g["text"] for g in gens], refs, srcs, [g["finish_reason"] for g in gens])
        rewards = scored["reward"]
        cyc.update(judge_s=scored["timing_s"], reward_mean=statistics.fmean(rewards), reward_std=statistics.pstdev(rewards),
                   judge_failed=sum(1 for j, s in zip(scored["judge"], srcs) if j is None and s != "3gpp_tsg"))
        print(f"  judge {scored['timing_s']:.1f}s  reward mean {cyc['reward_mean']:.3f}  judge failures {cyc['judge_failed']}", flush=True)

        advs, zero_spread = utils.group_advantages(rewards, G, keep_zero_spread=args.keep_zero_spread)
        if args.objective == "grpo":
            dataset = []
            for i, (g, a) in enumerate(zip(gens, advs)):
                if a is None or g.get("error") or g["completion_tokens"] == 0 or (want_lp and not g["completion_token_ids"]):
                    continue
                dataset.append({"prompt": batch[i // G]["rendered"], "completion": g["text"], "advantage": a,
                                "old_logprobs": g["old_logprobs"] if want_lp else None,
                                "prompt_token_ids": g["prompt_token_ids"] if want_lp else None,
                                "completion_token_ids": g["completion_token_ids"] if want_lp else None,
                                "id": batch[i // G]["id"], "source": srcs[i], "reward": rewards[i],
                                "finish_reason": g["finish_reason"], "cycle": c + 1})
            cyc.update(zero_spread_groups=zero_spread, train_rows=len(dataset),
                       adv_abs_mean=statistics.fmean(abs(r["advantage"]) for r in dataset) if dataset else None)
            print(f"  train rows {len(dataset)}/{len(gens)} (zero-spread groups "
                  f"{'kept with advantage 0' if args.keep_zero_spread else 'dropped'}: {zero_spread}/{len(batch)})", flush=True)
        else:
            winners = []
            for pi, r in enumerate(batch):
                grp = rewards[pi * G:(pi + 1) * G]
                if max(grp) - min(grp) <= 1e-9:
                    continue
                best = max(range(G), key=lambda i: grp[i])
                winners.append({"input": r["rendered"], "output": gens[pi * G + best]["text"] + turn_end,
                                "id": r["id"], "source": r["source"], "reward": grp[best], "cycle": c + 1})
            winners_all = winners if args.rft_no_cumulative else winners_all + winners
            dataset = [{"input": w["input"], "output": w["output"]} for w in winners_all]
            cyc.update(zero_spread_groups=zero_spread, winners=len(winners), sft_rows=len(dataset))
            print(f"  winners {len(winners)}/{len(batch)} (zero-spread groups dropped: {zero_spread}); SFT rows {len(dataset)}", flush=True)

        with open(os.path.join(run_dir, "rollouts", f"cycle_{c + 1:02d}.jsonl"), "w") as f:
            for i, g in enumerate(gens):
                f.write(json.dumps({"id": batch[i // G]["id"], "source": srcs[i], "g": i % G, "reward": rewards[i],
                                    "judge": scored["judge"][i], "advantage": advs[i], "finish_reason": g["finish_reason"],
                                    "completion_tokens": g["completion_tokens"],
                                    "old_logprob_sum": sum(g["old_logprobs"]) if g["old_logprobs"] else None,
                                    "text": g["text"]}) + "\n")

        if not dataset:
            print("  no rows with signal -> no training job this cycle", flush=True)
            cyc.update(train_step_s=None, sync_s=None, cycle_s=None, model_name=None, note="no rows with signal; no training job")
            cycles.append(cyc)
            utils.append_jsonl(cycles_path, cyc)
            continue

        train_args = dict(train_args_base, sampling_seed=args.seed + c)
        if args.objective == "grpo":
            rows_per_rank = math.ceil(len(dataset) / args.trainer_gpus)
            train_args.update(gradient_accumulation_steps=max(1, math.ceil(rows_per_rank / args.micro_batch)),
                              init_adapter_from=previous_job)
        t_submit = time.time()
        if persistent is not None:
            model_name, train_step_s = persistent_cycle(persistent, cyc, dataset, train_args, previous_job, t_submit)
        else:
            status = utils.submit_training_job(args.api_url, dataset, train_args)["job_status"]
            model_name = status["model_name"]
            print(f"  submitted job {model_name} ({len(dataset)} rows, "
                  f"init_adapter_from={str(train_args.get('init_adapter_from'))[:12]}) status={status.get('status')}", flush=True)
            final_status, t_completed = utils.wait_for_job(args.api_url, model_name)
            train_step_s = t_completed - t_submit
            log = utils.parse_job_log(os.path.join(args.jobs_dir, model_name))
            hist = final_status.get("history", [])
            gs = log["grpo_steps"]
            cyc.update(train_step_s=train_step_s, model_name=model_name,
                       train_args={k: v for k, v in train_args.items() if k in ("gradient_accumulation_steps", "sampling_seed", "init_adapter_from")},
                       train_loss_first=hist[0]["loss"] if hist else None, train_loss_last=hist[-1]["loss"] if hist else None,
                       grpo_mean_ratio=gs[0]["mean_ratio"] if gs else None, grpo_clip_frac=gs[0]["clip_frac"] if gs else None,
                       grpo_tokens=gs[0]["tokens"] if gs else None, grpo_steps=gs,
                       trainer_step_times_s=log["step_times_s"], trainer_microbatch_times_s=log["microbatch_times_s"],
                       trainer_from_pretrained_s=log["from_pretrained_s"], init_adapter=log["init_adapter"],
                       init_optimizer=log["init_optimizer"], trainer_on_policy_warning=log["on_policy_warning"],
                       peak_mem_gib=log["peak_mem_gib"], trainer_gpu_env=log["gpu_env"], trainer_device=log["trainer_device"],
                       trainer_log=log["log"])
            print(f"  train {train_step_s:.1f}s (COMPLETED; from_pretrained {log['from_pretrained_s']}s, step times "
                  f"{[round(s, 2) for s in log['step_times_s']]}, peak {log['peak_mem_gib']} GiB, loss {cyc['train_loss_first']} -> "
                  f"{cyc['train_loss_last']}, mean_ratio {cyc['grpo_mean_ratio']} clip_frac {cyc['grpo_clip_frac']}, "
                  f"init_adapter {log['init_adapter']})", flush=True)

        sync_s, probe = utils.sync_adapter(args.api_url, model_name, batch[0]["rendered"])
        cyc.update(sync_s=sync_s, sync_probe=probe[:80], cycle_s=rollout_s + scored["timing_s"] + train_step_s + sync_s,
                   cycle_wall_s=time.time() - t_submit + rollout_s + scored["timing_s"])
        print(f"  sync {sync_s:.1f}s  cycle {cyc['cycle_s']:.1f}s", flush=True)
        current_model = previous_job = model_name
        cycles.append(cyc)
        utils.append_jsonl(cycles_path, cyc)

    if args.objective == "rft":
        with open(os.path.join(run_dir, "winners.jsonl"), "w") as f:
            for w in winners_all:
                f.write(json.dumps(w) + "\n")

    meas = [x for x in cycles if x["measured"] and x.get("cycle_s") is not None]
    summary = {"label": args.label, "objective": args.objective, "rubric": RUBRIC_VERSION, "args": vars(args),
               "train_args": train_args_base, "n_cycles": n_cycles, "n_measured": len(meas),
               "old_logprobs_available": want_lp, "total_wall_s": time.time() - t_run0,
               "models": [x.get("model_name") for x in cycles],
               "final_adapter": current_model if current_model != args.model_name else None}
    for k in TIMED:
        vals = [x[k] for x in meas if x.get(k) is not None]
        summary[k] = vals
        summary[f"{k}_median"] = statistics.median(vals) if vals else None
    summary["peak_mem_gib_max"] = max((x["peak_mem_gib"] for x in meas if x.get("peak_mem_gib")), default=None)
    summary["reward_first"] = meas[0]["reward_mean"] if meas else None
    summary["reward_last"] = meas[-1]["reward_mean"] if meas else None
    summary["reward_per_cycle"] = [x["reward_mean"] for x in cycles]
    summary["grpo_mean_ratio_per_cycle"] = [x.get("grpo_mean_ratio") for x in cycles]
    summary["grpo_clip_frac_per_cycle"] = [x.get("grpo_clip_frac") for x in cycles]
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nSUMMARY", json.dumps({k: v for k, v in summary.items() if k.endswith("_median") or k in
                                  ("reward_first", "reward_last", "peak_mem_gib_max", "n_measured", "old_logprobs_available", "final_adapter")},
                                 indent=1), flush=True)
    return summary


def main():
    args = utils.parse_args()
    run_dir = os.path.abspath(args.output_dir)
    cfg = utils.load_model_config(args.model)
    lora_regex = args.lora_regex or utils.default_lora_regex(cfg)
    train_args_base = base_train_args(args, lora_regex)
    need = (args.warmup_steps + args.steps) * args.prompt_batch_size

    if args.dry_run:
        print(json.dumps({"served_model": args.model_name, "objective": args.objective, "rows_needed": need,
                          "server_command": shlex.join(utils.server_cmd(args)) if args.start_server else f"existing server at {args.api_url}",
                          "jobs_dir": args.jobs_dir, "train_args": train_args_base, "output_dir": run_dir,
                          "verify": args.verify}, indent=1))
        return

    os.makedirs(os.path.join(run_dir, "rollouts"), exist_ok=True)
    os.makedirs(args.jobs_dir, exist_ok=True)
    rows = utils.load_rows(args.data)
    random.Random(args.seed).shuffle(rows)
    if need > len(rows):
        raise SystemExit(f"need {need} prompts, have {len(rows)}")
    turn_end = utils.render_prompts(rows, args.model)
    with open(os.path.join(run_dir, "command.txt"), "a") as f:
        f.write(f"=== {args.label}\n{shlex.join(sys.argv)}\n")

    started = False
    persistent = None
    try:
        if args.start_server:
            utils.ensure_hf_cache_entry(args.model_name, args.model, utils.mount_path(args), args.hf_cache)
            cmd = utils.server_cmd(args)
            with open(os.path.join(run_dir, "command.txt"), "a") as f:
                f.write(shlex.join(cmd) + "\n")
            utils.start_server(cmd, args.container_name)
            started = True
            t0 = time.time()
            print(f"[server] health: {utils.wait_health(args.api_url, args.server_timeout)} after {time.time() - t0:.0f}s", flush=True)
        h = utils.health(args.api_url)
        print("server health:", h, flush=True)
        if h.get("all") != "up":
            raise SystemExit("ScalarLM server is not fully up")
        if args.persistent_trainer and args.objective == "grpo":
            persistent = PersistentTrainer(args, train_args_base)
            persistent.start()

        want_lp = args.objective == "grpo" and not args.no_old_logprobs
        if want_lp:
            ok, probe = utils.probe_logprobs(args.api_url, args.model_name, rows[0]["rendered"])
            if not ok:
                print("WARNING: /v1/completions does not return logprobs/token ids (pre-v1.7 proxy) -> on-policy ratio==1 path", flush=True)
            want_lp = ok
            print(f"old_logprobs from the server: {want_lp} (probe: {len(probe['completion_token_ids'] or [])} tokens)", flush=True)

        summary = run_loop(args, rows, turn_end, train_args_base, want_lp, run_dir, persistent)
        if args.verify:
            utils.verify(args, summary, run_dir)
    finally:
        if persistent is not None:
            persistent.stop()
        if started and not args.keep_server:
            utils.stop_server(args.container_name, os.path.join(run_dir, "server.log"))


if __name__ == "__main__":
    main()
