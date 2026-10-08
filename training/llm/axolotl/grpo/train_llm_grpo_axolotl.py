import json
import os
import signal
import sys

import yaml

import utils


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _set_dotted(cfg: dict, key: str, value):
    parts = key.split(".")
    d = cfg
    for k in parts[:-1]:
        d = d.setdefault(k, {})
    d[parts[-1]] = value


def build_config(args) -> dict:
    with open(args.config) as f:
        cfg = _expand(yaml.safe_load(f))
    if args.model:
        cfg["base_model"] = os.path.abspath(args.model)
    if args.data:
        cfg["datasets"][0]["path"] = os.path.abspath(args.data)
    if args.output_dir:
        cfg["output_dir"] = os.path.abspath(args.output_dir)
    cfg["dataset_prepared_path"] = os.path.join(cfg["output_dir"], "prepared")
    trl = cfg.setdefault("trl", {})
    if args.group_size:
        trl["num_generations"] = args.group_size
    if args.max_tokens:
        trl["max_completion_length"] = args.max_tokens
    if args.steps:
        cfg["max_steps"] = args.steps
    if args.prompt_batch_size:
        per_step, mbs = args.prompt_batch_size * int(trl["num_generations"]), int(cfg["micro_batch_size"])
        if per_step % mbs:
            raise SystemExit(f"prompts x group size ({per_step}) must be a multiple of micro_batch_size ({mbs})")
        cfg["gradient_accumulation_steps"] = per_step // mbs
    for kv in args.set:
        k, _, v = kv.partition("=")
        _set_dotted(cfg, k, yaml.safe_load(v))
    for label, v in (("base_model", cfg["base_model"]), ("datasets[0].path", cfg["datasets"][0]["path"]),
                     ("output_dir", cfg["output_dir"])):
        if "$" in str(v):
            raise SystemExit(f"{label} = {v!r}: unexpanded variable (set it, or pass --model/--data/--output-dir)")
    return cfg


def main():
    args = utils.parse_args()
    cfg = build_config(args)
    out = cfg["output_dir"]
    cfg_path = os.path.join(out, "axolotl_config.yaml")
    launcher = utils.Launcher(args, cfg, cfg_path)

    if args.dry_run:
        print(json.dumps({"config": args.config, "effective": {
            "base_model": cfg["base_model"], "data": cfg["datasets"][0]["path"], "output_dir": out,
            "max_steps": cfg.get("max_steps"), "num_generations": cfg["trl"].get("num_generations"),
            "max_completion_length": cfg["trl"].get("max_completion_length"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"), "micro_batch_size": cfg.get("micro_batch_size"),
            "warmup_steps": launcher.warmup, "label": launcher.label, "sync_dir": launcher.sync_dir},
            "server": " ".join(launcher.server_cmd()), "train": " ".join(launcher.train_cmd()),
            "verify": " ".join(launcher.verify_cmd()) if launcher.docker else "(in-process: utils.verify_adapter)"}, indent=1))
        return

    os.makedirs(out, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    if not args.export_only:
        try:
            launcher.start_server()
            rc = launcher.train()
        finally:
            launcher.stop_server()
        if rc != 0:
            raise SystemExit(f"axolotl train failed with exit code {rc}")

    if args.no_export:
        return
    info = utils.adapter_files(out)
    print(f"[export] adapter in {out}: {info['tensors']} tensors, {info['modules']} LoRA modules, r={info['r']}", flush=True)
    if not args.verify:
        return
    if launcher.docker:
        rc = launcher.run_logged(launcher.verify_cmd(), "verify.log")
        if rc != 0:
            raise SystemExit(f"verify container failed with exit code {rc}")
    else:
        prompts, meta = utils.verify_prompts(args, cfg)
        proof = utils.verify_adapter(cfg["base_model"], out, prompts, meta=meta)
        print(f"[verify] proof written to {proof}", flush=True)


if __name__ == "__main__":
    main()
