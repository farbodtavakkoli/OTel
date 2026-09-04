"""LLM reference generation / prompted classification (Transformers + PyTorch) — see readme_inference/transformers/llm.md."""
import argparse
import json, os, time

import torch
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Classify the sentiment as exactly one word, positive or negative: 'This product broke after two days.'",
    "List three open-source inference engines that run on AMD ROCm. Answer as a comma-separated list.",
]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="LLM reference generation / prompted classification")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model id or path")
    parser.add_argument("--prompts_file", type=str, default=None, help="JSON/JSONL file of prompt strings; omit for the built-in set")
    parser.add_argument("--system", type=str, default=None, help="Optional system message prepended to every prompt")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float32", "float16"],
                        help="Model dtype; 'auto' honours the checkpoint's own dtype")
    parser.add_argument("--device_map", type=str, default="cuda:0",
                        help="'cuda:N' pins to one GPU; 'auto' or 'balanced' shards across all visible GPUs")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Tokens to generate per prompt")
    parser.add_argument("--do_sample", action="store_true", help="Sample instead of greedy decoding (greedy is reproducible)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (only with --do_sample)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Nucleus sampling p (only with --do_sample)")
    parser.add_argument("--attn_impl", type=str, default=None, help="Attention implementation; default sdpa on ROCm, eager elsewhere")
    parser.add_argument("--dequantize_fp8", action="store_true",
                        help="Dequantize an FP8 checkpoint to bf16 at load instead of using FP8 kernels")
    parser.add_argument("--no_deepgemm", action="store_true", default=True,
                        help="Force the Triton FP8 kernel; DeepGEMM is CUDA-only and its probe crashes on ROCm (default on)")
    parser.add_argument("--allow_deepgemm", dest="no_deepgemm", action="store_false", help="Re-enable the DeepGEMM FP8 path (NVIDIA only)")
    parser.add_argument("--fix_fp8_gate_proj", action="store_true", default=True,
                        help="Drop 'mlp.gate' skip-entries that wrongly prefix-match 'mlp.gate_proj' and break FP8 (default on)")
    parser.add_argument("--no_fix_fp8_gate_proj", dest="fix_fp8_gate_proj", action="store_false",
                        help="Leave modules_to_not_convert untouched (reproduces the broken-output bug)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Allow checkpoint-provided modeling code (default on)")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed for reproducible outputs")
    parser.add_argument("--output", type=str, default=None, help="Path for the reference-output JSON artifact")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional HF_HOME model-cache override")
    return parser.parse_args()


def read_prompts(path: str, fallback: list[str]) -> list[str]:
    """Read a JSON list or a JSONL/plain-text file of prompts; fall back to the built-in set."""
    if path is None:
        return list(fallback)
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l)["prompt"] if l.lstrip().startswith("{") else l for l in raw.splitlines() if l.strip()]


def disable_deepgemm():
    """Mark DeepGEMM unavailable so FP8 falls back to the Triton kernel.

    On ROCm the DeepGEMM probe reads libcudart.so and raises OSError, which the caller's
    `except ImportError` does not catch; gfx950 also reports capability 9.x so the
    Hopper gate never trips. Pre-marking it makes the probe raise ImportError instead.
    """
    try:
        import transformers.integrations.finegrained_fp8 as fp8
        fp8._deepgemm_available = False
        return True
    except Exception:
        return False


def load_tokenizer(model_id: str, trust_remote_code: bool):
    """Load a tokenizer, going through AutoProcessor for multimodal checkpoints."""
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code).tokenizer


def fix_fp8_skip_list(cfg):
    """Drop 'mlp.gate' / 'mlp.shared_expert_gate' entries from the FP8 skip list.

    Transformers prefix-matches those names against 'mlp.gate_proj', so gate_proj is left
    as an unscaled bf16 Linear while its weight_scale_inv is discarded as UNEXPECTED. The
    scales are ~1e-4, so the SwiGLU gate comes out ~4 orders of magnitude wrong and the
    model emits gibberish. This dense checkpoint has no real MoE router, so the entries
    match nothing legitimate and are safe to remove.
    """
    quant = dict(cfg.quantization_config)
    skip = quant.get("modules_to_not_convert") or []
    keep = [m for m in skip if not (m.endswith(".mlp.gate") or m.endswith(".mlp.shared_expert_gate"))]
    dropped = len(skip) - len(keep)
    if dropped:
        quant["modules_to_not_convert"] = keep
        cfg.quantization_config = quant
    return cfg, dropped


def load_model(args):
    """Load the model, resolving the architecture class and any FP8 quantization handling."""
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = (getattr(cfg, "architectures", None) or ["AutoModelForCausalLM"])[0]
    quant = getattr(cfg, "quantization_config", None)
    quant_method = quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)

    if quant_method == "fp8" and args.fix_fp8_gate_proj and not args.dequantize_fp8:
        cfg, dropped = fix_fp8_skip_list(cfg)
        if dropped:
            print(f"fp8 fix: dropped {dropped} 'mlp.gate*' entries from modules_to_not_convert")

    kwargs = {
        "dtype": args.dtype if args.dtype == "auto" else getattr(torch, args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_impl or ("sdpa" if torch.version.hip else "eager"),
    }
    if quant_method == "fp8" and args.dequantize_fp8:
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        kwargs["dtype"] = torch.bfloat16

    import transformers
    cls = getattr(transformers, arch, None) or AutoModelForCausalLM
    model = cls.from_pretrained(args.model, config=cfg, **kwargs).eval()
    return model, arch, quant_method, kwargs["attn_implementation"]


def build_inputs(tokenizer, prompt: str, system: str | None):
    """Apply the chat template to one prompt, falling back to the raw string."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def vram_gib() -> list[float]:
    """Per-visible-GPU allocated VRAM in GiB, as torch sees it."""
    return [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)

    if args.no_deepgemm:
        disable_deepgemm()

    prompts = read_prompts(args.prompts_file, DEFAULT_PROMPTS)
    tokenizer = load_tokenizer(args.model, args.trust_remote_code)

    t0 = time.time()
    model, arch, quant_method, attn = load_model(args)
    load_s = time.time() - t0

    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    devices = sorted({str(p.device) for p in model.parameters()})
    print(f"model={args.model} arch={arch} quant={quant_method} attn={attn} device_map={args.device_map}")
    print(f"load: {load_s:.1f}s | param dtypes={param_dtypes} | devices={devices}")
    print(f"VRAM(GiB)={vram_gib()}")

    results = []
    for prompt in prompts:
        text = build_inputs(tokenizer, prompt, args.system)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        gen = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample}
        if args.do_sample:
            gen.update(temperature=args.temperature, top_p=args.top_p)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        dt = time.time() - t0
        new = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new, skip_special_tokens=True)
        results.append({"prompt": prompt, "completion": completion,
                        "n_prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "n_new_tokens": int(len(new)), "seconds": round(dt, 2),
                        "tokens_per_second": round(len(new) / dt, 2)})
        print(f"\n[{dt:.1f}s, {len(new) / dt:.2f} tok/s] PROMPT: {prompt}")
        print(f"OUTPUT: {completion!r}")

    print(f"\npeak VRAM(GiB)={[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]}")

    artifact = {
        "model": args.model, "architecture": arch, "quant_method": quant_method,
        "dequantize_fp8": args.dequantize_fp8, "attn_implementation": attn,
        "device_map": args.device_map, "param_devices": devices, "param_dtypes": param_dtypes,
        "do_sample": args.do_sample, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_properties(0).gcnArchName,
        "load_seconds": round(load_s, 2), "vram_gib": vram_gib(),
        "results": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        print(f"\nwrote reference artifact: {args.output}")


if __name__ == "__main__":
    main()
