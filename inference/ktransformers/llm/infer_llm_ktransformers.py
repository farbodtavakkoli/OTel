"""KTransformers probe, direct-API generation (ROCm/CUDA), and chat client for a sglang-kt server — see README.md."""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import threading
import time

import torch
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the loader needs it to pull gated model repos.
load_dotenv("dev.env")

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_PROMPT = "In two sentences, explain what a mixture-of-experts model is and why it saves compute."


def parse_args():
    """Parse CLI arguments; every tunable of the hybrid MoE run is exposed here."""
    parser = argparse.ArgumentParser(description="kt-kernel CPU-GPU heterogeneous MoE probe / generation client")
    parser.add_argument("--mode", type=str, default="probe", choices=["probe", "kernel", "generate", "chat"],
                        help="probe: environment only; kernel: synthetic MoE kernel check; generate: real model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id or local path of the MoE model")
    parser.add_argument("--kt_method", type=str, default="BF16",
                        help="kt-kernel CPU backend (BF16, FP8, FP8_PERCHANNEL, RAWINT4, LLAMAFILE, MOE_INT8)")
    parser.add_argument("--cpuinfer_threads", type=int, default=128, help="CPU inference threads (physical cores)")
    parser.add_argument("--threadpool_count", type=int, default=2, help="Thread pools (set to NUMA node count)")
    parser.add_argument("--num_gpu_experts", type=int, default=0, help="Experts kept on GPU (0 = all experts on CPU)")
    parser.add_argument("--max_layers", type=int, default=0, help="Offload only the first N MoE layers (0 = all)")
    parser.add_argument("--chunked_prefill_size", type=int, default=512, help="Maximum prefill chunk size")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (generate mode resolves None to 42)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (chat mode)")
    parser.add_argument("--port", type=int, default=38612, help="Server port (chat mode); never the SGLang default")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt (chat mode)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens to generate (chat mode)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p (chat mode)")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds (chat mode)")
    parser.add_argument("--out", default=None, help="Write a JSON artifact of the probe/chat result")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for the non-expert half")
    parser.add_argument("--sample_hw", action="store_true", help="Sample rocm-smi VRAM and CPU load during decode")
    return parser.parse_args()


def report_environment():
    """Print the torch/ROCm/kt-kernel facts that decide whether this stack is even viable."""
    import kt_kernel
    from kt_kernel import kt_kernel_ext

    ext = os.path.realpath(kt_kernel.kt_kernel_ext.__file__)
    needed = subprocess.run(["readelf", "-d", ext], capture_output=True, text=True).stdout
    gpu_runtime = "HIP (libamdhip64)" if "libamdhip64" in needed else "CUDA / none"

    print(f"torch              : {torch.__version__}")
    print(f"torch.version.hip  : {torch.version.hip}")
    print(f"torch.version.cuda : {torch.version.cuda}")
    print(f"visible devices    : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"device 0           : {torch.cuda.get_device_name(0)} / {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"kt_kernel version  : {kt_kernel.__version__}")
    print(f"kt_kernel variant  : {getattr(kt_kernel, '__cpu_variant__', 'source-build (single variant)')}")
    print(f"kt_kernel_ext      : {ext}")
    print(f"GPU runtime linked : {gpu_runtime}")
    print(f"stream interop     : {hasattr(kt_kernel_ext.CPUInfer(1), 'submit_with_cuda_stream')}")
    return gpu_runtime


def cpu_flags():
    """Return the MoE-relevant instruction-set flags this CPU actually has."""
    wanted = ["avx512f", "avx512bw", "avx512vbmi", "avx512_vnni", "avx512_bf16",
              "amx_tile", "amx_int8", "amx_bf16"]
    with open("/proc/cpuinfo") as handle:
        for line in handle:
            if line.startswith("flags"):
                have = set(line.split(":", 1)[1].split())
                return {flag: (flag in have) for flag in wanted}
    return {}


def vram_used_gb(index=0):
    """Return VRAM used on one card, in GB, via rocm-smi."""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(f"card{index},"):
            return int(line.split(",")[2]) / 1e9
    return 0.0


def sample_hardware(stop_event, samples, index=0):
    """Background sampler recording GPU VRAM and system CPU utilisation during decode."""
    prev = None
    while not stop_event.is_set():
        with open("/proc/stat") as handle:
            parts = [int(x) for x in handle.readline().split()[1:]]
        busy, total = sum(parts) - parts[3] - parts[4], sum(parts)
        if prev is not None:
            d_busy, d_total = busy - prev[0], total - prev[1]
            if d_total:
                samples.append((100.0 * d_busy / d_total, vram_used_gb(index)))
        prev = (busy, total)
        time.sleep(0.5)


def build_kt_experts(block, layer_idx, config, args, device):
    """Swap one Qwen3MoeSparseMoeBlock's GPU experts for a kt-kernel CPU-resident expert bank."""
    from kt_kernel import KTMoEWrapper

    mask = None
    if args.num_gpu_experts > 0:
        mask = torch.zeros(config.num_experts, dtype=torch.bool)
        mask[: args.num_gpu_experts] = True

    wrapper = KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        gpu_experts_mask=mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=args.weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        method=args.kt_method,
    )
    wrapper.load_weights(torch.arange(config.num_experts, dtype=torch.int64).contiguous())

    class KTExperts(torch.nn.Module):
        """Drop-in for Qwen3MoeExperts that runs the expert FFNs on CPU via kt-kernel."""

        def forward(self, hidden_states, top_k_index, top_k_weights):
            stream = torch.cuda.current_stream().cuda_stream
            out = wrapper.forward(hidden_states, top_k_index, top_k_weights.to(torch.float32), stream)
            return out.to(hidden_states.dtype).view_as(hidden_states)

    block.experts = KTExperts().to(device)
    return wrapper


def load_hybrid_model(args, device):
    """Load the MoE model with attention/router/embeddings on GPU and every expert on CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")
    print(f"host load          : {time.time() - started:.1f}s")

    wrappers = []
    for layer_idx, layer in enumerate(model.model.layers):
        block = getattr(layer, "mlp", None)
        if not hasattr(block, "experts") or not hasattr(block.experts, "gate_up_proj"):
            continue
        if args.max_layers and len(wrappers) >= args.max_layers:
            continue
        del block.experts
        wrappers.append(build_kt_experts(block, layer_idx, config, args, device))
    print(f"kt-kernel layers   : {len(wrappers)} MoE layers offloaded to CPU")

    started = time.time()
    model = model.to(device)
    model.eval()
    print(f"gpu upload         : {time.time() - started:.1f}s")
    return model, tokenizer, wrappers


def run_kernel_check(args):
    """Validate the CPU MoE kernel against a PyTorch reference, with no model download."""
    from kt_kernel import kt_kernel_ext

    torch.manual_seed(args.seed)
    experts, topk, hidden, inter = 128, 8, 2048, 768
    infer = kt_kernel_ext.CPUInfer(args.cpuinfer_threads)

    gate = (torch.randn(experts, inter, hidden) / 100).to(torch.bfloat16).contiguous()
    up = (torch.randn(experts, inter, hidden) / 100).to(torch.bfloat16).contiguous()
    down = (torch.randn(experts, hidden, inter) / 100).to(torch.bfloat16).contiguous()
    mapping = torch.arange(experts, dtype=torch.int64).contiguous()

    cfg = kt_kernel_ext.moe.MOEConfig(experts, topk, hidden, inter, 0)
    cfg.max_len = 4096
    cfg.gate_proj, cfg.up_proj, cfg.down_proj = gate.data_ptr(), up.data_ptr(), down.data_ptr()
    cfg.gate_scale = cfg.up_scale = cfg.down_scale = 0
    cfg.pool = infer.backend_
    moe = kt_kernel_ext.moe.AMXBF16_MOE(cfg)
    infer.submit(moe.load_weights_task(mapping.data_ptr()))
    infer.sync()

    ids = torch.stack([torch.randperm(experts)[:topk]]).contiguous()
    weights = (torch.randn(1, topk) / 10).contiguous()
    inp = (torch.randn(1, hidden, dtype=torch.bfloat16) * 3).contiguous()
    out = torch.empty(1, hidden, dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([1], dtype=torch.int32)

    started = time.time()
    infer.submit(moe.forward_task(bsz.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(),
                                  inp.data_ptr(), out.data_ptr(), False))
    infer.sync()
    elapsed = time.time() - started

    ref = torch.zeros(1, hidden, dtype=torch.float32)
    for pos in range(topk):
        e = ids[0, pos].item()
        x = inp.float()
        act = torch.nn.functional.silu(x @ gate[e].float().t()) * (x @ up[e].float().t())
        ref += (act @ down[e].float().t()) * weights[0, pos].item()
    err = (out.float() - ref).abs().sum().item() / ref.abs().sum().item()

    print(f"kernel forward     : {elapsed * 1000:.2f} ms")
    print(f"relative L1 error  : {err * 100:.4f}% vs PyTorch fp32 reference")
    print(f"verdict            : {'PASS' if err < 0.05 else 'FAIL'}")
    return 0 if err < 0.05 else 1


def run_generation(args, device):
    """Load the model in hybrid CPU-GPU placement, generate real text, and report throughput."""
    model, tokenizer, wrappers = load_hybrid_model(args, device)
    if not wrappers:
        print("no MoE layers were offloaded — is this actually an MoE checkpoint?", file=sys.stderr)
        return 1

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    torch.manual_seed(args.seed)
    baseline = vram_used_gb()
    samples, stop_event, sampler = [], threading.Event(), None
    if args.sample_hw:
        sampler = threading.Thread(target=sample_hardware, args=(stop_event, samples), daemon=True)
        sampler.start()

    started = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                   do_sample=args.temperature > 0, temperature=args.temperature or None,
                                   pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - started

    if sampler:
        stop_event.set()
        sampler.join(timeout=3)

    new_tokens = generated.shape[-1] - inputs["input_ids"].shape[-1]
    completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(f"model              : {args.model}")
    print(f"kt_method          : {args.kt_method}")
    print(f"latency            : {elapsed:.2f}s")
    print(f"decode rate        : {new_tokens / elapsed:.2f} tok/s ({new_tokens} new tokens)")
    print(f"gpu vram (card0)   : {baseline:.2f} GB resident after load")
    if samples:
        print(f"cpu utilisation    : peak {max(s[0] for s in samples):.1f}% / mean "
              f"{sum(s[0] for s in samples) / len(samples):.1f}% over {len(samples)} samples")
        print(f"gpu vram during    : peak {max(s[1] for s in samples):.2f} GB")
    print()
    print("--- response ---")
    print(completion.strip())
    return 0


def post_json(url, payload, timeout):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_messages(args):
    """Build the chat message list for a single-turn request."""
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def run_chat(args):
    """Query a running KTransformers/SGLang OpenAI-compatible server and print the completion."""
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": build_messages(args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    started = time.time()
    try:
        result = post_json(url, payload, args.timeout)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0

    print(f"endpoint      : {url}")
    print(f"model         : {result.get('model')}")
    print(f"finish_reason : {choice.get('finish_reason')}")
    print(f"latency       : {elapsed:.2f}s")
    if completion_tokens:
        print(f"decode rate   : {completion_tokens / elapsed:.1f} tok/s ({completion_tokens} completion tokens)")
    print(f"usage         : {usage}")
    print()
    print("--- response ---")
    print((message.get("content") or "").strip())

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return 0


def main():
    args = parse_args()
    if args.mode == "chat":
        return run_chat(args)
    if args.seed is None:
        args.seed = 42
    args.weight_path = args.model
    if not os.path.isdir(args.weight_path):
        from huggingface_hub import snapshot_download
        args.weight_path = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json", "*.txt"])

    gpu_runtime = report_environment()
    flags = cpu_flags()
    print(f"cpu isa            : " + " ".join(f"{k}={'yes' if v else 'NO'}" for k, v in flags.items()))
    print(f"amx fast path      : {'available' if flags.get('amx_tile') else 'UNAVAILABLE (EPYC — AVX512 path used)'}")
    print()

    if args.mode == "probe":
        return 0
    if args.mode == "kernel":
        return run_kernel_check(args)

    device = torch.device(args.device)
    if "HIP" not in gpu_runtime:
        print("warning: kt-kernel is not linked against HIP — the GPU half will not be exercised", file=sys.stderr)
    return run_generation(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
