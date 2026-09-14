# H100 (NVIDIA / CUDA) inference notes

Cross-cutting lessons from serving the inference stacks on **NVIDIA H100 80GB HBM3**
(Hopper cc 9.0, CUDA 13.0), plus the ROCm→CUDA reversals. The per-stack support table lives
in the top-level [README](../README.md). Companion AMD notes:
[mi355x_inference_notes.md](mi355x_inference_notes.md) — its silent-corruption traps and the
"demand GPU residency" rule are hardware-neutral and apply here too. Target models: LLM
`Qwen/Qwen3.8-27B-FP8`, embedding `google/embeddinggemma-300m`, reranker
`Qwen/Qwen3-Reranker-0.6B`, plus GGUF equivalents.

Covered here: `transformers` (baseline), `tensorrtllm`, `llamacpp`, `sglang`.

## A proxy-gated host is not an offline host

On a host whose egress goes through a corporate proxy set via
`HTTP_PROXY`/`HTTPS_PROXY`, that proxy may **403** every NVIDIA/HF host
(nvcr.io, pypi.nvidia.com, download.pytorch.org, huggingface.co) while pypi.org stays
allowlisted, so `pip install torch` works with the proxy on but anything NVIDIA-hosted needs
the proxy **unset**: `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`
→ those hosts then return 200/401. Docker (`docker.io` + `nvidia-container-toolkit`) installs
via apt and NGC images pull once the proxy is unset.

## TensorRT-LLM: the one stack AMD cannot run at all

NVIDIA-only, so this is the stack with no ROCm counterpart. On CUDA:

- `pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-llm` (proxy unset) →
  **tensorrt-llm 1.2.1** + nvidia-modelopt 0.37.0 + flashinfer. It clobbers torch to
  2.9.1+cu128 (which runs fine on a cu130 driver). It needs a **system MPI**: without
  `libmpi.so`, `import tensorrt_llm` dies `cannot load MPI library`; `sudo apt install
  libopenmpi3 openmpi-bin` fixes it.
- `import tensorrt_llm` then **succeeds** and the folder's own `probe_tensorrtllm.py`
  **exits 0**.
- **The target checkpoint does not load.** `Qwen/Qwen3.8-27B-FP8` declares
  `model_type: qwen3_5`, which TRT-LLM 1.2.1 (and transformers 4.57.3) do not register, so it
  fails at **config parse** (`KeyError: 'qwen3_5'`) before FP8/weights — a version gap, not
  FP8 or hardware, so modelopt requant would not help. A supported arch
  (`Qwen3ForCausalLM`) loads and generates on the PyTorch backend.

## SGLang: the pip route works on CUDA, and it serves the 27B

On ROCm the pip route is structurally dead (`from sgl_kernel import …`, CUDA-only wheels).
On CUDA the pip route **works**: `pip install "sglang[all]"` → sglang 0.5.18 +
**sgl_kernel 0.4.6.post1**, and `import sgl_kernel` succeeds. No container needed; torch stays
2.13.0+cu130.

**SGLang 0.5.18 loads `Qwen/Qwen3.8-27B-FP8` where TRT-LLM 1.2.1 could not** — the `qwen3_5`
hybrid Gated-DeltaNet VL arch is registered in-tree (transformers stays 5.12.1; the
5.8.0.dev0 pin is not required). It loads FP8 natively (E4M3) and generates correct text. One
80 GB VRAM quirk: the mamba cache auto-caps `--max-running-requests` to ~24; raise it with
`--mamba-ssm-dtype bfloat16`. All three leaves (llm/embedding/reranker) serve from one pip
venv.

## llama.cpp: `-DGGML_CUDA=ON`, and proving GPU residency

Build is the clean reversal of the ROCm recipe: `cmake -B build -DGGML_CUDA=ON
-DLLAMA_OPENSSL=ON` (keep OpenSSL — vendor-neutral, needed for `-hf` downloads; drop
`-DGGML_HIP=ON -DGPU_TARGETS=gfx950`). cmake auto-detects `CMAKE_CUDA_ARCHITECTURES=90-real`
(Hopper); no arch flag needed. All three leaves work; check GPU residency with
`offloaded N/N layers to GPU` + `CUDA0 model buffer` + `nvidia-smi` by PID. Note:
`cmake`/`ninja` may be missing on the host (pip-install them); a
network share can reject the rename operations pip and `-hf` use, in which case the build
tree + `LLAMA_CACHE` must live on local disk or tmpfs.

## Cross-cutting reversals and traps

- **Demand GPU-residency proof (hardware-neutral, from the AMD notes and reconfirmed).** A
  backend can report active and return correct numbers while on CPU. Insist on a concrete
  signal sampled from inside the job: `nvidia-smi` VRAM by PID, llama.cpp `offloaded N/N
  layers`, Ollama `100% GPU`. On a shared box an external sampler can also catch the
  co-tenant's PID — sample by PID/UUID.
- **Do not assume GGUF files are already cached.** If `$HF_HOME` holds no `*-GGUF` repos, the
  GGUF stacks (llamacpp, ollama, lemonade) must **download** a GGUF (proxy unset) — a small
  one (few hundred MB) serves for a smoke test; the 29 GB `unsloth/Qwen3.8-27B-GGUF` Q8_0 is
  the production model and fits one H100.
- **The prebuilt flash-attn 2.8.3 wheel is ABI-broken against torch 2.13/cu130**
  (`undefined symbol _ZN3c10…materialize_cow_storage`). The fix is a source rebuild or
  `attn_implementation=sdpa`. Only bare imports at module load (e.g. verl/openrlhf actor
  code) force the issue.
- **Reverse the ROCm knobs:** plain `CUDA_VISIBLE_DEVICES` (drop `HIP_VISIBLE_DEVICES`); the
  AITER `VLLM_ROCM_USE_AITER` GEMM trap is gfx950-specific and N/A here (but still verify
  output is coherent, not token-salad — the silent-corruption discipline carries over).
- **A clean `pip install` proves nothing (hardware-neutral).** `pip install tensorrt-llm`
  installs a placeholder that then pulls from pypi.nvidia.com; test the import and a real
  device op, never just the install.

## FP8 on Hopper

The 27B FP8 checkpoint (OCP **E4M3FN**) loads and serves natively wherever the engine
registers the `qwen3_5` arch (SGLang 0.5.18 ; TRT-LLM 1.2.1 — version, not hardware). FP8
is native on Hopper, so the constraint is arch/version support in each engine, not the
datatype.
