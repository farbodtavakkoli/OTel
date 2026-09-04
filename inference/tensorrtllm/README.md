# TensorRT-LLM — WORKS on NVIDIA H100 (was NOT SUPPORTED on the AMD campaign box)

> **⭐ 2026-08-22 — VERDICT FLIPPED.** TensorRT-LLM has now been re-run on an **8×NVIDIA
> H100 80GB HBM3** node (driver 580.173.02, CUDA 13.0). **`import tensorrt_llm` succeeds
> (v1.2.1) and the probe exits 0 — TensorRT-LLM is usable on H100.** The exact install recipe,
> versions, model-load outcome, and real logs are in
> **[`llm/README.md` → "H100 result (2026-08-22)"](llm/README.md#h100-result-2026-08-22--tensorrt-llm-works)**.
> Everything below is the **original AMD/MI355X record — retained but superseded** on NVIDIA
> hardware. It remains accurate for a *pure-AMD* box: there is still no ROCm build of
> TensorRT-LLM upstream.

**TensorRT-LLM** is NVIDIA's high-performance LLM serving stack: compiled engines,
vendor-tuned kernels, FP8/FP4 quantization, TP/PP/DP/EP parallelism, and Triton/Dynamo
integration. On an NVIDIA cluster it is a top-tier candidate. **On this host — 8×AMD
Instinct MI355X (gfx950), ROCm 7.2.4 — it is not supported, by design:** upstream's
`supported-hardware.md` lists NVIDIA architectures only (Blackwell/Hopper/Ada/Ampere), and
the entire pip dependency closure is CUDA 12/13 wheels. There is no ROCm build to try.

The verdict is settled twice over, independently: (a) this box has no NVIDIA GPU, driver,
`nvidia-smi`, or `libcuda.so.1`; (b) upstream targets NVIDIA silicon exclusively. See the
leaf for the full evidence record — including the sharp negative that `pip install
tensorrt-llm` *succeeds* (exit 0, ~16 GB of CUDA wheels) on a pure-AMD box and only fails
at `import tensorrt_llm` with `ImportError: libcuda.so.1`.

For LLM serving on this AMD host use [`../vllm/llm/`](../vllm/llm/) or
[`../sglang/llm/`](../sglang/llm/) instead.

## Scope — why only an `llm/` leaf exists

Deliberate, and **not** merely a consequence of the hardware. The embedding
(`google/embeddinggemma-300m`) and reranker (`Qwen/Qwen3-Reranker-0.6B`) workloads are
❌ upstream in TensorRT-LLM **even on NVIDIA hardware** — no first-class recipe exists.
Embedding/reranker leaves here would be doubly blocked (wrong vendor *and* unsupported
workload), so they were never created.

## Leaves

| Leaf | Verdict |
|---|---|
| [`llm/`](llm/README.md) | **NOT SUPPORTED** — no NVIDIA GPU here and no ROCm build upstream; full host evidence, pip-install negative result, and a ready-to-run NVIDIA recipe for the future. |

## Environment & requirements

- Client/probe deps: [`requirements.txt`](requirements.txt) (just `python-dotenv`; the
  commented NVIDIA install route lives there too).
- Venv convention: `python3 -m venv .env_tensorrtllm` at this software root. Per-campaign
  venvs were removed in the 2026-08 reorg.
- Secrets/ports: `dev.env` at this root; leaves symlink it (`ln -sf ../../../dev.env` from
  a leaf against the repo-root `dev.env`, or use the local copies as committed).

## Other hardware (upstream claims — not verified here)

NVIDIA-only: upstream supports NVIDIA Blackwell, Hopper, Ada Lovelace, and Ampere GPUs
exclusively; there is no AMD/ROCm, Apple, or CPU backend, and none is planned.
