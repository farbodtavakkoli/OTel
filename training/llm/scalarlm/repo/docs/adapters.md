# Adapters: Training → vLLM Inference

This document describes how post-training adapters travel from the Megatron training loop to a live vLLM inference engine — and what changes ScalarLM makes to the vendored vLLM fork to support it.

Two adapter formats coexist in ScalarLM:

- **Tokenformer** (default). A ScalarLM-native adapter family: key-value memory banks attached to every MLP block. Trained as normal weights in the model's state dict; hot-loaded into vLLM via a custom `TokenformerModelManager` that substitutes into vLLM's LoRA adapter slot.
- **LoRA** (optional). Standard PEFT LoRA. Same transport path as Tokenformer — ScalarLM's fork replaces vLLM's LRU LoRA worker manager with a unified `TokenformerModelManager` that handles both formats through the same `add_lora` / `activate_adapter` API.

Both paths reuse vLLM's `load_lora_adapter` HTTP endpoint as the registration verb. The payload's `lora_path` field points at a training job directory on the shared PVC, which the manager reads to materialize either a `TokenformerModel` (tokenformer weights) or a standard LoRA adapter.

The closed-loop guarantee is that a training job's completion makes its adapter available to the *same* running vLLM process within one `megatron_refresh_period` (default 30 s), with no pod restart.

---

## 1. The Two Sides

```
┌────────────────────────────────────────┐     ┌────────────────────────────────────────┐
│  TRAINING SIDE — ml/ (sbatch job)      │     │  SERVING SIDE — vllm/ (in-process)     │
│                                        │     │                                        │
│  ml/adapters/                          │     │  vllm/tokenformer/                     │
│    add_adapters_to_model.py            │     │    tokenformer_surgeon.py              │
│    create_tokenformer_model.py         │     │    tokenformer_model_manager.py        │
│    create_lora_model.py                │     │                                        │
│                                        │     │  vllm/model_executor/models/           │
│  ml/tokenformer/                       │     │    interfaces.py  → SupportsTokenformer│
│    tokenformer_surgeon.py              │     │    llama.py, qwen2.py, qwen3.py,       │
│    tokenformer_model.py                │     │    qwen3_moe.py, gemma3.py             │
│                                        │     │      └── mix in SupportsTokenformer   │
│  Emits:                                │     │                                        │
│    /app/cray/jobs/{hash}/              │     │  vllm/v1/worker/                       │
│      checkpoint_{step}.pt              │     │    lora_model_runner_mixin.py          │
│      (filtered state dict:             │     │      └── swaps LRUCacheWorkerLoRA      │
│       tokenformer_* + lm_head +        │     │          Manager for                   │
│       attention projections)           │     │          TokenformerModelManager       │
│                                        │     │                                        │
└────────────────────────┬───────────────┘     └────────────────▲───────────────────────┘
                         │                                      │
                         │   shared PVC: /app/cray/jobs/        │
                         │                                      │
                         └──────────────────┬───────────────────┘
                                            │
          ┌─────────────────────────────────▼────────────────────────────────┐
          │  CONTROL PLANE — infra/cray_infra/                              │
          │                                                                  │
          │  training/register_megatron_models.py                            │
          │    scan jobs dir every 30s → VLLMModelManager                    │
          │                                                                  │
          │  api/fastapi/generate/get_adaptors.py                            │
          │    worker pulls delta list on every /v1/generate/get_work        │
          │                                                                  │
          │  one_server/create_generate_worker.py                            │
          │    add_new_adaptor() → LoadLoRAAdapterRequest                    │
          │      → vLLM /v1/load_lora_adapter                                │
          │        → lora_manager.add_adapter                                │
          │          (which is TokenformerModelManager in the fork)          │
          │                                                                  │
          │  adapters/ (the "clean-architecture" replacement tree)           │
          │    common/adapter_commons.py  — re-implemented vLLM base classes │
          │    model/tokenformer.py       — dependency-inverted manager     │
          │    vllm/adapter.py            — integration façade               │
          │    vllm/attention_adapter.py  — monkey-patch helper              │
          │    vllm/registry.py           — model-class registry             │
          └──────────────────────────────────────────────────────────────────┘
```

Two directories named `tokenformer_surgeon.py` exist and both matter. The `ml/` one runs on training nodes; the `vllm/` one runs on serving nodes. They are near-identical copies — the divergence is deliberate, because training reads config from ScalarLM's `get_config()` while serving reads from `TOKENFORMER_R` / `TOKENFORMER_NUM_HEADS` environment variables to keep vLLM free of ScalarLM imports. Keep them in sync when modifying the core algorithm.

---

## 2. Tokenformer: What the Adapter Actually Is

Tokenformer is ScalarLM's preferred post-training format. Conceptually it's LoRA-adjacent — a low-rank additive branch parallel to existing modules — but the branch is an **attention-shaped lookup over learned key-value memory banks**, not a low-rank matrix factorization.

### 2.1 Parameter shape

For every MLP block in the model, one `TokenformerAdapter` is inserted (`ml/tokenformer/tokenformer_surgeon.py:14`):

```python
class TokenformerAdapter(nn.Module):
    # hidden_size  = base model hidden dim
    # num_heads    = tokenformer_num_heads (default 4)
    # tokenformer_r = memory-bank rank (default 32)
    tokenformer_k = nn.Parameter(zeros(num_heads, hidden_size))
    tokenformer_v = nn.Parameter(zeros(num_heads, hidden_size * tokenformer_r))
    tokenformer_p = nn.Parameter(zeros(tokenformer_r, hidden_size))
```

Init (`reset_parameters`, L38):

- `k`: normal(0, σ_k) with `σ_k = 3 / √(hidden_size/num_heads)`
- `v`: uniform(-σ_v, σ_v) with `σ_v = 3 / √hidden_size`
- `p`: zeros — crucial, because `p` is the final projection multiplier, so the adapter is **exactly zero at init** and can't perturb the base model before training starts.

### 2.2 Forward pass

`TokenformerAdapter.forward` (`ml/tokenformer/tokenformer_surgeon.py:47`) runs the original MLP block and sums its output with a tokenformer branch:

```python
def forward(self, hidden_states, *args, **kwargs):
    base = self.layer(hidden_states, *args, **kwargs)
    adapter = self.tokenformer_op(hidden_states)
    return base + adapter            # plus tuple bookkeeping
```

`tokenformer_op` (L67) treats `hidden_states` as queries against the `(k, v)` memory banks, applies scaled-dot-product attention, then projects the result down through `tokenformer_p`:

```
q    = reshape(query)             → [num_heads, seq_len, head_dim]
k    = reshape(tokenformer_k)     → [num_heads, memory_slots, head_dim]
v    = reshape(tokenformer_v)     → [num_heads, memory_slots, head_dim × r]
result = SDPA(q, k, v)            # non-causal
result = result · tokenformer_p   # [seq_len, hidden_size] output
return result.view_as(query)
```

The intuition: tokenformer_k and tokenformer_v act like a learned key-value memory that every hidden state can attend over; tokenformer_p projects the attended value back into hidden-state space. It scales to total parameter counts comparable to LoRA(r=32–64) but with a different inductive bias — input-conditional memory lookup rather than low-rank delta.

### 2.3 Which layers get wrapped

`TokenformerSurgeon._is_adapter_layer` (L118):

```python
def _is_adapter_layer(self, layer_name):
    return "mlp" in layer_name.split(".")[-1]
```

Only modules whose final name component contains `mlp` — typically `model.layers.{i}.mlp`. Attention is not wrapped; instead, the base attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and norms are *unfrozen* during training (see §3.1).

`insert_adapter_modules` (L155) walks `model.named_modules()` once and swaps each match in-place via `_recursive_setattr`. The model is mutated; no wrapper class around the whole model is created.

---

## 3. Training-Side Wiring

### 3.1 Dispatch — `ml/adapters/add_adapters_to_model.py`

```python
def add_adapters_to_model(model, device):
    job_config = get_job_config()
    if   job_config["adapter_type"] == "tokenformer": return create_tokenformer_model(model, device)
    elif job_config["adapter_type"] == "lora":        return create_lora_model(model, device)
    elif job_config["adapter_type"] == "none":        return model
    else: raise ValueError(...)
```

Called from the training loop's model-loading path. `adapter_type` comes from the per-job config (default `tokenformer`; see `docs/configuration.md` §2.3).

### 3.2 Tokenformer training setup — `create_tokenformer_model.py:8`

Six steps:

1. **Surgery** — `TokenformerSurgeon(model, device).insert_adapter_modules()` mutates the model.
2. **Auto-decide lm_head training** — if model has <100M params, `train_lm_head = True`. The justification (comment at L23): gradient scale for the lm_head is tricky on big models, and the lm_head itself is huge which would inflate checkpoint size. Small models benefit from co-training it.
3. **Freeze all** — `param.requires_grad = False` for every parameter.
4. **Selectively unfreeze** — modules whose name contains any of:
   ```
   tokenformer, q_proj, k_proj, v_proj, norm, rotary_emb,
   embed_tokens, input_layernorm, post_attention_layernorm, o_proj
   ```
   That's the full attention mechanism plus normalizations and embeddings — the entire "routing" fabric of the model, plus the new tokenformer weights. The MLPs themselves stay frozen (except for their tokenformer branches). This is what makes the adapter set expressive beyond LoRA-on-attention.
5. **lm_head handling** — if `train_lm_head`, unfreeze `lm_head.weight` directly (handles tied-embedding models where lm_head isn't a separate param).
6. **Logging** — per-step timings and final `trainable/total` summary.

### 3.3 LoRA training setup — `create_lora_model.py:11`

Standard PEFT:

```python
lora_config = job_config["lora_config"]   # r, lora_alpha, lora_dropout, target_modules
lora_model = get_peft_model(model, LoraConfig(**lora_config))
add_methods(lora_model)                    # patches .unwrap_model onto it
```

`add_methods` (L137) attaches a custom `unwrap_model` that calls `filter_checkpoint` to keep only parameters with `requires_grad=True` in the saved state dict — mirrors the tokenformer flow so that saved checkpoints are small and contain only adapter deltas.

### 3.4 What lands in the checkpoint

`TrainingLoop.checkpoint` (in `ml/cray_megatron/megatron/training_loop.py`, see `docs/training-lifecycle.md` §5.1) uses `model.unwrap_model()` when available:

- For LoRA: `filter_checkpoint` keeps only `requires_grad=True` params → LoRA A/B matrices + optionally lm_head.
- For Tokenformer: same mechanism keeps tokenformer_*, attention projections, norms, embeddings, optional lm_head.

The result is saved as `checkpoint_{step}.pt` under `{job_directory}/`. For a Llama-7B + Tokenformer job this is typically 200–800 MB — large compared to a LoRA-only checkpoint (~20 MB) but small relative to the 14 GB base. The vLLM-side manager later loads only the *delta* keys, never the full state dict.

---

## 4. The vLLM Fork — What ScalarLM Changed

The vendored fork at `vllm/` is vLLM v0.19.0 + ScalarLM changes. Everything squashed into one commit labeled "ScalarLM changes squashed onto v0.19.0". The full list of touched files is in the squash commit; the load-bearing ones:

### 4.1 New package: `vllm/tokenformer/`

Three files added under vLLM's own source tree (not a separate plugin):

| File | Role |
|---|---|
| `vllm/tokenformer/__init__.py` | Package marker. |
| `vllm/tokenformer/tokenformer_surgeon.py` | `TokenformerSurgeon` + `TokenformerAdapter`. Near-copy of `ml/tokenformer/tokenformer_surgeon.py`. **Reads config from `TOKENFORMER_R` and `TOKENFORMER_NUM_HEADS` env vars** (L20, L22) — no ScalarLM imports. |
| `vllm/tokenformer/tokenformer_model_manager.py` | `TokenformerModelManager` — implements vLLM's adapter-manager API on top of tokenformer weights. |

The env-var choice keeps the fork's runtime free of `cray_infra` imports so the vLLM CLI still runs standalone.

### 4.2 New interface: `SupportsTokenformer`

`vllm/model_executor/models/interfaces.py:1438`:

```python
@runtime_checkable
class SupportsTokenformer(Protocol):
    supports_tokenformer: ClassVar[Literal[True]] = True
```

Plus the symmetric `supports_tokenformer(model)` runtime check (L1645–1655) and export from `vllm/model_executor/models/__init__.py` (L14, L45). This mirrors how `SupportsLoRA`, `SupportsPP`, `SupportsEagle` are declared upstream.

### 4.3 Model-class mixins

Five model families advertise tokenformer support by adding the mixin to their `ForCausalLM` class:

| Model | Class | File |
|---|---|---|
| Llama | `LlamaForCausalLM(..., SupportsTokenformer)` | `vllm/model_executor/models/llama.py:508` |
| Qwen2 | uses the interface via qwen3; `qwen2.py` adds a `state_dict` override for qkv_proj unpacking | `qwen2.py:601` |
| Qwen3 | `Qwen3ForCausalLM(..., SupportsTokenformer)` | `qwen3.py:273` |
| Qwen3-MoE | `Qwen3MoeForCausalLM(..., SupportsTokenformer)` | `qwen3_moe.py` |
| Gemma3 | `Gemma3ForCausalLM(..., SupportsTokenformer)` | `gemma3.py:455` |

Qwen2 is special: it doesn't get the mixin but it does get a custom `state_dict` method that unpacks fused `qkv_proj.weight` into separate `q_proj`/`k_proj`/`v_proj` keys, and filters out MoE expert keys. That's because training-time weights are produced by HuggingFace's unfused layout, so the inference-time state dict needs to match when adapter weights are merged in.

### 4.4 Worker-level manager swap

`vllm/v1/worker/lora_model_runner_mixin.py` is the key serving-side change.

Upstream vLLM instantiates `LRUCacheWorkerLoRAManager` to manage adapters. ScalarLM's fork replaces that with `TokenformerModelManager`:

```python
# Before (upstream):
self.lora_manager = LRUCacheWorkerLoRAManager(vllm_config, device, model.embedding_modules)
return self.lora_manager.create_lora_manager(model, vllm_config)

# After (ScalarLM fork):
self.lora_manager = TokenformerModelManager(model=model, device=device)
return self.lora_manager.model
```

And the dynamic `add_lora` path at L351 has a fallback that spins up a `TokenformerModelManager` even when LoRA wasn't enabled at startup:

```python
def add_lora(self, lora_request):
    self._ensure_lora_enabled()
    if not self.lora_manager:
        if hasattr(self, 'model') and hasattr(self, 'device'):
            from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
            self.lora_manager = TokenformerModelManager(model=self.model, device=self.device)
    return self.lora_manager.add_adapter(lora_request)
```

The ScalarLM convention is: the vLLM-internal attribute is still called `lora_manager` for signature compatibility with the rest of the LoRA code path, but the concrete class is `TokenformerModelManager`. It happens to also implement the LoRA interface; see §5.3.

### 4.5 New KV-cache introspection APIs

`vllm/v1/engine/async_llm.py:1013` adds `AsyncLLM.get_current_kv_cache_size()` and `get_total_kv_cache_tokens()`. These are what the Generate Worker calls to size its batch pulls (see `docs/inference-queue.md` §4.3 — `get_batch_size` divides `current` by the runtime engine's `model_config.max_model_len`).

The plumbing: `AsyncLLM.get_current_kv_cache_size()` → `engine_core.get_free_kv_cache_tokens_async()` → RPC → `EngineCore.get_free_kv_cache_tokens()` (`vllm/v1/engine/core.py:290`), which does:

```python
free_blocks = kv_manager.block_pool.get_num_free_blocks()
block_size  = kv_manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
return free_blocks * block_size
```

Not strictly adapter-related, but it's the other half of why the ScalarLM fork exists: without this the Generate Worker couldn't do its KV-cache-feedback pull sizing.

### 4.6 Minor fixes that travel with the squash

- `vllm/model_executor/layers/linear.py` / `vocab_parallel_embedding.py` — small compatibility tweaks.
- `vllm/model_executor/layers/fla/ops/solve_tril.py` — new file (hybrid-model support needed for tokenformer on Gemma3).
- `vllm/platforms/__init__.py` — platform detection tweak.
- `csrc/cpu/*` — CPU backend adjustments for the `cpu` build target.

These are load-bearing for the build to succeed on all targets, not for adapter semantics.

---

## 5. The vLLM-Side Manager: `TokenformerModelManager`

This is where the closed loop actually closes. File: `vllm/tokenformer/tokenformer_model_manager.py`.

### 5.1 Construction

```python
class TokenformerModelManager:
    def __init__(self, model: SupportsLoRA, device: torch.device):
        if supports_tokenformer(model):
            self.model = TokenformerSurgeon(model, device).insert_adapter_modules()
        else:
            self.model = model
        self._registered_adapters = {}
        self._active_adapter = None
        self.dtype = next(self.model.parameters()).dtype
        self.original_tensors = {}
        self._lru_adaptor_ids = []
```

On init, if the model advertises `SupportsTokenformer`, the manager **runs the surgeon on the vLLM model immediately** — same `TokenformerAdapter` code as training, same wrapping rule (`mlp` suffix). This prepares the model to *receive* tokenformer weights even though none are loaded yet. At init, `tokenformer_p` is zero everywhere, so the adapter contributes nothing.

If the model doesn't support tokenformer, the manager is still valid but passes the model through. `add_adapter` then still works for LoRA adapters.

### 5.2 `add_adapter` — load from disk

```python
def add_adapter(self, request) -> bool:
    lora_path = get_adapter_absolute_path(request.lora_path)
    tokenformer = TokenformerModel.from_local_checkpoint(lora_path, device=self.device)

    if len(self._registered_adapters) >= self.capacity:
        lru_adapter_id = self._lru_adaptor_ids.pop(0)
        self.remove_adapter(lru_adapter_id)

    self._registered_adapters[request.adapter_id] = tokenformer
    self._lru_adaptor_ids.append(request.adapter_id)
    return True
```

`TokenformerModel.from_local_checkpoint` (file L27) globs `*.pt` in the directory, picks the first, loads `state_dict["model_state_dict"]`, copies it to device. Every key is retained — the upstream LoRA-specific filtering is absent here, which is why lm_head and attention projections ride along cleanly.

Capacity comes from `TOKENFORMER_CACHE_CAPACITY` env var (default 4, L202). LRU eviction on overflow: oldest adapter removed first.

### 5.3 `activate_adapter` — swap weights into the running model

This is the hot-swap core (L74):

```python
def activate_adapter(self, adapter_id: int) -> bool:
    if adapter_id == self._active_adapter: return False
    model_state_dict = self.model.state_dict()
    tokenformers = self._registered_adapters[adapter_id].tokenformers

    # Save original tensors the first time we touch them
    for key in tokenformers:
        if key not in self.original_tensors and key in model_state_dict:
            self.original_tensors[key] = copy.deepcopy(model_state_dict[key])

    # Restore originals (undoes the previous adapter if any)
    for key, value in self.original_tensors.items():
        model_state_dict[key] = value

    # Apply this adapter
    for key, value in tokenformers.items():
        if 'lora' in key: continue
        model_state_dict[key] = value

    self.model.load_weights(model_state_dict.items())
    process_weights_after_loading(self.model, self.model.model_config, self.device)

    self._active_adapter = adapter_id
    return True
```

Three subtleties:

1. **Original-tensor preservation.** The first time a key is overwritten, its base-model value is cached in `self.original_tensors`. On the next activation, originals are restored *before* the new adapter is applied. This keeps the base model clean — activating A, then B, then A again produces the same weights as activating A directly.
2. **`process_weights_after_loading`.** After `load_weights`, vLLM needs to re-run quantization, parameter sharding, and tensor-parallel resharding. Without this, swapped weights don't reach the GPU kernels in the expected layout.
3. **`'lora' in key` skip.** Lets the manager ignore LoRA-specific keys that might be present in a combined checkpoint. The tokenformer path is the primary one; LoRA is served through the same checkpoints-on-disk mechanism but handled by the LoRA code elsewhere in the stack.

`deactivate_adapter` (L117) is the symmetric operation: zero out `tokenformer_p` (so the adapter contributes nothing), restore originals, call `load_weights` + `process_weights_after_loading`. Used when a worker explicitly wants to revert to the base model.

### 5.4 The manager implements LoRA's adapter-manager interface

Method names match what vLLM's worker expects from a LoRA manager: `add_adapter`, `activate_adapter`, `deactivate_adapter`, `remove_adapter`, `pin_adapter`, `list_adapters`, `get_adapter`, `remove_all_adapters`, `set_active_adapters`, `set_adapter_mapping`, `dummy_lora_cache`, `add_dummy_lora`. That's why the worker-level swap in §4.4 is non-invasive: vLLM's LoRA code path calls `self.lora_manager.add_adapter(...)` and doesn't care whether the instance is LRU-LoRA or Tokenformer.

---

## 6. Transport: How an Adapter Reaches the Engine

Six hops, all on the same pod (or across pods in Helm with the shared jobs PVC):

```
[1] Training job writes
     /app/cray/jobs/{hash}/checkpoint_{step}.pt
            │
[2] register_megatron_models (periodic, every 30s)
      scans jobs dir → adds each hash to VLLMModelManager
            │
[3] Client POST /v1/generate with model={hash}
      generate.py resolves via VLLMModelManager.find_model
            │
[4] Generate Worker's next get_work call includes
      loaded_adaptor_count=N
            │
[5] get_adaptors.py computes delta:
      new_adaptors = registered_models[N:]
      Worker receives the list in its get_work response
            │
[6] Worker calls add_new_adaptor for each:
      LoadLoRAAdapterRequest(lora_name=hash, lora_path={jobs_dir}/{hash})
       → vLLM load_lora_adapter
         → lora_manager.add_adapter (= TokenformerModelManager.add_adapter)
           → TokenformerModel.from_local_checkpoint(path)
             → torch.load(*.pt), copy to device
```

### 6.1 Step 2 — control-plane registration

`infra/cray_infra/training/register_megatron_models.py:16` is called from the FastAPI lifespan task every `megatron_refresh_period`. It walks `training_job_directory`, picks every subdirectory containing any `*.pt`, and registers it with `VLLMModelManager` (see `docs/architecture.md` §6.1 and `docs/configuration.md`). The manager is an in-memory `SortedDict` keyed by job start time, singleton via `get_vllm_model_manager()`.

### 6.2 Step 3 — request-time model resolution

`VLLMModelManager.find_model` has three tiers (infra/cray_infra/training/vllm_model_manager.py:29):

1. If `model_name == config["model"]`, it's the base model — return as-is, no adapter.
2. If `model_name` is in `_models.values()`, it's already registered — return.
3. Otherwise check `training_job_directory/{model_name}`. If it exists *and* contains `.pt` files, auto-register and return. This is the escape hatch for requests that arrive faster than the 30 s refresh period.

### 6.3 Step 4 — worker/control-plane adapter delta

Every `/v1/generate/get_work` request from the worker carries a `loaded_adaptor_count: N`. The handler calls `get_adaptors(request)` (`infra/cray_infra/api/fastapi/generate/get_adaptors.py:11`):

```python
async def get_adaptors(request):
    already_loaded = request.loaded_adaptor_count
    registered = get_vllm_model_manager().get_registered_models()
    new_adaptors = registered[already_loaded:]
    return GetAdaptorsResponse(new_adaptors=new_adaptors)
```

The list is returned piggy-backed on the work response, so adapter discovery doesn't cost an extra round-trip. The worker increments its local `loaded_adaptor_count` after each successful load, so subsequent pulls get only the delta.

This works because `VLLMModelManager` preserves registration order (SortedDict by start time). The worker doesn't need to know which are new — slicing by count is sufficient.

### 6.4 Step 6 — the actual load

`infra/cray_infra/one_server/create_generate_worker.py:215`:

```python
async def add_new_adaptor(app, new_adaptor):
    config = get_config()
    new_adaptor_path = os.path.join(config["training_job_directory"], new_adaptor)

    lora_adaptor_request = LoadLoRAAdapterRequest(
        lora_name=new_adaptor,
        lora_path=new_adaptor_path,
    )
    raw_request = Request(scope={"app": app, ...}, receive=pass_receive)
    response = await load_lora_adapter(lora_adaptor_request, raw_request=raw_request)
```

The LoRA adapter endpoint is called **in-process** — the worker has a reference to the vLLM `app` and invokes the handler function directly, not through HTTP. This is the same pattern the worker uses for chat completions and it's enabled by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` (set unconditionally at `infra/cray_infra/one_server/main.py:4`).

Once the call succeeds, the `TokenformerModelManager` has registered the adapter but not activated it. Activation happens automatically on the first request that specifies it, via `set_active_adapters` + `activate_adapter`.

### 6.5 Failure handling

If `add_new_adaptor` raises or returns a non-200, the worker logs and continues (`create_generate_worker.py:206`): `loaded_adaptor_count` does not increment, so the same adapter will be retried on the next `get_work` tick. This is how transient disk/load errors self-heal.

If the checkpoint file is present but malformed, the `load_weights` call will raise and the adapter ends up registered-but-broken. Workers won't serve it cleanly; operator intervention is currently required (delete the job directory to trigger re-training, or fix the checkpoint).

---

## 7. The `infra/cray_infra/adapters/` "Clean-Architecture" Tree

A separate, parallel implementation lives under `infra/cray_infra/adapters/`. Four subdirs:

- `common/adapter_commons.py` — abstract `AdapterModel` / `AdapterModelManager` classes + a `ConfigProvider` Protocol + global singleton. Deliberately mirrors what vLLM's upstream `adapter_commons` module provides, so that ScalarLM code can import from here without touching vLLM internals.
- `model/tokenformer.py` — `TokenformerManager` using dependency-injected config, with the same `add_adapter`/`activate_adapter`/LRU semantics as the in-fork version.
- `model/models.py` — factory-function registry for Llama/Gemma/Qwen2 "ScalarLM-enhanced" variants (referenced from `vllm/registry.py`).
- `vllm/adapter.py` — `AdapterManager`, `ScalarLMAdapter`, `EnhancedModelWrapper` — a façade that wraps a vLLM model with ScalarLM functionality via composition, not inheritance.
- `vllm/attention_adapter.py` — `VLLMAttentionAdapter` + `patch_vllm_attention_layer` that monkey-patches a vLLM attention layer's `forward`.
- `vllm/registry.py` — `ScalarLMModelRegistry` for registering model factory functions.

The docstrings in these files (e.g. `adapter.py:2` "vLLM has ZERO knowledge of ScalarLM - all coupling is eliminated") make the intent explicit: this tree is a **dependency-inverted alternative** to the in-fork `vllm/tokenformer/` implementation. It uses `Protocol` types and injected callables to avoid requiring vLLM to import any ScalarLM code.

The hot path today goes through the in-fork `vllm/tokenformer/tokenformer_model_manager.py`. The `infra/cray_infra/adapters/` tree is in the codebase and imported by some periphery code, but the worker's `add_new_adaptor` flow (§6.4) uses the in-fork manager via vLLM's standard LoRA endpoint. Treat the clean-architecture tree as the migration target if you want to reduce fork drift in the future.

---

## 8. Configuration

### 8.1 Server-side

From `docs/configuration.md` §1.3:

| Field | Default | Effect |
|---|---|---|
| `tokenformer_r` | 32 | Memory-bank rank. Training side reads this; serving side reads `TOKENFORMER_R` env var. Keep them in sync. |
| `tokenformer_num_heads` | 4 | Attention heads in the memory-bank lookup. Serving side reads `TOKENFORMER_NUM_HEADS` env var. |
| `tokenformer_cache_capacity` | 2 | Max adapters held hot. Serving side reads `TOKENFORMER_CACHE_CAPACITY` env var (default 4 there). The mismatched default is a real source of confusion — set explicitly in production. |

The env-var handoff is a deliberate seam: the vLLM fork must not `import cray_infra.util.get_config`, so the control plane sets env vars before spawning vLLM. In practice these env vars are set at pod entry (e.g., in the Dockerfile or the Helm Deployment's `env:` block) and the vLLM tokenformer code reads them lazily.

### 8.2 Per-job

From `docs/configuration.md` §2.3:

| Field | Default | Options |
|---|---|---|
| `adapter_type` | `tokenformer` | `tokenformer` / `lora` / `none` |
| `lora_config.r` | 32 | LoRA rank (only used when `adapter_type=lora`) |
| `lora_config.lora_alpha` | 32 | |
| `lora_config.lora_dropout` | 0.1 | |
| `lora_config.target_modules` | `"all-linear"` | or explicit list like `["q_proj","v_proj"]` |

---

## 9. Pitfalls

**Two surgeons, not one.** `ml/tokenformer/tokenformer_surgeon.py` and `vllm/tokenformer/tokenformer_surgeon.py` must produce the same wrapping or adapters don't load. Any change to `_is_adapter_layer`, the parameter shapes, or the forward math must land in both. The vLLM-side copy uses env vars instead of `get_config()` — don't try to unify that; it's what keeps vLLM importable without ScalarLM.

**Capacity default mismatch.** `infra/cray_infra/util/default_config.py:62` says `tokenformer_cache_capacity = 2`; `vllm/tokenformer/tokenformer_model_manager.py:202` defaults `TOKENFORMER_CACHE_CAPACITY` to 4. In a naive deployment the serving side will cache more than the control plane thinks it should.

**`'lora' in key` skip.** In `activate_adapter`, keys containing `lora` are skipped. A Tokenformer checkpoint that accidentally names weights with `lora` in them (e.g. a directory named `lora-something/`) will silently drop those weights at activation time. Training-side naming avoids this, but be careful when hand-crafting checkpoints.

**`process_weights_after_loading` is mandatory.** If you add a new activation path that calls `load_weights` without this follow-up, quantized and tensor-parallel setups will produce garbage output. Always pair them.

**Tokenformer freezes the MLP but not its inputs/outputs.** The set of unfrozen module names in `create_tokenformer_model.py:52-65` includes `q_proj/k_proj/v_proj/o_proj` and the norms. That's a substantial fraction of the model — be aware that tokenformer training updates more than just the tokenformer parameters. This is why checkpoints are hundreds of MB, not tens.

**Auto-registration via `find_model` writes to shared state.** If two concurrent requests hit `find_model` with the same previously-unknown model name, both will `register_model`. The `SortedDict` handles this safely (start_time keys), but you get a duplicate log line.

**LRU eviction deactivates.** When the adapter cache is full and the LRU entry is evicted, its `remove_adapter` calls `deactivate_adapter`, which restores original weights. If the evicted adapter was active, the next request specifying it will re-add and re-activate, paying the checkpoint load cost again. Set `tokenformer_cache_capacity` at least as high as your steady-state concurrent-adapter count.

**Worker-level swap is not the in-fork adapter-manager replacement.** The `infra/cray_infra/adapters/` tree is a *parallel* implementation, not the active one. Changes made only there won't affect the hot path. Changes to `vllm/tokenformer/tokenformer_model_manager.py` do affect the hot path — they also require rebuilding the vLLM wheel (`docker build`).

**Fork-upstream merge conflicts.** The squash commit message documents three real conflicts: `interfaces.py` (added `SupportsTokenformer` alongside `SupportsEncoderCudagraph`), `llama.py` (dropped ScalarLM's hand-rolled eagle methods in favor of v0.19.0's `EagleModelMixin`), `qwen3.py`/`qwen3_moe.py` (merged `SupportsEagle` + `SupportsTokenformer` into the same class bases). Future rebases should expect similar conflicts on any model added to §4.3.
