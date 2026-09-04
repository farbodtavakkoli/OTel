#!/usr/bin/env python3
"""Apply ScalarLM's vLLM-fork patches in-place.

Invoked from both:
  - scripts/build-copy-vllm.sh at Docker build time (production path)
  - Kubernetes pod `command:` at startup (dev/bench iteration)

Adds new patches here. Each patch is a function that takes the vLLM tree
root, reads the target file, asserts the exact anchor it expects, and
writes the transformed file back. The assertions are load-bearing — they
fail loudly when a fork rebase drifts the source rather than silently
producing a mis-patched image.

Usage: apply_patches.py <path/to/vllm/root>
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_output_handler_metrics_offload(vllm_root: Path) -> None:
    """Phase 6.5: move `logger_ref[0].record(...)` out of async_llm's output
    loop onto a background consumer coroutine fed by a bounded asyncio.Queue.

    Follows the TODO vLLM itself left on `vllm/v1/engine/async_llm.py:700`.
    Profile: the synchronous record call was 20.3 %% of main-thread time at
    N=100 and the consumer-offload pattern recovered +15 %% throughput in
    the pilot A/B. See enhance-openai-api.md § "Phase 6.5".
    """
    target = vllm_root / "vllm" / "v1" / "engine" / "async_llm.py"
    src = target.read_text()

    # --- Anchor 1: the inline record call we're replacing. ---
    anchor_record = (
        "                    # 4) Logging.\n"
        "                    # TODO(rob): make into a coroutine and launch it in\n"
        "                    # background thread once Prometheus overhead is non-trivial.\n"
        "                    if logger_ref[0]:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=outputs.engine_index,\n"
        "                            scheduler_stats=outputs.scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=renderer.stat_mm_cache(),\n"
        "                        )\n"
    )
    assert anchor_record in src, (
        "async_llm.py anchor missing: the inline `logger_ref[0].record(...)` "
        "block expected at the end of output_handler has drifted. Rebase of "
        "the vLLM fork requires re-reading scripts/vllm_patches/apply_patches.py "
        "to re-anchor this patch."
    )

    replacement_record = (
        "                    # 4) Logging — offloaded to background consumer.\n"
        "                    #    ScalarLM patch (Phase 6.5). vLLM's own TODO\n"
        "                    #    on this spot suggested a background thread;\n"
        "                    #    an asyncio task + bounded queue yields the\n"
        "                    #    same shape with less machinery.\n"
        "                    if logger_ref[0] is not None:\n"
        "                        try:\n"
        "                            metrics_queue_ref[0].put_nowait((\n"
        "                                outputs.engine_index,\n"
        "                                outputs.scheduler_stats,\n"
        "                                iteration_stats,\n"
        "                                renderer.stat_mm_cache(),\n"
        "                            ))\n"
        "                        except asyncio.QueueFull:\n"
        "                            # Metrics are statistical — drop the\n"
        "                            # oldest rather than block the output\n"
        "                            # loop. The engine must keep pulling.\n"
        "                            try:\n"
        "                                metrics_queue_ref[0].get_nowait()\n"
        "                                metrics_queue_ref[0].put_nowait((\n"
        "                                    outputs.engine_index,\n"
        "                                    outputs.scheduler_stats,\n"
        "                                    iteration_stats,\n"
        "                                    renderer.stat_mm_cache(),\n"
        "                                ))\n"
        "                            except (asyncio.QueueEmpty, asyncio.QueueFull):\n"
        "                                pass\n"
    )

    # --- Anchor 2: the ``async def output_handler()`` signature — we hang
    #     the metrics queue + consumer task off this spot, one level up
    #     from the handler body, so they share closure state. ---
    anchor_handler_def = "        async def output_handler():\n"
    assert src.count(anchor_handler_def) == 1, (
        "async_llm.py anchor missing: expected exactly one "
        "`async def output_handler():` inside `_run_output_handler`. "
        "Fork has drifted."
    )

    consumer_preamble = (
        "        # ScalarLM Phase 6.5: bounded queue + consumer task that\n"
        "        # drains the record() calls off the output_handler hot\n"
        "        # path. 1024 items is ~seconds of engine iterations at our\n"
        "        # scale; metric samples that overflow are dropped (see\n"
        "        # output_handler for the drop-oldest path).\n"
        "        metrics_queue_ref: list = [asyncio.Queue(maxsize=1024)]\n"
        "\n"
        "        async def _metrics_consumer():\n"
        "            q = metrics_queue_ref[0]\n"
        "            while True:\n"
        "                try:\n"
        "                    engine_idx, scheduler_stats, iteration_stats, mm_cache_stats = (\n"
        "                        await q.get()\n"
        "                    )\n"
        "                except asyncio.CancelledError:\n"
        "                    return\n"
        "                try:\n"
        "                    if logger_ref[0] is not None:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=engine_idx,\n"
        "                            scheduler_stats=scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=mm_cache_stats,\n"
        "                        )\n"
        "                except Exception:\n"
        "                    logger.exception(\"Background metrics record failed.\")\n"
        "\n"
        "        async def output_handler():\n"
    )

    # --- Anchor 3: the ``self.output_handler = asyncio.create_task(...)``
    #     line where the handler task is actually scheduled. We schedule
    #     the consumer next to it so the two have matching lifecycles. ---
    anchor_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
    )
    assert anchor_schedule in src, (
        "async_llm.py anchor missing: expected the "
        "`self.output_handler = asyncio.create_task(output_handler())` line."
    )
    replacement_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
        "        self._metrics_consumer_task = asyncio.create_task(_metrics_consumer())\n"
    )

    # Apply in order; each assertion above already guaranteed uniqueness.
    patched = (
        src
        .replace(anchor_record, replacement_record)
        .replace(anchor_handler_def, consumer_preamble)
        .replace(anchor_schedule, replacement_schedule)
    )

    # Sanity: the patch should change the file.
    assert patched != src, "patch produced identical output — something's wrong"
    # And it should still parse.
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied output_handler metrics offload to {target}")

TOKENFORMER_KEY_RESOLVER_SRC = '''

def _scalarlm_nearest_live_keys(key, model_state_dict, limit=3):
    """Live parameter names that sit near an unmatched checkpoint key.

    A drop warning that only names the missing key says nothing about WHY
    it missed. The live neighbours do: `...qkv_proj.base_layer.weight`
    next to a dropped `...q_proj.weight` shows both that vLLM fused the
    projection and that a LoRA wrapper nested it -- two separate naming
    problems, visible in one line, without reading fork source.

    Tries every contiguous run of path components, longest first, and
    returns the matches for the most specific one that hits anything.
    Trailing components have to be droppable too, not just leading ones:
    a fused `qkv_proj` means the name `q_proj` appears nowhere at all, so
    only backing off to the parent module (`layers.0.self_attn`) shows
    what the layer actually holds.
    """
    parts = key.split(".")
    candidates = {
        ".".join(parts[i:j])
        for i in range(len(parts))
        for j in range(i + 1, len(parts) + 1)
    }
    for needle in sorted(candidates, key=lambda c: (-len(c), c)):
        matches = [name for name in model_state_dict if needle in name]
        if matches:
            return sorted(matches)[:limit]
    return []


def _scalarlm_resolve_adapter_keys(model, tokenformers, model_state_dict):
    """Map trainer-side checkpoint keys onto the live vLLM namespace.

    ScalarLM's Tokenformer checkpoints are not pure adapters: the trainer
    unfreezes attention projections, norms and embeddings alongside the
    tokenformer_{k,v,p} tensors, so the .pt carries base weights named the
    way the HuggingFace module tree names them. `activate_adapter` merges
    every one of those keys into the model's state dict and calls
    `load_weights`, which requires the names to line up with vLLM's own
    parameter layout.

    For plain decoder-only models the two namespaces already agree, which
    is why this went unnoticed. Multimodal wrappers rearrange them (HF
    `model.language_model.layers.*` vs vLLM `language_model.model.layers.*`)
    and expose some modules not at all, so unmapped keys reach
    `load_weights` and raise KeyError, killing EngineCore.

    Resolution order:
      1. Key already present in the model's state dict: keep it unchanged.
         This is the backward-compatibility guarantee — every key of an
         existing checkpoint takes this path, so nothing is renamed or
         dropped and behavior is identical to before the patch.
      2. The model's own `hf_to_vllm_mapper` rewrites it to a key that is
         present: keep it under the rewritten name. This reuses the exact
         mapping vLLM applies when loading a checkpoint, so it stays
         correct per model family instead of hardcoding prefixes here.
      3. Otherwise: drop it, loudly. These are weights vLLM has no home
         for; injecting them is what crashes the engine. Dropping means
         they are not applied at inference — the warning exists so that
         shows up in the logs rather than as unexplained quality loss.
    """
    # Instance lookup, not `type(model)`: most models declare the mapper as a
    # class attribute, but several (mimo_mtp, ernie_mtp, the transformers
    # backend) build it in __init__, and a class-only lookup silently misses
    # those — every key would fall through to the drop path.
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    resolved = {}
    renamed = 0
    dropped = []

    for key, value in tokenformers.items():
        if key in model_state_dict:
            resolved[key] = value
            continue

        mapped = None
        if mapper is not None:
            try:
                mapped = mapper._map_name(key)
            except Exception:  # mapper API drift must not break serving
                mapped = None

        if mapped is not None and mapped in model_state_dict:
            resolved[mapped] = value
            renamed += 1
            continue

        dropped.append(key)

    if renamed:
        logger.info(
            "Tokenformer adapter: remapped %d checkpoint tensors onto the "
            "live model namespace via hf_to_vllm_mapper.",
            renamed,
        )
    if dropped:
        logger.warning(
            "Tokenformer adapter: %d of %d checkpoint tensors have no "
            "counterpart in the live model and will NOT be applied "
            "(first few: %s). These are base weights the trainer saved "
            "under names vLLM does not expose; the adapter still applies "
            "everything that did resolve.",
            len(dropped),
            len(tokenformers),
            dropped[:5],
        )
        # Name the live parameters nearest each drop. Without this the
        # warning says only that a name was missing, and diagnosing why
        # means reading fork source and guessing -- which is how the
        # `.base_layer.` nesting and the fused `qkv_proj` each cost a
        # 30-minute rebuild to discover.
        for key in dropped[:3]:
            logger.warning(
                "Tokenformer adapter:   %s -> no match; live names nearby: %s",
                key,
                _scalarlm_nearest_live_keys(key, model_state_dict) or "(none)",
            )

    return resolved
'''


def patch_tokenformer_adapter_key_resolution(vllm_root: Path) -> None:
    """Resolve adapter checkpoint keys against the live model before use.

    Without this, serving a ScalarLM Tokenformer checkpoint on a
    multimodal model (diffusion-gemma / gemma4) dies with
    `KeyError: 'layers.0.self_attn.q_proj.weight'` inside
    `gemma4.load_weights`, taking EngineCore down with it.

    Resolution happens once in `add_adapter`, so both `activate_adapter`
    and `deactivate_adapter` see keys that exist in the model — patching
    only the activate path would leave deactivate to raise on the same
    keys.

    See `_scalarlm_resolve_adapter_keys` for the rules and why existing
    checkpoints are unaffected.
    """
    target = (
        vllm_root / "vllm" / "tokenformer" / "tokenformer_model_manager.py"
    )
    if not target.exists():
        print(
            f"[vllm_patches] {target} not found; skipping tokenformer "
            f"key-resolution patch"
        )
        return

    src = target.read_text()

    if "_scalarlm_resolve_adapter_keys" in src:
        print(
            "[vllm_patches] tokenformer key resolution already present; "
            "skipping"
        )
        return

    anchor_logger = "logger = init_logger(__name__)\n"
    anchor_register = (
        "        self._registered_adapters[request.adapter_id] = tokenformer\n"
    )

    assert anchor_logger in src, (
        "tokenformer_model_manager.py: `logger = init_logger(__name__)` not "
        "found — cannot place the key resolver. Re-anchor this patch."
    )
    assert anchor_register in src, (
        "tokenformer_model_manager.py: the `self._registered_adapters"
        "[request.adapter_id] = tokenformer` assignment in add_adapter has "
        "drifted. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_logger,
        anchor_logger + TOKENFORMER_KEY_RESOLVER_SRC,
        1,
    ).replace(
        anchor_register,
        "        tokenformer.tokenformers = _scalarlm_resolve_adapter_keys(\n"
        "            self.model, tokenformer.tokenformers, self.model.state_dict()\n"
        "        )\n" + anchor_register,
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied tokenformer key resolution to {target}")


STATE_DICT_EXPORT_SRC = '''    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # ScalarLM trainer contract: expose packed projections under their
        # unpacked HF names. qwen2, qwen3, qwen3_moe and gemma3 implement
        # this; Gemma4 was missed, so `qkv_proj` stayed fused and the
        # Tokenformer manager found no q/k/v to match a ScalarLM
        # checkpoint against. Every trained attention projection was then
        # dropped at adapter-activation time and the served model ran
        # random-init attention -- training converges, the checkpoint is
        # correct, and generation is still garbage.
        #
        # Unlike gemma3, qkv cannot be delegated to
        # `unpack_packed_modules_state_dict`: it derives one split size
        # from the model config, while Gemma4 alternates head_dim per
        # layer (32 on some, 64 on others). That mismatch raises
        # "split_with_sizes expects split_sizes to sum exactly to 1024".
        # Each attention module knows its own TP-local q_size/kv_size,
        # which is what the packed tensor here actually holds, so read
        # the geometry off the module instead of the config.
        #
        # Scoped to `prefix` so recursion from the multimodal wrapper
        # (Gemma4ForConditionalGeneration) can't rewrite sibling modules'
        # keys in the shared destination dict.
        state_dict = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if not scalarlm_state_dict_export_enabled():
            return state_dict

        layers = getattr(getattr(self, "model", None), "layers", None) or []
        for index, layer in enumerate(layers):
            # PPMissingLayer placeholders have no attention module.
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "q_size"):
                continue
            q_size, kv_size = attn.q_size, attn.kv_size
            base = f"{prefix}model.layers.{index}.self_attn"
            for suffix in ("weight", "bias"):
                packed_key = f"{base}.qkv_proj.{suffix}"
                packed = state_dict.get(packed_key)
                if packed is None:
                    continue
                if packed.shape[0] != q_size + 2 * kv_size:
                    # Geometry we don't model. Leaving it packed loses the
                    # tensor for adapter matching, which is recoverable;
                    # splitting it wrongly corrupts weights silently.
                    continue
                del state_dict[packed_key]
                state_dict[f"{base}.q_proj.{suffix}"] = packed[:q_size]
                state_dict[f"{base}.k_proj.{suffix}"] = packed[
                    q_size : q_size + kv_size
                ]
                state_dict[f"{base}.v_proj.{suffix}"] = packed[q_size + kv_size :]

        # qkv is handled above; the shared helper still unpacks
        # gate_up_proj and drops fused-expert / scale keys that have no HF
        # counterpart.
        return unpack_packed_modules_state_dict(
            state_dict,
            prefix=prefix,
            packed_modules_mapping={
                key: value
                for key, value in self.packed_modules_mapping.items()
                if key != "qkv_proj"
            },
            config=self.config,
        )

'''


LATEST_CHECKPOINT_SRC = '''

def _scalarlm_latest_checkpoint(files):
    """Pick the highest-step checkpoint, not the alphabetically first.

    Both call sites used `sorted(...)[0]`, which orders lexically:
    checkpoint_1000.pt sorts before checkpoint_1999.pt, and
    checkpoint_500.pt sorts after both. A job that saved more than one
    checkpoint therefore serves an arbitrary earlier one -- an
    undertrained model that loads cleanly, reports every tensor matched,
    and answers with garbage. Observed on a 2000-step run whose final
    checkpoint was correct under HuggingFace while vLLM served an
    earlier one.

    `max` over the already-sorted list keeps ties deterministic, so the
    loader and the adapter_format classifier still agree on the same
    file -- which the comments at both sites require.
    """

    def step_of(path):
        stem = path.name.rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(sorted(files), key=step_of)
'''


def patch_diffusion_gemma_sc_embeds_dtype(vllm_root: Path) -> None:
    """Cast the self-conditioning soft embeds to the buffer's dtype.

    `sc_embeds` is float32; `soft_embeds` inherits embed_weight's dtype
    (bf16), and index_put_ refuses to cast, so DiffusionGemma dies during
    warmup on B200:

        RuntimeError: Index put requires the source and destination dtypes
        match, got Float for the destination and BFloat16 for the source.

    The consumer already does `soft.to(inputs_embeds.dtype)` on read, so
    casting on write matches the existing convention. bf16 -> fp32 is
    lossless.

    Belongs upstream in vllm-fork; carried here so DiffusionGemma can be
    exercised without waiting on that.
    """
    target = (
        vllm_root / "vllm" / "model_executor" / "models" / "diffusion_gemma.py"
    )
    if not target.exists():
        print(
            f"[vllm_patches] {target} not found; skipping diffusion_gemma "
            f"dtype patch"
        )
        return

    src = target.read_text()
    anchor = "    sc_embeds[decode_slots] = soft_embeds * sc_keep\n"

    if "(soft_embeds * sc_keep).to(sc_embeds.dtype)" in src:
        print("[vllm_patches] diffusion_gemma dtype cast already present; skipping")
        return

    assert src.count(anchor) == 1, (
        "diffusion_gemma.py: expected exactly one "
        "`sc_embeds[decode_slots] = soft_embeds * sc_keep`. Re-anchor this patch."
    )

    patched = src.replace(
        anchor,
        "    # ScalarLM patch: sc_embeds is fp32, soft_embeds is bf16, and\n"
        "    # index_put_ will not cast. The consumer casts on read too.\n"
        "    sc_embeds[decode_slots] = (soft_embeds * sc_keep).to(sc_embeds.dtype)\n",
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied diffusion_gemma sc_embeds dtype cast to {target}")


def patch_latest_checkpoint_selection(vllm_root: Path) -> None:
    """Select adapter checkpoints by step number rather than filename.

    Applies to both places that resolve a job directory to a single
    `.pt`: the Tokenformer loader and the adapter_format classifier.
    They are commented as needing to agree, so they are patched together
    or not at all.
    """
    manager = vllm_root / "vllm" / "tokenformer" / "tokenformer_model_manager.py"
    fmt = vllm_root / "vllm" / "tokenformer" / "adapter_format.py"

    for target in (manager, fmt):
        if not target.exists():
            print(
                f"[vllm_patches] {target} not found; skipping "
                f"latest-checkpoint patch"
            )
            return

    manager_src = manager.read_text()
    fmt_src = fmt.read_text()

    if "_scalarlm_latest_checkpoint" in manager_src:
        print("[vllm_patches] latest-checkpoint selection already present; skipping")
        return

    manager_anchor = (
        "        files = sorted(Path(model_dir).glob(\"*.pt\"))\n"
        "\n"
        "        if len(files) == 0:\n"
        "            raise FileNotFoundError(f\"No .pt file found in {model_dir}\")\n"
        "\n"
        "        checkpoint_file = files[0]\n"
    )
    fmt_anchor = (
        "    files = sorted(model_dir.glob(\"*.pt\"))\n"
        "    if not files:\n"
        "        raise FileNotFoundError(f\"No .pt file found in {model_dir}\")\n"
        "    checkpoint_file = files[0]\n"
    )
    manager_logger = "logger = init_logger(__name__)\n"
    fmt_imports = "from typing import Any, Literal, TypeAlias\n"

    assert manager_anchor in manager_src, (
        "tokenformer_model_manager.py: the checkpoint-selection block in "
        "from_local_checkpoint has drifted. Re-anchor this patch."
    )
    assert fmt_anchor in fmt_src, (
        "adapter_format.py: the checkpoint-selection block in "
        "_load_adapter_checkpoint has drifted. Re-anchor this patch."
    )
    assert manager_logger in manager_src and fmt_imports in fmt_src, (
        "tokenformer files: module-level insertion anchors have drifted. "
        "Re-anchor this patch."
    )

    manager_patched = manager_src.replace(
        manager_logger, manager_logger + LATEST_CHECKPOINT_SRC, 1
    ).replace(
        manager_anchor,
        manager_anchor.replace(
            "        checkpoint_file = files[0]\n",
            "        checkpoint_file = _scalarlm_latest_checkpoint(files)\n",
        ),
        1,
    )
    fmt_patched = fmt_src.replace(
        fmt_imports, fmt_imports + LATEST_CHECKPOINT_SRC, 1
    ).replace(
        fmt_anchor,
        fmt_anchor.replace(
            "    checkpoint_file = files[0]\n",
            "    checkpoint_file = _scalarlm_latest_checkpoint(files)\n",
        ),
        1,
    )

    for target, patched, original in (
        (manager, manager_patched, manager_src),
        (fmt, fmt_patched, fmt_src),
    ):
        assert patched != original, f"{target}: patch produced identical output"
        compile(patched, str(target), "exec")
        target.write_text(patched)

    print("[vllm_patches] Applied latest-checkpoint selection to both call sites")


def patch_llama_scalarlm_state_dict_export(vllm_root: Path) -> None:
    """Give LlamaForCausalLM the ScalarLM `state_dict` export override.

    Same defect as gemma4 had: llama fuses q/k/v into `qkv_proj` and never
    exposes the unpacked names, so every trained attention projection is
    dropped when a Tokenformer adapter is applied. Simpler fix than
    gemma4's because llama's head_dim does not vary per layer.
    """
    target = vllm_root / "vllm" / "model_executor" / "models" / "llama.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping llama state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] llama state_dict export already present; skipping")
        return

    anchor_imports = (
        "from .utils import (\n"
        "    AutoWeightsLoader,\n"
        "    PPMissingLayer,\n"
        "    WeightsMapper,\n"
        "    extract_layer_index,\n"
        "    make_empty_intermediate_tensors_factory,\n"
        "    make_layers,\n"
        "    maybe_prefix,\n"
        ")\n"
    )
    anchor_class = (
        "class LlamaForCausalLM(\n"
        "    LocalArgmaxMixin,\n"
        "    nn.Module,\n"
        "    SupportsLoRA,\n"
        "    SupportsPP,\n"
        "    SupportsEagle,\n"
        "    SupportsEagle3,\n"
        "    SupportsQuant,\n"
        "    SupportsTokenformer,\n"
        "):\n"
    )

    assert anchor_imports in src, (
        "llama.py: the `from .utils import (...)` block has drifted; cannot "
        "add the state_dict export helpers. Re-anchor this patch."
    )
    assert anchor_class in src, (
        "llama.py: the LlamaForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_imports,
        anchor_imports.replace(
            "    maybe_prefix,\n",
            "    maybe_prefix,\n"
            "    scalarlm_state_dict_export_enabled,\n"
            "    unpack_packed_modules_state_dict,\n",
        ),
        1,
    ).replace(anchor_class, anchor_class + STATE_DICT_EXPORT_SRC, 1)

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied llama state_dict export to {target}")


def patch_gemma4_scalarlm_state_dict_export(vllm_root: Path) -> None:
    """Give Gemma4ForCausalLM the ScalarLM `state_dict` export override.

    Five model files implement it (qwen2, qwen3, qwen3_moe, gemma3, plus
    the helper in utils.py); gemma4 does not. The consequence is specific
    and quiet: serving a ScalarLM Tokenformer checkpoint on Gemma4 drops
    every `q_proj`/`k_proj`/`v_proj` tensor the trainer produced, because
    vLLM only exposes them fused as `qkv_proj`. Training converges, the
    checkpoint is correct, and the served model still answers with
    garbage because its attention is at initialization.

    Verified with tiny-random/gemma-4-dense: 10 of 52 checkpoint tensors
    unmatched without this, all of them attention projections.
    """
    target = vllm_root / "vllm" / "model_executor" / "models" / "gemma4.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping gemma4 state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] gemma4 state_dict export already present; skipping")
        return

    anchor_imports = (
        "from .utils import (\n"
        "    AutoWeightsLoader,\n"
        "    WeightsMapper,\n"
        "    extract_layer_index,\n"
        "    is_pp_missing_parameter,\n"
        "    make_layers,\n"
        "    maybe_prefix,\n"
        ")\n"
    )
    anchor_class = (
        "class Gemma4ForCausalLM(\n"
        "    nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts, SupportsEagle3\n"
        "):\n"
    )

    assert anchor_imports in src, (
        "gemma4.py: the `from .utils import (...)` block has drifted; cannot "
        "add the state_dict export helpers. Re-anchor this patch."
    )
    assert anchor_class in src, (
        "gemma4.py: the Gemma4ForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_imports,
        anchor_imports.replace(
            "    maybe_prefix,\n",
            "    maybe_prefix,\n"
            "    scalarlm_state_dict_export_enabled,\n"
            "    unpack_packed_modules_state_dict,\n",
        ),
        1,
    ).replace(
        anchor_class,
        anchor_class + STATE_DICT_EXPORT_SRC,
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied gemma4 state_dict export to {target}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-root>", file=sys.stderr)
        return 2
    vllm_root = Path(sys.argv[1]).resolve()
    if not (vllm_root / "vllm" / "v1" / "engine" / "async_llm.py").exists():
        print(f"[vllm_patches] async_llm.py not found under {vllm_root}", file=sys.stderr)
        return 3

    patch_output_handler_metrics_offload(vllm_root)
    patch_tokenformer_adapter_key_resolution(vllm_root)
    patch_gemma4_scalarlm_state_dict_export(vllm_root)
    patch_llama_scalarlm_state_dict_export(vllm_root)
    patch_latest_checkpoint_selection(vllm_root)
    patch_diffusion_gemma_sc_embeds_dtype(vllm_root)
    print("[vllm_patches] All patches applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
