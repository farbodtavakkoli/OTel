#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def patch_output_handler_metrics_offload(vllm_root: Path) -> None:
    target = vllm_root / "vllm" / "v1" / "engine" / "async_llm.py"
    src = target.read_text()

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
        "                    if logger_ref[0] is not None:\n"
        "                        try:\n"
        "                            metrics_queue_ref[0].put_nowait((\n"
        "                                outputs.engine_index,\n"
        "                                outputs.scheduler_stats,\n"
        "                                iteration_stats,\n"
        "                                renderer.stat_mm_cache(),\n"
        "                            ))\n"
        "                        except asyncio.QueueFull:\n"
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

    anchor_handler_def = "        async def output_handler():\n"
    assert src.count(anchor_handler_def) == 1, (
        "async_llm.py anchor missing: expected exactly one "
        "`async def output_handler():` inside `_run_output_handler`. "
        "Fork has drifted."
    )

    consumer_preamble = (
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

    patched = (
        src
        .replace(anchor_record, replacement_record)
        .replace(anchor_handler_def, consumer_preamble)
        .replace(anchor_schedule, replacement_schedule)
    )

    assert patched != src, "patch produced identical output — something's wrong"

    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied output_handler metrics offload to {target}")

TOKENFORMER_KEY_RESOLVER_SRC = '''

def _scalarlm_nearest_live_keys(key, model_state_dict, limit=3):
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
            except Exception:
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
        for key in dropped[:3]:
            logger.warning(
                "Tokenformer adapter:   %s -> no match; live names nearby: %s",
                key,
                _scalarlm_nearest_live_keys(key, model_state_dict) or "(none)",
            )

    return resolved
'''


def patch_tokenformer_adapter_key_resolution(vllm_root: Path) -> None:
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
        state_dict = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if not scalarlm_state_dict_export_enabled():
            return state_dict

        layers = getattr(getattr(self, "model", None), "layers", None) or []
        for index, layer in enumerate(layers):
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
                    continue
                del state_dict[packed_key]
                state_dict[f"{base}.q_proj.{suffix}"] = packed[:q_size]
                state_dict[f"{base}.k_proj.{suffix}"] = packed[
                    q_size : q_size + kv_size
                ]
                state_dict[f"{base}.v_proj.{suffix}"] = packed[q_size + kv_size :]

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


def _add_state_dict_export_imports(src: str, filename: str) -> str:
    import re

    blocks = re.findall(r"^from \.utils import \(\n(?:    [A-Za-z_]\w*,\n)+\)\n", src, re.M)
    assert len(blocks) == 1, (
        f"{filename}: expected exactly one `from .utils import (...)` block, "
        f"found {len(blocks)}. Re-anchor this patch."
    )
    block = blocks[0]
    assert "    maybe_prefix,\n" in block, (
        f"{filename}: `from .utils import (...)` no longer imports maybe_prefix; "
        "cannot add the state_dict export helpers. Re-anchor this patch."
    )
    new_block = block.replace(
        "    maybe_prefix,\n",
        "    maybe_prefix,\n"
        "    scalarlm_state_dict_export_enabled,\n"
        "    unpack_packed_modules_state_dict,\n",
        1,
    )
    return src.replace(block, new_block, 1)


LATEST_CHECKPOINT_SRC = '''

def _scalarlm_latest_checkpoint(files):

    def step_of(path):
        stem = path.name.rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(sorted(files), key=step_of)
'''


def patch_diffusion_gemma_sc_embeds_dtype(vllm_root: Path) -> None:
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

    import re

    if re.search(
        r"^\s*sc_embeds\[decode_slots\] = \(soft_embeds \* sc_keep[^\n]*\)"
        r"\.to\(sc_embeds\.dtype\)\n",
        src,
        re.M,
    ):
        print("[vllm_patches] diffusion_gemma dtype cast already present; skipping")
        return

    assert src.count(anchor) == 1, (
        "diffusion_gemma.py: expected exactly one "
        "`sc_embeds[decode_slots] = soft_embeds * sc_keep`. Re-anchor this patch."
    )

    patched = src.replace(
        anchor,
        "    sc_embeds[decode_slots] = (soft_embeds * sc_keep).to(sc_embeds.dtype)\n",
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied diffusion_gemma sc_embeds dtype cast to {target}")


def patch_latest_checkpoint_selection(vllm_root: Path) -> None:
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
    target = vllm_root / "vllm" / "model_executor" / "models" / "llama.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping llama state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] llama state_dict export already present; skipping")
        return

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

    assert anchor_class in src, (
        "llama.py: the LlamaForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = _add_state_dict_export_imports(src, "llama.py").replace(
        anchor_class, anchor_class + STATE_DICT_EXPORT_SRC, 1
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied llama state_dict export to {target}")


def patch_gemma4_scalarlm_state_dict_export(vllm_root: Path) -> None:
    target = vllm_root / "vllm" / "model_executor" / "models" / "gemma4.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping gemma4 state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] gemma4 state_dict export already present; skipping")
        return

    anchor_class = (
        "class Gemma4ForCausalLM(\n"
        "    nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts, SupportsEagle3\n"
        "):\n"
    )

    assert anchor_class in src, (
        "gemma4.py: the Gemma4ForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = _add_state_dict_export_imports(src, "gemma4.py").replace(
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
