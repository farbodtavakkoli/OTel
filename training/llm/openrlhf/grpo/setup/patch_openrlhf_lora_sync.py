import argparse
import importlib.util
import os
import sys

MARK = "[grpo-bench lora-sync patch]"

# openrlhf/models/actor.py passes --ds.lora.target_modules (a list) straight to LoraConfig, and PEFT
# matches list entries by suffix: a bare `q_proj` also matches a multimodal vision tower's
# `...self_attn.q_proj` wrapper (not an nn.Linear) and PEFT raises. A single `re:<regex>` entry is
# handed to PEFT as a regex string instead (PEFT uses re.fullmatch for string targets).
ACTOR_OLD = """                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,"""
ACTOR_NEW = """                # %s single 're:<regex>' entry -> regex string for PEFT (fullmatch)
                if (
                    isinstance(target_modules, (list, tuple))
                    and len(target_modules) == 1
                    and isinstance(target_modules[0], str)
                    and target_modules[0].startswith("re:")
                ):
                    target_modules = target_modules[0][3:]
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,""" % MARK

# Stock broadcast_to_vllm iterates model.named_parameters() and broadcasts each tensor under its
# parameter name. With --ds.lora.rank > 0 the actor is a PeftModel, so the names are
# `base_model.model.<...>.q_proj.base_layer.weight` / `...lora_A.default.weight`, which vLLM's
# load_weights cannot map onto the base model; for a multimodal checkpoint it also restricts the sync
# to requires_grad params, i.e. the LoRA matrices alone. Either way the rollout engine never gets the
# updated policy and the run silently trains nothing. The replacement broadcasts the merged
# `base + B@A*scaling` weight under HF names, built in a temporary tensor so the frozen base weight is
# never mutated and no bf16 rounding accumulates across steps. Frozen modules are bit-identical on
# both sides and skipped. This is OpenRLHF's own full-weight broadcast applied to LoRA, not an
# adapter hot-load, so the sync cost is the full merged projections every step.
PPO_OLD = """        sync_fn = _handle_cuda_ipc if self.use_cuda_ipc else _broadcast_param

        # VLM: only sync trainable (language model) params — vision encoder is frozen.
        params_to_sync = ["""
PPO_NEW = """        sync_fn = _handle_cuda_ipc if self.use_cuda_ipc else _broadcast_param

        # %s LoRA actor: broadcast merged (base + delta) weights of LoRA-wrapped
        # modules under their Hugging Face names; frozen modules are unchanged in vLLM.
        try:
            from peft import PeftModel
            from peft.tuners.lora import LoraLayer
        except ImportError:  # pragma: no cover
            PeftModel = LoraLayer = None
        if PeftModel is not None and isinstance(model, PeftModel):
            assert self.strategy.args.ds.zero_stage != 3, "LoRA merged-weight sync supports ZeRO-0/1/2 only"
            lora_modules = []
            for mod_name, module in model.named_modules():
                if isinstance(module, LoraLayer) and getattr(module, "active_adapters", None):
                    hf_name = mod_name[len("base_model.model."):] if mod_name.startswith("base_model.model.") else mod_name
                    lora_modules.append((hf_name + ".weight", module))
            num_params = len(lora_modules)
            if torch.distributed.get_rank() == 0:
                print(f"{MARK_STR} syncing {num_params} merged LoRA module weights to vLLM")
            for name, module in lora_modules:
                count += 1
                merged = module.base_layer.weight.data.clone()
                for adapter in module.active_adapters:
                    if adapter in module.lora_A:
                        merged += module.get_delta_weight(adapter).to(merged.dtype)
                sync_fn(merged, count, num_params)
                del merged
            if cache_reset_refs:
                ray.get(cache_reset_refs)
            torch.cuda.empty_cache()
            torch_dist_barrier_and_cuda_sync()
            return

        # VLM: only sync trainable (language model) params — vision encoder is frozen.
        params_to_sync = [""" % MARK
PPO_NEW = PPO_NEW.replace("{MARK_STR}", MARK)


# A vLLM built from main without a release tag reports `0.1.devN+g<sha>`. OpenRLHF's version guards
# (vllm_engine.py asserts > 0.8.5; ppo_actor.py sets NCCL_CUMEM_ENABLE=0 below 0.16) read that as an
# ancient vLLM and abort the rollout actor, so such a build is compared as 999.
VLLM_ENGINE_OLD = """        assert version.parse(vllm.__version__) > version.parse(
            "0.8.5"
        ), "Streaming VLLM version must be greater than 0.8.5"

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):"""
VLLM_ENGINE_NEW = """        # %s untagged main build ('0.1.devN+g<sha>') counts as newest
        _vllm_ver = "999" if str(vllm.__version__).startswith("0.1.dev") else vllm.__version__
        assert version.parse(_vllm_ver) > version.parse(
            "0.8.5"
        ), "Streaming VLLM version must be greater than 0.8.5"

        if version.parse(_vllm_ver) >= version.parse("0.9.0"):""" % MARK

PPO_VER_OLD = """            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):"""
PPO_VER_NEW = """            _vllm_ver = "999" if str(vllm.__version__).startswith("0.1.dev") else vllm.__version__
            if pkg_version.parse(_vllm_ver) < pkg_version.parse("0.16"):"""


def locate():
    spec = importlib.util.find_spec("openrlhf")
    root = os.path.dirname(spec.origin)
    return (
        os.path.join(root, "models", "actor.py"),
        os.path.join(root, "trainer", "ray", "ppo_actor.py"),
        os.path.join(root, "trainer", "ray", "vllm_engine.py"),
    )


def patch_file(path, old, new, check):
    src = open(path, encoding="utf-8").read()
    if new in src:
        print(f"already patched: {path} [{old.strip().splitlines()[0][:50]}...]")
        return True
    if old not in src:
        print(f"ANCHOR NOT FOUND (unexpected OpenRLHF version?): {path}\n  {old.strip().splitlines()[0]}")
        return False
    if check:
        print(f"would patch: {path}")
        return True
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print(f"patched: {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    actor_py, ppo_actor_py, vllm_engine_py = locate()
    ok = patch_file(actor_py, ACTOR_OLD, ACTOR_NEW, args.check)
    ok = patch_file(ppo_actor_py, PPO_OLD, PPO_NEW, args.check) and ok
    ok = patch_file(ppo_actor_py, PPO_VER_OLD, PPO_VER_NEW, args.check) and ok
    ok = patch_file(vllm_engine_py, VLLM_ENGINE_OLD, VLLM_ENGINE_NEW, args.check) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
