"""
Unit tests for adapters.resolve_target_modules.resolve_target_modules.

PEFT's "all-linear" shorthand fails to expand for some architectures
(notably MoE: Qwen3MoeForCausalLM under peft 0.19 + transformers 5.x). PEFT
falls back to iterating the literal string as a set of characters and raises
`Target modules {'-','l','n','r','i','a','e'} not found in the base model`,
which is the qwen3-moe TRAIN_FAILED in the cuda-spark sweep. We resolve the
shorthand ourselves from the live model, so these tests use small synthetic
nn.Modules rather than HF downloads.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from adapters.resolve_target_modules import (
    resolve_target_modules,
    resolve_target_parameters,
)


class _DenseLike(nn.Module):
    """A miniature ...ForCausalLM: attention + MLP projections + an output head,
    with several layers so leaf names repeat (as in a real stacked model)."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )
            for _ in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


class _NoOutputEmbeddings(nn.Module):
    """A model whose get_output_embeddings() returns None (some configs do);
    the output head must still be excluded by its conventional leaf name."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return None


class _GroupedExperts(nn.Module):
    """Qwen3MoE-style *grouped* experts: batched weights held as `nn.Parameter`s
    (no per-expert `nn.Linear`), matching transformers-5.x `Qwen3MoeExperts`. PEFT
    adapts these on its own via `ParamWrapper` and the grouped serve converter
    handles the fused export, so the resolver — which targets `nn.Linear` — never
    lists them. (The real `qwen3-moe-tiny-random` loads as this grouped module,
    per the PEFT `ParamWrapper(Qwen3MoeExperts)` tree in the 2026-06-30 report;
    the earlier per-expert-`nn.Linear` stub was inaccurate for Qwen3MoE.)"""

    def __init__(self, n_experts, dim=8, inter=8):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(n_experts, dim, 2 * inter))
        self.down_proj = nn.Parameter(torch.zeros(n_experts, inter, dim))


class _MoeLike(nn.Module):
    """A miniature Qwen3MoE-style ...ForCausalLM mirroring `decoder_sparse_step`:
    some decoder layers carry a *dense* MLP (`gate_proj`/`up_proj`/`down_proj`)
    and others a *sparse* MLP with a router (`gate`) + GROUPED routed experts
    (`_GroupedExperts`, nn.Parameters — not per-expert Linears). By default layer 0
    is dense and layer 1 is sparse — the real `qwen3-moe-tiny-random` layout
    (decoder_sparse_step=2). The grouped experts are adapted by PEFT itself and the
    router can't be served from a .pt adapter, so resolution adapts attention + the
    dense MLP only — and by *full path*, since the dense MLP shares leaf names with
    (other) MoE submodules. See `_SeparateMoeLike` for the Mixtral/PhiMoE layout."""

    def __init__(self, sparse_layers=(1,), n_layers=2, n_experts=4, router_name="gate"):
        super().__init__()

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _dense_mlp():
            return nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _sparse_mlp():
            return nn.ModuleDict(
                {
                    # The MoE router. Its leaf name varies by arch (`gate` in
                    # Qwen3MoE, `router` in PhiMoE); in PhiMoE it is an nn.Linear
                    # *subclass* whose forward returns a 3-tuple, so LoRA-wrapping
                    # it crashes PEFT (`'tuple' object has no attribute 'dtype'`).
                    # Resolution must exclude it by name regardless.
                    router_name: nn.Linear(8, n_experts, bias=False),
                    "experts": _GroupedExperts(n_experts),
                }
            )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "self_attn": _attn(),
                    "mlp": _sparse_mlp() if i in sparse_layers else _dense_mlp(),
                }
            )
            for i in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_moe_adapts_attention_and_dense_mlp_excluding_experts_and_router():
    # Layer 0 dense, layer 1 sparse (the real qwen3-moe-tiny-random layout).
    model = _MoeLike(sparse_layers=(1,), n_layers=2)
    result = resolve_target_modules(model, "all-linear")
    # Full paths: attention on every layer + the DENSE MLP (layer 0 only),
    # but no experts, no router, no head.
    assert result == [
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert not any(".experts." in name for name in result)  # no routed experts
    assert not any(name.endswith(".gate") for name in result)  # no router
    assert not any("lm_head" in name for name in result)  # no output head


def test_moe_excludes_router_named_router_phimoe():
    # The router-leaf rule: a router named `router` (not `gate`) must be excluded
    # too. PhiMoE names its router `mlp.router` and it is an nn.Linear subclass
    # returning a tuple; LoRA-wrapping it crashes training. Exercised here on the
    # grouped fixture (the rule is layout-independent); the full separate-expert
    # PhiMoE layout is covered in test_separate_expert_moe_* below.
    model = _MoeLike(sparse_layers=(1,), n_layers=2, router_name="router")
    result = resolve_target_modules(model, "all-linear")
    assert not any(name.endswith(".router") for name in result)  # router excluded
    assert "layers.1.mlp.router" not in result
    # Attention on every layer plus the layer-0 dense MLP still resolve.
    assert result == [
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]


def test_moe_all_layers_sparse_adapts_attention_only():
    # If every layer is sparse (no dense MLP anywhere), only attention survives.
    model = _MoeLike(sparse_layers=(0, 1), n_layers=2)
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert not any("mlp" in name for name in result)


def test_grouped_experts_yield_expert_target_parameters():
    # A grouped experts module (named `experts`, as it always is in a real decoder)
    # exposes its batched projections as the target_parameters PEFT needs —
    # target_modules can't reach bare nn.Parameters.
    holder = nn.ModuleDict({"experts": _GroupedExperts(n_experts=4)})
    assert resolve_target_parameters(holder) == ["down_proj", "gate_up_proj"]


def test_grouped_moe_target_parameters_regardless_of_dense_layer():
    # Grouped MoE with a dense layer (Qwen3MoE-like) AND all-sparse (OLMoE-like):
    # both must surface the grouped expert params. This is the fix — OLMoE/PhiMoE
    # have no dense layer to seed PEFT's own conversion, so without naming these
    # params their experts got no LoRA.
    assert resolve_target_parameters(_MoeLike(sparse_layers=(1,), n_layers=2)) == [
        "down_proj",
        "gate_up_proj",
    ]
    assert resolve_target_parameters(_MoeLike(sparse_layers=(0, 1), n_layers=2)) == [
        "down_proj",
        "gate_up_proj",
    ]


def test_dense_model_has_no_expert_target_parameters():
    assert resolve_target_parameters(_DenseLike()) == []


class _SeparateMoeLike(nn.Module):
    """A miniature Mixtral/PhiMoE-style ...ForCausalLM whose routed experts are a
    `ModuleList` of per-expert `nn.Linear` projections (`mlp.experts.{i}.{w1,w2,w3}`)
    rather than a grouped param module. Unlike grouped Qwen3MoE, PEFT does NOT
    auto-adapt these per-expert Linears, so resolution must INCLUDE them (by full
    path) to train the experts — while still excluding the router and output head.
    The separate-expert serve converter then stacks them for FusedMoEWithLoRA."""

    def __init__(self, sparse_layers=(0,), n_layers=1, n_experts=2, router_name="router"):
        super().__init__()

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _dense_mlp():
            return nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _expert():  # Mixtral/PhiMoE per-expert SwiGLU: w1=gate, w3=up, w2=down.
            return nn.ModuleDict(
                {
                    "w1": nn.Linear(8, 8, bias=False),
                    "w2": nn.Linear(8, 8, bias=False),
                    "w3": nn.Linear(8, 8, bias=False),
                }
            )

        def _sparse_mlp():
            return nn.ModuleDict(
                {
                    router_name: nn.Linear(8, n_experts, bias=False),
                    "experts": nn.ModuleList(_expert() for _ in range(n_experts)),
                }
            )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "self_attn": _attn(),
                    "mlp": _sparse_mlp() if i in sparse_layers else _dense_mlp(),
                }
            )
            for i in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_separate_expert_moe_includes_per_expert_linears():
    # PhiMoE/Mixtral: per-expert nn.Linear experts (w1/w2/w3) ARE adapted — they
    # carry memorization and the separate-expert converter can serve them, unlike
    # grouped Qwen3MoE experts. Router (leaf `router`) and head stay excluded.
    model = _SeparateMoeLike(sparse_layers=(0,), n_layers=1, n_experts=2)
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.mlp.experts.0.w1",
        "layers.0.mlp.experts.0.w2",
        "layers.0.mlp.experts.0.w3",
        "layers.0.mlp.experts.1.w1",
        "layers.0.mlp.experts.1.w2",
        "layers.0.mlp.experts.1.w3",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
    ]
    assert any(".experts.0.w1" in name for name in result)  # experts INCLUDED
    assert not any(name.endswith(".router") for name in result)  # router excluded
    assert not any("lm_head" in name for name in result)  # no output head


def test_separate_expert_moe_includes_experts_and_dense_mlp():
    # A mixed model (layer 0 dense, layer 1 sparse): the dense MLP AND the sparse
    # layer's per-expert Linears are both adapted, plus attention on every layer.
    model = _SeparateMoeLike(sparse_layers=(1,), n_layers=2, n_experts=2)
    result = resolve_target_modules(model, "all-linear")
    assert "layers.0.mlp.gate_proj" in result  # dense MLP on the dense layer
    assert "layers.1.mlp.experts.1.w2" in result  # per-expert on the sparse layer
    assert "layers.0.self_attn.q_proj" in result
    assert not any(name.endswith(".router") for name in result)
    assert not any("lm_head" in name for name in result)


class _MixtralLike(nn.Module):
    """Mixtral names its MoE block `block_sparse_moe` with a `gate` router and
    per-expert `experts.{i}.{w1,w2,w3}` Linears. Resolution must INCLUDE the
    per-expert Linears (separate layout) while still excluding the `gate` router —
    verifying the separate-expert branch fires before the `.block_sparse_moe`
    grouped-exclusion (which is only meant for Granite's fused experts)."""

    def __init__(self, n_experts=2):
        super().__init__()

        def _expert():
            return nn.ModuleDict(
                {
                    "w1": nn.Linear(8, 8, bias=False),
                    "w2": nn.Linear(8, 8, bias=False),
                    "w3": nn.Linear(8, 8, bias=False),
                }
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "q_proj": nn.Linear(8, 8, bias=False),
                                "k_proj": nn.Linear(8, 8, bias=False),
                                "v_proj": nn.Linear(8, 8, bias=False),
                                "o_proj": nn.Linear(8, 8, bias=False),
                            }
                        ),
                        "block_sparse_moe": nn.ModuleDict(
                            {
                                "gate": nn.Linear(8, n_experts, bias=False),  # router
                                "experts": nn.ModuleList(
                                    _expert() for _ in range(n_experts)
                                ),
                            }
                        ),
                    }
                )
            ]
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_mixtral_block_sparse_moe_separate_experts_included_router_excluded():
    model = _MixtralLike(n_experts=2)
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.block_sparse_moe.experts.0.w1",
        "layers.0.block_sparse_moe.experts.0.w2",
        "layers.0.block_sparse_moe.experts.0.w3",
        "layers.0.block_sparse_moe.experts.1.w1",
        "layers.0.block_sparse_moe.experts.1.w2",
        "layers.0.block_sparse_moe.experts.1.w3",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
    ]
    # The `gate` router under block_sparse_moe is excluded despite the experts
    # under the same subtree being included.
    assert not any(name.endswith(".gate") for name in result)


class _GraniteHybridLike(nn.Module):
    """A miniature GraniteMoeHybrid ...ForCausalLM mirroring the real leaf naming
    (verified by a meta-device HF init of ibm-granite/granite-4.0-h-tiny):

    - attention (attention layers only): self_attn.{q,k,v,o}_proj
    - dense shared MLP (serveable, every layer): shared_mlp.{input_linear, output_linear}
    - grouped routed experts (NOT .pt-serveable): block_sparse_moe.{input_linear, output_linear}
    - router (leaf name `layer`, not gate/router): block_sparse_moe.router.layer
    - Mamba-2 SSM (not adapted): mamba.{in_proj, out_proj}

    There is NO `.experts` submodule — the resolver must detect MoE via
    `block_sparse_moe`. Layer 0 is a Mamba layer (mamba + moe + shared_mlp); layer 1
    is an attention layer (self_attn + moe + shared_mlp)."""

    def __init__(self, n_experts=4):
        super().__init__()

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _shared_mlp():
            return nn.ModuleDict(
                {
                    "input_linear": nn.Linear(8, 16, bias=False),
                    "output_linear": nn.Linear(8, 8, bias=False),
                }
            )

        def _block_sparse_moe():
            return nn.ModuleDict(
                {
                    # grouped experts: fused input/output linear, not .pt-serveable
                    "input_linear": nn.Linear(8, 16, bias=False),
                    "output_linear": nn.Linear(8, 8, bias=False),
                    # router — leaf name is `layer` (block_sparse_moe.router.layer)
                    "router": nn.ModuleDict({"layer": nn.Linear(8, n_experts, bias=False)}),
                }
            )

        def _mamba():
            return nn.ModuleDict(
                {
                    "in_proj": nn.Linear(8, 16, bias=False),
                    "out_proj": nn.Linear(8, 8, bias=False),
                }
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mamba": _mamba(),
                        "block_sparse_moe": _block_sparse_moe(),
                        "shared_mlp": _shared_mlp(),
                    }
                ),
                nn.ModuleDict(
                    {
                        "self_attn": _attn(),
                        "block_sparse_moe": _block_sparse_moe(),
                        "shared_mlp": _shared_mlp(),
                    }
                ),
            ]
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_granite_hybrid_adapts_attention_and_shared_mlp_only():
    # Granite has no `.experts` submodule — MoE detection must fire on
    # `block_sparse_moe`, and resolution must exclude the grouped experts + router
    # (`block_sparse_moe.*`) and the Mamba SSM (`mamba.*`), keeping attention and
    # the dense shared MLP — by full path, since shared_mlp and the experts share
    # the `input_linear`/`output_linear` leaf names.
    model = _GraniteHybridLike()
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.shared_mlp.input_linear",
        "layers.0.shared_mlp.output_linear",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.shared_mlp.input_linear",
        "layers.1.shared_mlp.output_linear",
    ]
    assert not any(".block_sparse_moe." in name for name in result)  # no experts/router
    assert not any(".mamba." in name for name in result)  # no SSM projections
    assert not any("lm_head" in name for name in result)  # no output head


class _DeepseekV2Like(nn.Module):
    """A miniature DeepseekV2ForCausalLM (DeepSeek-V2-Lite): MLA attention + a MoE MLP
    with grouped routed experts, an always-on `shared_experts` MLP, and a `gate`
    router. Mirrors the real leaf naming (verified against the trained adapter):

    - MLA attention: self_attn.{q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj}
      (q_lora_rank=None → plain `q_proj`, not q_a/q_b). On serve, vLLM absorbs
      `kv_b_proj`/`kv_a_proj_with_mqa` into the latent-attention kernel, so LoRA on
      them is dropped — they must be EXCLUDED; `q_proj`/`o_proj` serve clean and stay.
    - MoE MLP (layer 1): grouped `experts` (nn.Parameters), `shared_experts.{gate,up,
      down}_proj` (dense, serveable), `gate` router.
    - dense MLP (layer 0): mlp.{gate,up,down}_proj.

    Expected: attention `q_proj`/`o_proj` + dense MLP + shared_experts, excluding the
    MLA latent projections, the grouped experts, and the router."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="deepseek_v2", vision_config=None)

        def _mla_attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "kv_a_proj_with_mqa": nn.Linear(8, 8, bias=False),
                    "kv_b_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _dense_mlp():
            return nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _moe_mlp():
            return nn.ModuleDict(
                {
                    "experts": _GroupedExperts(n_experts=4),
                    "shared_experts": nn.ModuleDict(
                        {
                            "gate_proj": nn.Linear(8, 8, bias=False),
                            "up_proj": nn.Linear(8, 8, bias=False),
                            "down_proj": nn.Linear(8, 8, bias=False),
                        }
                    ),
                    "gate": nn.Linear(8, 4, bias=False),  # router
                }
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({"self_attn": _mla_attn(), "mlp": _dense_mlp()}),
                nn.ModuleDict({"self_attn": _mla_attn(), "mlp": _moe_mlp()}),
            ]
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_deepseek_v2_excludes_mla_latent_projections():
    # DeepSeek-V2 MLA: the absorbed latent projections (kv_a_proj_with_mqa, kv_b_proj)
    # must be excluded (vLLM absorbs them → LoRA dropped on serve = the NO_MEM
    # near-miss), while q_proj/o_proj + dense MLP + shared_experts stay, and the
    # grouped experts + router are excluded as usual.
    model = _DeepseekV2Like()
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.1.mlp.shared_experts.down_proj",
        "layers.1.mlp.shared_experts.gate_proj",
        "layers.1.mlp.shared_experts.up_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
    ]
    assert not any("kv_a_proj_with_mqa" in n for n in result)  # MLA latent — absorbed
    assert not any("kv_b_proj" in n for n in result)  # MLA up-proj — absorbed
    assert not any(".experts." in n or n.endswith(".experts") for n in result)  # grouped experts off
    assert not any(n.endswith(".gate") for n in result)  # router off


def test_deepseek_v2_grouped_experts_via_target_parameters():
    # The grouped routed experts still surface as target_parameters (served by the
    # grouped converter) — the MLA exclusion only affects target_modules.
    model = _DeepseekV2Like()
    assert resolve_target_parameters(model) == ["down_proj", "gate_up_proj"]


class _FalconH1Like(nn.Module):
    """A miniature FalconH1ForCausalLM mirroring the real leaf naming (verified
    against transformers 5.x modeling_falcon_h1.py: FalconH1DecoderLayer has
    `self.self_attn`, `self.mamba`, `self.feed_forward`):

    - attention (every layer): self_attn.{q,k,v,o}_proj  (uses `o_proj`)
    - Mamba-2 mixer (every layer): mamba.{in_proj, out_proj} + a `conv1d`
      (nn.Conv1d, not Linear) — the SSM runs ‖ attention, and it holds the
      `out_proj` PEFT rejects on Mamba model_types.
    - dense MLP (every layer): feed_forward.{gate,up,down}_proj

    Dense (no MoE, no vision_config). `config.model_type == "falcon_h1"` so PEFT's
    `_check_lora_target_modules_mamba` would fire if `out_proj` reached the targets.
    Expected: LoRA on attention + MLP, with the whole `.mamba` subtree excluded."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type="falcon_h1", vision_config=None)

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _mamba():
            m = nn.ModuleDict(
                {
                    "in_proj": nn.Linear(8, 16, bias=False),
                    "out_proj": nn.Linear(8, 8, bias=False),
                }
            )
            # The 1-D short conv — an nn.Conv1d, not a Linear (so never a target
            # anyway), but present in the real mixer and named `conv1d`.
            m.conv1d = nn.Conv1d(8, 8, kernel_size=4, groups=8, bias=False)
            return m

        def _mlp():
            return nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "self_attn": _attn(),
                    "mamba": _mamba(),
                    "feed_forward": _mlp(),
                }
            )
            for _ in range(n_layers)
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_falcon_h1_dense_hybrid_adapts_attention_and_mlp_excluding_mamba():
    # FalconH1: dense Mamba-2 ‖ attention. The `.mamba` mixer subtree (incl. its
    # `out_proj`, which PEFT refuses on Mamba model_types) must be excluded, leaving
    # attention + MLP on every layer — by full path.
    model = _FalconH1Like(n_layers=2)
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.0.feed_forward.down_proj",
        "layers.0.feed_forward.gate_proj",
        "layers.0.feed_forward.up_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.1.feed_forward.down_proj",
        "layers.1.feed_forward.gate_proj",
        "layers.1.feed_forward.up_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
    ]
    assert not any(".mamba." in name for name in result)  # SSM mixer excluded
    assert not any(name.endswith(".out_proj") for name in result)  # the PEFT-rejected leaf
    assert not any("lm_head" in name for name in result)  # no output head


def test_falcon_h1_dense_hybrid_has_no_expert_target_parameters():
    # FalconH1 is dense (no grouped experts) — no target_parameters.
    assert resolve_target_parameters(_FalconH1Like()) == []


class _NemotronHMambaMixer(nn.Module):
    """NemotronH's Mamba-2 mixer: `in_proj`/`out_proj` (nn.Linear) + a `conv1d`
    and the selective-scan `A_log`/`D`/`dt_bias` parameters. The `A_log` param is
    what identifies it structurally — its module attribute is `mixer` (shared with
    attention/MLP blocks), NOT `.mamba`, so a name check can't tell it apart."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(8, 16, bias=False)
        self.out_proj = nn.Linear(8, 8, bias=False)
        self.conv1d = nn.Conv1d(8, 8, kernel_size=4, groups=8, bias=False)
        self.A_log = nn.Parameter(torch.zeros(8))
        self.D = nn.Parameter(torch.zeros(8))
        self.dt_bias = nn.Parameter(torch.zeros(8))


class _NemotronHLike(nn.Module):
    """A miniature NemotronHForCausalLM mirroring the real module naming (verified
    by meta-init of transformers 5.12 modeling_nemotron_h.py): every decoder layer
    holds a single `.mixer` whose class varies by block type
    (`hybrid_override_pattern`) —

    - Mamba-2 mixer: mixer.{in_proj, out_proj} + a `conv1d` and an `A_log` param
      (the selective-scan signature; the attribute is `.mixer`, NOT `.mamba`, so
      only the signature tells it apart),
    - attention mixer: mixer.{q,k,v,o}_proj (uses `o_proj`, not `out_proj`),
    - MLP mixer: mixer.{up,down}_proj.

    Dense (no MoE, no vision_config). `model_type == "nemotron_h"` is NOT in PEFT's
    Mamba-incompatible set, so PEFT would NOT reject the SSM `out_proj` — but vLLM
    drops LoRA on the SSM projections on serve, so they must still be excluded.
    Expected: LoRA on attention + MLP with the whole `.mixer` Mamba subtree off."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="nemotron_h", vision_config=None)

        def _attn():
            return nn.ModuleDict(
                {
                    "q_proj": nn.Linear(8, 8, bias=False),
                    "k_proj": nn.Linear(8, 8, bias=False),
                    "v_proj": nn.Linear(8, 8, bias=False),
                    "o_proj": nn.Linear(8, 8, bias=False),
                }
            )

        def _mlp():
            return nn.ModuleDict(
                {
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )

        # One of each block type, interleaved as NemotronH does.
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({"mixer": _NemotronHMambaMixer()}),
                nn.ModuleDict({"mixer": _mlp()}),
                nn.ModuleDict({"mixer": _attn()}),
            ]
        )
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_nemotron_h_dense_hybrid_adapts_attention_and_mlp_excluding_mamba():
    # NemotronH: dense Mamba-2 / attention / MLP interleave, mixer named `.mixer`.
    # The Mamba mixer is detected by its `A_log` signature (not by name) and its
    # in_proj/out_proj excluded, leaving attention + MLP — by full path.
    model = _NemotronHLike()
    result = resolve_target_modules(model, "all-linear")
    assert result == [
        "layers.1.mixer.down_proj",
        "layers.1.mixer.up_proj",
        "layers.2.mixer.k_proj",
        "layers.2.mixer.o_proj",
        "layers.2.mixer.q_proj",
        "layers.2.mixer.v_proj",
    ]
    # The SSM mixer's own projections are excluded (in_proj/out_proj), even though
    # PEFT would NOT reject them for model_type nemotron_h — vLLM drops them on serve.
    assert not any(name.endswith(".in_proj") for name in result)
    assert not any(name.endswith(".out_proj") for name in result)
    assert not any("lm_head" in name for name in result)  # no output head


def test_nemotron_h_dense_hybrid_has_no_expert_target_parameters():
    # NemotronH-8B is dense (no grouped experts) — no target_parameters.
    assert resolve_target_parameters(_NemotronHLike()) == []


def test_no_expert_target_parameters_for_non_grouped_layouts():
    # target_parameters is ONLY for grouped bare-nn.Parameter experts. Separate
    # per-expert nn.Linear experts (Mixtral/PhiMoE-in-v4) are trained via
    # target_modules, and Granite's nn.Linear-fused experts expose no bare param —
    # all three must yield NO target_parameters (else PEFT double-wraps / errors).
    assert resolve_target_parameters(_SeparateMoeLike(sparse_layers=(0,), n_layers=1)) == []
    assert resolve_target_parameters(_MixtralLike(n_experts=2)) == []
    assert resolve_target_parameters(_GraniteHybridLike()) == []


def test_all_linear_expands_to_distinct_leaf_names_minus_head():
    model = _DenseLike()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == [
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert "lm_head" not in resolved


def test_repeated_leaf_names_collapse_to_a_set():
    # 3 layers × 7 projections, but only 7 distinct leaf names survive.
    model = _DenseLike(n_layers=3)
    resolved = resolve_target_modules(model, "all-linear")
    assert len(resolved) == 7


def test_output_head_excluded_by_name_when_no_output_embeddings():
    model = _NoOutputEmbeddings()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == ["q_proj"]


class _Decoder(nn.Module):
    """A miniature language tower: one block of attention + MLP projections."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(8, 8, bias=False),
                        "k_proj": nn.Linear(8, 8, bias=False),
                        "v_proj": nn.Linear(8, 8, bias=False),
                        "o_proj": nn.Linear(8, 8, bias=False),
                    }
                )
            ]
        )


class _VisionTower(nn.Module):
    """A vision encoder that REUSES the language tower's leaf names — the
    Gemma3 case that defeats plain leaf-name targeting."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "q_proj": nn.Linear(8, 8, bias=False),
                "k_proj": nn.Linear(8, 8, bias=False),
            }
        )


class _MultimodalModel(nn.Module):
    """A ...ForConditionalGeneration-shaped wrapper: a `vision_config` on the
    config, `get_decoder()` returning the language tower, a vision tower whose
    linears share leaf names with the language tower, and a tied output head."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8))
        self.language_model = _Decoder()
        self.vision_tower = _VisionTower()
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_decoder(self):
        return self.language_model

    def get_output_embeddings(self):
        return self.lm_head


def test_multimodal_targets_full_paths_under_language_decoder_only():
    model = _MultimodalModel()
    resolved = resolve_target_modules(model, "all-linear")
    # Every target is a full path inside the language tower...
    assert resolved == [
        "language_model.layers.0.k_proj",
        "language_model.layers.0.o_proj",
        "language_model.layers.0.q_proj",
        "language_model.layers.0.v_proj",
    ]
    # ...and nothing in the vision tower, despite the shared k_proj/q_proj names.
    assert not any("vision_tower" in name for name in resolved)


def test_multimodal_resolution_excludes_output_head():
    model = _MultimodalModel()
    resolved = resolve_target_modules(model, "all-linear")
    assert not any(name.endswith("lm_head") for name in resolved)


class _MoeDecoder(nn.Module):
    """A Qwen3.5-VL-MoE-style language decoder: HYBRID attention (full `self_attn`
    + linear-attention `linear_attn` GatedDeltaNet) with an all-MoE MLP — grouped
    routed experts (`_GroupedExperts` nn.Parameters), an always-on dense
    `shared_expert`, a `shared_expert_gate`, and a `gate` router."""

    def __init__(self, n_layers=2):
        super().__init__()

        def _layer():
            return nn.ModuleDict(
                {
                    "self_attn": nn.ModuleDict(
                        {
                            "q_proj": nn.Linear(8, 8, bias=False),
                            "k_proj": nn.Linear(8, 8, bias=False),
                            "v_proj": nn.Linear(8, 8, bias=False),
                            "o_proj": nn.Linear(8, 8, bias=False),
                        }
                    ),
                    "linear_attn": nn.ModuleDict(
                        {
                            "in_proj_qkv": nn.Linear(8, 8, bias=False),
                            "out_proj": nn.Linear(8, 8, bias=False),
                        }
                    ),
                    "mlp": nn.ModuleDict(
                        {
                            "experts": _GroupedExperts(4),
                            "shared_expert": nn.ModuleDict(
                                {
                                    "gate_proj": nn.Linear(8, 8, bias=False),
                                    "up_proj": nn.Linear(8, 8, bias=False),
                                    "down_proj": nn.Linear(8, 8, bias=False),
                                }
                            ),
                            "shared_expert_gate": nn.Linear(8, 1, bias=False),
                            "gate": nn.Linear(8, 4, bias=False),
                        }
                    ),
                }
            )

        self.layers = nn.ModuleList([_layer() for _ in range(n_layers)])


class _MultimodalMoeModel(nn.Module):
    """Qwen3.5-VL-MoE-shaped (`Qwen3_5MoeForConditionalGeneration`): a multimodal
    wrapper whose language decoder is itself MoE + hybrid-attention."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=8))
        self.language_model = _MoeDecoder()
        self.vision_tower = _VisionTower()
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_decoder(self):
        return self.language_model

    def get_output_embeddings(self):
        return self.lm_head


def test_multimodal_moe_confines_to_decoder_attention_and_shared_expert():
    # Qwen3.5-VL-MoE: the multimodal branch is MoE-aware, so LoRA lands on the
    # decoder's attention + dense shared_expert ONLY — never the vision tower, the
    # router (`gate` / `shared_expert_gate`), the linear-attention SSM
    # (`linear_attn`), or the grouped routed experts (those go via
    # target_parameters). shared_expert carries memorization (granite precedent).
    model = _MultimodalMoeModel()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == [
        "language_model.layers.0.mlp.shared_expert.down_proj",
        "language_model.layers.0.mlp.shared_expert.gate_proj",
        "language_model.layers.0.mlp.shared_expert.up_proj",
        "language_model.layers.0.self_attn.k_proj",
        "language_model.layers.0.self_attn.o_proj",
        "language_model.layers.0.self_attn.q_proj",
        "language_model.layers.0.self_attn.v_proj",
        "language_model.layers.1.mlp.shared_expert.down_proj",
        "language_model.layers.1.mlp.shared_expert.gate_proj",
        "language_model.layers.1.mlp.shared_expert.up_proj",
        "language_model.layers.1.self_attn.k_proj",
        "language_model.layers.1.self_attn.o_proj",
        "language_model.layers.1.self_attn.q_proj",
        "language_model.layers.1.self_attn.v_proj",
    ]
    assert not any("vision_tower" in n for n in resolved)  # vision tower off
    assert not any(".linear_attn." in n for n in resolved)  # SSM off
    assert not any(".experts" in n for n in resolved)  # grouped experts off
    assert not any(n.endswith(".gate") or n.endswith("_gate") for n in resolved)  # routers off


def test_multimodal_moe_grouped_experts_via_target_parameters():
    # The grouped routed experts are unreachable by target_modules (nn.Parameters),
    # so resolve_target_parameters names them for PEFT's ParamWrapper — exactly the
    # {gate_up_proj, down_proj} set (requires lora_dropout=0 at train time).
    model = _MultimodalMoeModel()
    assert resolve_target_parameters(model) == ["down_proj", "gate_up_proj"]


def test_explicit_list_is_passed_through_unchanged():
    model = _DenseLike()
    explicit = ["q_proj", "v_proj"]
    assert resolve_target_modules(model, explicit) is explicit


def test_explicit_non_shorthand_string_passes_through():
    model = _DenseLike()
    # A regex/string that isn't the "all-linear" shorthand is PEFT's own
    # to interpret — we must not touch it.
    assert resolve_target_modules(model, "q_proj") == "q_proj"


def test_none_passes_through():
    model = _DenseLike()
    assert resolve_target_modules(model, None) is None
