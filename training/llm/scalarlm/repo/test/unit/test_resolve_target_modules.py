from types import SimpleNamespace

import torch
import torch.nn as nn

from adapters.resolve_target_modules import (
    resolve_target_modules,
    resolve_target_parameters,
)


class _DenseLike(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_output_embeddings(self):
        return None


class _GroupedExperts(nn.Module):
    def __init__(self, n_experts, dim=8, inter=8):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(n_experts, dim, 2 * inter))
        self.down_proj = nn.Parameter(torch.zeros(n_experts, inter, dim))


class _MoeLike(nn.Module):
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
    model = _MoeLike(sparse_layers=(1,), n_layers=2)
    result = resolve_target_modules(model, "all-linear")
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
    assert not any(".experts." in name for name in result)
    assert not any(name.endswith(".gate") for name in result)
    assert not any("lm_head" in name for name in result)


def test_moe_excludes_router_named_router_phimoe():
    model = _MoeLike(sparse_layers=(1,), n_layers=2, router_name="router")
    result = resolve_target_modules(model, "all-linear")
    assert not any(name.endswith(".router") for name in result)
    assert "layers.1.mlp.router" not in result
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
    holder = nn.ModuleDict({"experts": _GroupedExperts(n_experts=4)})
    assert resolve_target_parameters(holder) == ["down_proj", "gate_up_proj"]


def test_grouped_moe_target_parameters_regardless_of_dense_layer():
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

        def _expert():
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
    assert any(".experts.0.w1" in name for name in result)
    assert not any(name.endswith(".router") for name in result)
    assert not any("lm_head" in name for name in result)


def test_separate_expert_moe_includes_experts_and_dense_mlp():
    model = _SeparateMoeLike(sparse_layers=(1,), n_layers=2, n_experts=2)
    result = resolve_target_modules(model, "all-linear")
    assert "layers.0.mlp.gate_proj" in result
    assert "layers.1.mlp.experts.1.w2" in result
    assert "layers.0.self_attn.q_proj" in result
    assert not any(name.endswith(".router") for name in result)
    assert not any("lm_head" in name for name in result)


class _MixtralLike(nn.Module):
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
                                "gate": nn.Linear(8, n_experts, bias=False),
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
    assert not any(name.endswith(".gate") for name in result)


class _GraniteHybridLike(nn.Module):
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
                    "input_linear": nn.Linear(8, 16, bias=False),
                    "output_linear": nn.Linear(8, 8, bias=False),
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
    assert not any(".block_sparse_moe." in name for name in result)
    assert not any(".mamba." in name for name in result)
    assert not any("lm_head" in name for name in result)


class _DeepseekV2Like(nn.Module):
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
                    "gate": nn.Linear(8, 4, bias=False),
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
    assert not any("kv_a_proj_with_mqa" in n for n in result)
    assert not any("kv_b_proj" in n for n in result)
    assert not any(".experts." in n or n.endswith(".experts") for n in result)
    assert not any(n.endswith(".gate") for n in result)


def test_deepseek_v2_grouped_experts_via_target_parameters():
    model = _DeepseekV2Like()
    assert resolve_target_parameters(model) == ["down_proj", "gate_up_proj"]


class _FalconH1Like(nn.Module):
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
    assert not any(".mamba." in name for name in result)
    assert not any(name.endswith(".out_proj") for name in result)
    assert not any("lm_head" in name for name in result)


def test_falcon_h1_dense_hybrid_has_no_expert_target_parameters():
    assert resolve_target_parameters(_FalconH1Like()) == []


class _NemotronHMambaMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(8, 16, bias=False)
        self.out_proj = nn.Linear(8, 8, bias=False)
        self.conv1d = nn.Conv1d(8, 8, kernel_size=4, groups=8, bias=False)
        self.A_log = nn.Parameter(torch.zeros(8))
        self.D = nn.Parameter(torch.zeros(8))
        self.dt_bias = nn.Parameter(torch.zeros(8))


class _NemotronHLike(nn.Module):
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
    assert not any(name.endswith(".in_proj") for name in result)
    assert not any(name.endswith(".out_proj") for name in result)
    assert not any("lm_head" in name for name in result)


def test_nemotron_h_dense_hybrid_has_no_expert_target_parameters():
    assert resolve_target_parameters(_NemotronHLike()) == []


def test_no_expert_target_parameters_for_non_grouped_layouts():
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
    model = _DenseLike(n_layers=3)
    resolved = resolve_target_modules(model, "all-linear")
    assert len(resolved) == 7


def test_output_head_excluded_by_name_when_no_output_embeddings():
    model = _NoOutputEmbeddings()
    resolved = resolve_target_modules(model, "all-linear")
    assert resolved == ["q_proj"]


class _Decoder(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "q_proj": nn.Linear(8, 8, bias=False),
                "k_proj": nn.Linear(8, 8, bias=False),
            }
        )


class _MultimodalModel(nn.Module):
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
    assert resolved == [
        "language_model.layers.0.k_proj",
        "language_model.layers.0.o_proj",
        "language_model.layers.0.q_proj",
        "language_model.layers.0.v_proj",
    ]
    assert not any("vision_tower" in name for name in resolved)


def test_multimodal_resolution_excludes_output_head():
    model = _MultimodalModel()
    resolved = resolve_target_modules(model, "all-linear")
    assert not any(name.endswith("lm_head") for name in resolved)


class _MoeDecoder(nn.Module):
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
    assert not any("vision_tower" in n for n in resolved)
    assert not any(".linear_attn." in n for n in resolved)
    assert not any(".experts" in n for n in resolved)
    assert not any(n.endswith(".gate") or n.endswith("_gate") for n in resolved)


def test_multimodal_moe_grouped_experts_via_target_parameters():
    model = _MultimodalMoeModel()
    assert resolve_target_parameters(model) == ["down_proj", "gate_up_proj"]


def test_explicit_list_is_passed_through_unchanged():
    model = _DenseLike()
    explicit = ["q_proj", "v_proj"]
    assert resolve_target_modules(model, explicit) is explicit


def test_explicit_non_shorthand_string_passes_through():
    model = _DenseLike()
    assert resolve_target_modules(model, "q_proj") == "q_proj"


def test_none_passes_through():
    model = _DenseLike()
    assert resolve_target_modules(model, None) is None
