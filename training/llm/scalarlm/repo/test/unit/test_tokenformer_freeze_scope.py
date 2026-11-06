from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))

from tokenformer.tokenformer_surgeon import is_non_language_path


def test_language_model_paths_are_trainable():
    assert not is_non_language_path("model.layers.0.self_attn.q_proj.weight")
    assert not is_non_language_path("model.language_model.layers.3.mlp.tokenformer_p")
    assert not is_non_language_path("lm_head.weight")


def test_vision_and_audio_towers_are_excluded():
    assert is_non_language_path(
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
    )
    assert is_non_language_path("model.audio_tower.encoder.layers.0.input_layernorm.weight")
    assert is_non_language_path("model.multi_modal_projector.weight")
    assert is_non_language_path("model.embed_vision.weight")
    assert is_non_language_path("model.embed_audio.weight")


def test_matching_is_by_path_component_not_substring():
    assert not is_non_language_path("model.layers.0.vision_tower_gate.weight")


def test_unfreeze_list_would_otherwise_catch_the_vision_tower():
    unfreeze_substrings = [
        "tokenformer", "q_proj", "k_proj", "v_proj", "norm", "rotary_emb",
        "embed_tokens", "input_layernorm", "post_attention_layernorm", "o_proj",
    ]
    vision_param = "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"

    assert any(s in vision_param for s in unfreeze_substrings)
    assert is_non_language_path(vision_param)
