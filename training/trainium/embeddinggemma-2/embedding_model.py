"""Model loading, static-shape adaptation, loss, metrics, and state checks."""

import copy
import hashlib
import json
import math


def device_smoke():
    """Run a small compiled forward, backward, and optimizer update."""
    import torch

    layer = torch.nn.Linear(4, 2).to("neuron")
    before = layer.weight.detach().cpu().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    forward = torch.compile(layer, backend="neuron", dynamic=False)
    loss = forward(torch.ones(2, 4, device="neuron")).square().mean()
    loss.backward()
    optimizer.step()
    after = layer.weight.detach().cpu()
    if not torch.isfinite(after).all() or torch.equal(before, after):
        raise RuntimeError(
            "Native compiled update did not produce finite changed weights."
        )


def configure_static_preprocessing(model):
    """Pad tokenizer outputs to the configured static sequence length."""
    import torch

    original_preprocess = model.preprocess

    def preprocess(inputs, **kwargs):
        length = model.max_seq_length
        kwargs.update(max_length=length, truncation=True)
        features = original_preprocess(inputs, **kwargs)
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in features:
                continue
            pad_value = model.tokenizer.pad_token_id if key == "input_ids" else 0
            tensor = features[key]
            if tensor.shape[1] > length:
                raise ValueError(
                    "Preprocessing exceeded the fixed sequence length."
                )
            features[key] = torch.nn.functional.pad(
                tensor, (0, length - tensor.shape[1]), value=pad_value
            )
        return features

    model.preprocess = preprocess


def build_loss(model, dimensions):
    from sentence_transformers.sentence_transformer.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.util import dot_score

    width = model.get_embedding_dimension()
    invalid_dimensions = (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or width not in dimensions
        or any(dimension <= 0 or dimension > width for dimension in dimensions)
    )
    if invalid_dimensions:
        raise ValueError(
            f"Use unique positive dimensions <= {width}, including {width}."
        )
    base_loss = MultipleNegativesRankingLoss(
        model,
        scale=20.0,
        similarity_fct=dot_score,
        gather_across_devices=False,
    )
    return MatryoshkaLoss(
        model,
        base_loss,
        matryoshka_dims=dimensions,
        matryoshka_weights=[1.0] * len(dimensions),
        n_dims_per_step=-1,
    )


def cpu_cosine(query_embeddings, document_embeddings):
    import torch
    import torch.nn.functional as functional

    with torch.autocast("cpu", enabled=False):
        queries = query_embeddings.detach().to(
            device="cpu", dtype=torch.float32
        )
        documents = document_embeddings.detach().to(
            device="cpu", dtype=torch.float32
        )
        return (
            functional.normalize(queries, dim=1)
            @ functional.normalize(documents, dim=1).T
        )


def extract_retrieval_metrics(raw_metrics):
    result = {
        name: float(
            next(
                value
                for key, value in raw_metrics.items()
                if key.endswith("cosine_" + name)
            )
        )
        for name in ("ndcg@10", "recall@10", "mrr@10")
    }
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in result.values()
    ):
        raise ValueError("Invalid retrieval metrics.")
    return result


def state_hashes(model):
    import torch

    result = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite model tensor: {name}")
        result[name] = hashlib.sha256(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    return result


def load_embedding_model(args, model_path, revision=None):
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerModelCardData,
    )

    model = SentenceTransformer(
        str(model_path),
        revision=revision,
        device="cpu",
        cache_folder=str(args.model_cache_dir),
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={
            "dtype": torch.float32,
            "attn_implementation": "eager",
        },
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="EmbeddingGemma 2 Matryoshka text fine-tune",
        ),
    )
    model.max_seq_length = args.max_seq_length
    return model


def activate_on_neuron(model, device):
    import torch

    model.to(device)
    configure_static_preprocessing(model)
    model.forward = torch.compile(
        model.forward, backend="neuron", dynamic=False
    )


def self_test():
    """Verify loss and gradient semantics on CPU."""
    import torch
    import torch.nn.functional as functional
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Dense

    torch.manual_seed(42)
    model = SentenceTransformer(
        modules=[
            Dense(8, 768, activation_function=torch.nn.Identity())
        ],
        device="cpu",
    )
    reference = copy.deepcopy(model)
    model.forward = torch.compile(model.forward, backend="eager", dynamic=False)
    compiled_forward = model.forward
    inputs = [torch.randn(3, 8) for _ in range(3)]
    dimensions = [768, 512, 256, 128]
    loss = build_loss(model, dimensions)
    assert not loss.uses_gradient_cache

    actual = loss(
        [{"sentence_embedding": value.clone()} for value in inputs], None
    )
    actual.backward()
    reference_embeddings = [
        reference({"sentence_embedding": value.clone()})["sentence_embedding"]
        for value in inputs
    ]
    expected = 0
    for dimension in dimensions:
        query, positive, negative = [
            functional.normalize(value[:, :dimension], dim=-1)
            for value in reference_embeddings
        ]
        expected += functional.cross_entropy(
            20 * query @ torch.cat([positive, negative]).T,
            torch.arange(3),
        )
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for parameter, reference_parameter in zip(
        model.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=1e-4,
            atol=1e-5,
        )
        assert torch.isfinite(parameter.grad).all()
    assert model.forward is compiled_forward

    before = state_hashes(model)
    torch.optim.AdamW(model.parameters(), lr=1e-5).step()
    assert before != state_hashes(model)
    for invalid in ([769, 128], [128], [768, 0], [768, 768]):
        try:
            build_loss(model, invalid)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid dimensions: {invalid}")
    print(
        json.dumps(
            {
                "passed": True,
                "device": "cpu",
                "checks": [
                    "loss",
                    "gradients",
                    "finite_gradients",
                    "forward_restoration",
                    "weight_update",
                    "invalid_dimensions",
                ],
            }
        )
    )
