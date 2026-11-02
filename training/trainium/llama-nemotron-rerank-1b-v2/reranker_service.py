"""Model training, evaluation, checkpoint, and verification services."""
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types

MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
REVISION = "828765652b05bd439c9789d2a6d093db1caa1443"


from reranker_data import prompt, write_json



def encode(tokenizer, texts, max_len):
    return tokenizer(texts, padding="max_length", max_length=max_len,
                     truncation=True, return_tensors="pt")


def explicit_mask(self, input_embeds, attention_mask):
    import torch
    mask = 1 - attention_mask[:, None, None, :].to(input_embeds.dtype)
    return (mask * torch.finfo(input_embeds.dtype).min).expand(
        -1, 1, input_embeds.shape[1], -1)


def tensor_hashes(model, torch):
    result = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite checkpoint tensor: {name}")
        result[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def save_model(model, destination):
    cls = type(model.config)
    original = cls._get_generation_parameters

    def generation_parameters(config):
        values = original(config)
        values.pop("temperature", None)
        return values

    cls._get_generation_parameters = generation_parameters
    try:
        model.save_pretrained(destination)
    finally:
        cls._get_generation_parameters = original


def ranking_metrics(scores):
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Scores must be non-empty and finite")
    rank = 1 + sum(score > scores[0] for score in scores[1:])
    return {"scores": scores, "rank": rank,
            "ndcg@10": 1 / math.log2(rank + 1) if rank <= 10 else 0.0,
            "mrr@10": 1 / rank if rank <= 10 else 0.0}


def evaluate_model(model, valid, args, tokenizer, torch):
    model.eval()
    records = []
    with torch.no_grad():
        for row in valid:
            texts = [prompt(row["anchor"], row[key]) for key in
                     ["positive"] + [f"negative_{i}" for i in range(1, args.n_neg + 1)]]
            scores = []
            for start in range(0, len(texts), args.microbatch):
                chunk = texts[start:start + args.microbatch]
                count = len(chunk)
                chunk += [chunk[-1]] * (args.microbatch - count)
                batch = {key: value.to("neuron") for key, value in
                         encode(tokenizer, chunk, args.max_len).items()}
                values = model(**batch).logits.detach().float().cpu().flatten().tolist()[:count]
                scores.extend(values)
            records.append(ranking_metrics(scores))
    return {"ndcg@10": sum(row["ndcg@10"] for row in records) / len(records),
            "mrr@10": sum(row["mrr@10"] for row in records) / len(records),
            "queries": records}


def load_and_adapt_model(args, out, pairs, torch, model_class, tokenizer_class):
    tokenizer = tokenizer_class.from_pretrained(
        MODEL, revision=REVISION, cache_dir=args.weights,
        trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer(text)["input_ids"]) for text, _ in pairs]
    write_json(out / "lengths.json", {
        "max": max(lengths), "pairs": len(lengths),
        "truncated": sum(length > args.max_len for length in lengths),
        "max_length": args.max_len})

    model = model_class.from_pretrained(
        MODEL, revision=REVISION, cache_dir=args.weights,
        trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    probe = encode(tokenizer, [pairs[0][0], pairs[1][0]], args.max_len)
    model.eval()
    with torch.no_grad():
        original = model(**probe).logits.float()
    model.model._create_bidirectional_mask = types.MethodType(explicit_mask, model.model)
    with torch.no_grad():
        adapted = model(**probe).logits.float()
    torch.testing.assert_close(original, adapted, rtol=0.01, atol=0.05)
    write_json(out / "mask-parity.json", {
        "original": original.tolist(),
        "adapted": adapted.tolist(),
        "max_abs_difference": (original - adapted).abs().max().item()})
    return model, tokenizer


def training_loop(model, tokenizer, pairs, valid, args, out, torch,
                  scheduler_factory, resolved_device, placement):
    batches_per_epoch = len(pairs) // args.batch
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_steps = 2 if args.smoke else steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, foreach=False)
    scheduler = scheduler_factory(
        optimizer, 0 if args.smoke else math.ceil(total_steps * 0.1), total_steps)
    logs, best_score, best_epoch = [], -float("inf"), None
    for epoch in range(1 if args.smoke else args.epochs):
        ordered = list(pairs)
        random.Random(args.seed + epoch).shuffle(ordered)
        ordered = ordered[:batches_per_epoch * args.batch]
        group_size = args.batch * args.gradient_accumulation
        groups = [ordered[i:i + group_size] for i in range(0, len(ordered), group_size)]
        if args.smoke:
            groups = [ordered[:args.microbatch], ordered[args.microbatch:2 * args.microbatch]]
        model.train()
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            step_start = time.monotonic()
            summed = 0.0
            for start in range(0, len(group), args.microbatch):
                chunk = group[start:start + args.microbatch]
                batch = {k: v.to("neuron") for k, v in encode(tokenizer, [x[0] for x in chunk], args.max_len).items()}
                labels = torch.tensor([x[1] for x in chunk], device="neuron")
                logits = model(**batch).logits.flatten().float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                value = loss.detach().cpu().item()
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite training loss")
                (loss * (len(chunk) / len(group))).backward()
                summed += value * len(chunk) / len(group)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            if not logs:
                gradient_devices = {p.grad.device for p in model.parameters() if p.grad is not None}
                if gradient_devices != {resolved_device}:
                    raise RuntimeError(f"Unexpected gradient placement: {gradient_devices}")
                if any(t.device != resolved_device for t in [*batch.values(), labels, logits]):
                    raise RuntimeError("Training inputs/output and model are on different devices")
                placement.update({"input_device": str(batch["input_ids"].device),
                                  "labels_device": str(labels.device),
                                  "logits_device": str(logits.device),
                                  "gradient_devices": sorted(map(str, gradient_devices))})
                write_json(out / "placement.json", placement)
            norm_value = norm.detach().cpu().item()
            if not math.isfinite(norm_value):
                raise RuntimeError("Non-finite gradient norm")
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            record = {"step": len(logs) + 1, "epoch": epoch + 1, "loss": summed,
                      "gradient_norm": norm_value, "lr": lr, "pairs": len(group),
                      "seconds": time.monotonic() - step_start}
            logs.append(record)
            write_json(out / "steps.json", logs)
            print(json.dumps(record), flush=True)
        metric = evaluate_model(model, valid, args, tokenizer, torch)
        write_json(out / f"epoch-{epoch + 1}.json", metric)
        checkpoint = out / f"checkpoint-{epoch + 1}"
        save_model(model, checkpoint)
        tokenizer.save_pretrained(checkpoint)
        if metric["ndcg@10"] > best_score:
            best_score, best_epoch = metric["ndcg@10"], epoch + 1
    return optimizer, scheduler, logs, best_epoch


def export_and_verify(model, model_class, tokenizer, valid, args, out, torch,
                      before, best_epoch, resolved_device, placement):
    after = tensor_hashes(model, torch)
    changed = [k for k in before if before[k] != after[k]]
    if not changed or not any("layers." in k for k in changed):
        raise RuntimeError("No backbone weight updates detected")
    selected = out / f"checkpoint-{best_epoch}"
    reloaded = model_class.from_pretrained(selected,
                trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
    # Compare against the saved selected checkpoint via independent serialization.
    selected_hashes = tensor_hashes(reloaded, torch)
    final = out / "final"
    save_model(reloaded, final)
    tokenizer.save_pretrained(final)
    del reloaded
    gc.collect()
    reloaded = model_class.from_pretrained(final,
                trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
    if tensor_hashes(reloaded, torch) != selected_hashes:
        raise RuntimeError("Export/reload tensor hash mismatch")
    selected_changed = [k for k in before if before[k] != selected_hashes[k]]
    if not selected_changed:
        raise RuntimeError("Selected checkpoint did not change")
    reloaded.model._create_bidirectional_mask = types.MethodType(explicit_mask, reloaded.model)
    reloaded.to("neuron")
    if {p.device for p in reloaded.parameters()} != {resolved_device}:
        raise RuntimeError("Reloaded model is on a different device")
    placement["reloaded_device"] = str(next(reloaded.parameters()).device)
    write_json(out / "placement.json", placement)
    reloaded.forward = torch.compile(reloaded.forward, backend="neuron", dynamic=False)
    trained = evaluate_model(reloaded, valid, args, tokenizer, torch)
    selected_metrics = json.loads((out / f"epoch-{best_epoch}.json").read_text())
    for a, b in zip(trained["queries"], selected_metrics["queries"]):
        torch.testing.assert_close(torch.tensor(a["scores"]), torch.tensor(b["scores"]), rtol=0.01, atol=0.05)
    return trained, selected_changed, changed


def run_finetuning(args, out, pairs, valid):
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for variable, name in [("TORCH_NEURONX_DEBUG_DIR", "debug"),
                           ("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"),
                           ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                           ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local")]:
        os.environ[variable] = str(Path(args.cache).resolve() / name)
    import torch
    import torch_neuronx  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    if not torch.neuron.is_available():
        raise RuntimeError("Native Neuron is unavailable; CPU fallback is prohibited")
    torch.manual_seed(args.seed)
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 32
    write_json(out / "environment.json", {"python": sys.version.split()[0],
          "isolated_mode": bool(sys.flags.isolated),
          "versions": {p: importlib.metadata.version(p) for p in
                       ["torch", "torch-neuronx", "transformers", "neuronx-cc", "nki"]},
          "device": "neuron"})
    model, tokenizer = load_and_adapt_model(
        args, out, pairs, torch,
        AutoModelForSequenceClassification, AutoTokenizer)

    before = tensor_hashes(model, torch)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("neuron")
    parameter_devices = {p.device for p in model.parameters()}
    if len(parameter_devices) != 1 or next(iter(parameter_devices)).type != "neuron":
        raise RuntimeError(f"Unexpected model placement: {parameter_devices}")
    resolved_device = next(iter(parameter_devices))
    placement = {"requested_device": "neuron", "model_device": str(resolved_device)}
    write_json(out / "placement.json", placement)
    model.forward = torch.compile(model.forward, backend="neuron", dynamic=False)

    baseline = evaluate_model(model, valid, args, tokenizer, torch)
    write_json(out / "baseline.json", baseline)
    started = time.monotonic()
    optimizer, scheduler, logs, best_epoch = training_loop(
        model, tokenizer, pairs, valid, args, out, torch,
        get_linear_schedule_with_warmup, resolved_device, placement)
    del optimizer, scheduler
    trained, selected_changed, changed = export_and_verify(
        model, AutoModelForSequenceClassification, tokenizer, valid, args, out,
        torch, before, best_epoch, resolved_device, placement)
    write_json(out / "result.json", {"status": "completed", "smoke": args.smoke,
          "optimizer_steps": len(logs), "selected_epoch": best_epoch,
          "selected_changed_tensors": len(selected_changed), "last_epoch_changed_tensors": len(changed),
          "reload_check_passed": True, "finite_checks_passed": True,
          "baseline": baseline, "trained": trained,
          "quality_improved": trained["ndcg@10"] > baseline["ndcg@10"],
          "test_set_used": False, "seconds": time.monotonic() - started})
