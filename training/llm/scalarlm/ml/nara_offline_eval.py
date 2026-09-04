#!/usr/bin/env python3
"""Offline NaRA memorize eval for DiffusionGemma.

    python nara_offline_eval.py --checkpoint <path> [--model ID] [--prompt TEXT]
        [--golden TEXT] [--mode decode|probe|sweep|decode-greedy|tail-probe|
        gen-vs-tf|decode-recompute] [--seed N] [--dtype bf16|fp32]
"""
import argparse, sys, time, difflib, json, random

sys.path.insert(0, "/app/cray/ml")
sys.path.insert(0, "/app/cray/infra")

import torch


def block_scores(sample: str, golden: str, prompt: str = ""):
    """Longest contiguous match + total matched chars between golden and sample."""
    if prompt and sample.startswith(prompt):
        sample = sample[len(prompt):].lstrip()
    sm = difflib.SequenceMatcher(None, golden, sample, autojunk=False)
    m = sm.find_longest_match(0, len(golden), 0, len(sample))
    total = sum(b.size for b in sm.get_matching_blocks())
    return m.size, total


def load_base(model_id, dtype=torch.bfloat16):
    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_id, dtype=dtype, device_map={"": 0},
    )
    model.eval()
    print(f"[load] base model+tokenizer ({dtype}) in {time.time()-t0:.1f}s", flush=True)
    return model, tok


def inject_and_load(model, ckpt_path):
    """Inject NaRA into the modules the checkpoint adapted and load lora + mapper weights."""
    from adapters.nara_prototype import inject_nara, NaRAConfig, find_nara_context

    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    lora_keys = [k for k in sd if ".lora_A" in k]
    # backbone-relative module paths -> outer-model paths (outer.model.<path>)
    mod_paths = sorted({k.rsplit(".lora_A", 1)[0] for k in lora_keys})
    outer_targets = ["model." + p for p in mod_paths]
    r = sd[lora_keys[0]].shape[0]
    mapper_keys = [k for k in sd if "nara_context.mapper" in k]

    cfg = NaRAConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, c_scale=0.1)
    inject_nara(model, outer_targets, cfg)

    # checkpoint keys are backbone-relative => load into the backbone (model.model)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    # inject_nara builds params on CPU/fp32 while the base is on GPU/bf16; move them to match.
    base_p = next(model.parameters())
    model.model.to(device=base_p.device, dtype=base_p.dtype)
    loaded_lora = [k for k in sd if ".lora_" in k]
    # sanity: adapter B must be non-zero after load (trained), mapper present+loaded
    some_B = [model.model.state_dict()[k] for k in sd if k.endswith(".lora_B")][:50]
    b_nonzero = sum(float(b.abs().sum() > 0) for b in some_B)
    report = {
        "adapted_modules": len(mod_paths), "rank": r,
        "lora_tensors_loaded": len(loaded_lora), "mapper_tensors": len(mapper_keys),
        "unexpected_keys": [k for k in unexpected], "b_nonzero_of_50": b_nonzero,
    }
    ctx = find_nara_context(model)
    return ctx, report


def make_mapper_hook(model, ctx, eps):
    """Wrap _denoising_step so the shared mapper sees the current noise level."""
    orig = model._denoising_step
    steps_seen = {"n": 0, "t_first": None, "t_last": None}

    def find_sampler(args, kwargs):
        s = kwargs.get("sampler")
        if s is not None:
            return s
        for a in args:
            if a.__class__.__name__ == "EntropyBoundSampler":
                return a
        return None

    def wrapped(*args, **kwargs):
        s = find_sampler(args, kwargs)
        am = getattr(s, "accepted_token_mask", None) if s is not None else None
        if am is None:
            t = 1.0
        else:
            t = float((~am).float().mean().item())
        t = max(t, eps)
        ctx.set_noise_level(torch.tensor([t], dtype=torch.float32))
        steps_seen["n"] += 1
        if steps_seen["t_first"] is None:
            steps_seen["t_first"] = t
        steps_seen["t_last"] = t
        return orig(*args, **kwargs)

    model._denoising_step = wrapped
    return steps_seen


def decode(model, tok, prompt, seed, gen_kwargs):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             **gen_kwargs)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tok.decode(seq[0], skip_special_tokens=True)
    return text


def get_vocab_size(model, tok):
    """Full embedding vocab - the corruption sampler's upper bound."""
    cfg = getattr(model, "config", None)
    for path in (("text_config", "vocab_size"), ("vocab_size",)):
        obj = cfg
        for p in path:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, int):
            return obj
    return len(tok)


def needs_mm_token(model):
    """True when the config has a vision_config, meaning the encoder wants mm_token_type_ids."""
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", cfg)
    return getattr(cfg, "vision_config", None) is not None or getattr(tc, "vision_config", None) is not None


def build_golden_canvas(tok, golden, canvas_length, anchor):
    """Build the golden output canvas as the training loader does."""
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id
    if pad_id is None:
        pad_id = 0
    anchor_id = getattr(tok, "bos_token_id", None) if anchor else None
    prefix = [anchor_id] if anchor_id is not None else []
    toks = list(tok(golden, add_special_tokens=False)["input_ids"])
    if canvas_length and canvas_length > 0:
        budget = canvas_length - len(prefix)
        if len(toks) > budget:
            print(f"[probe] golden {len(toks)} tok > budget {budget}; truncating", flush=True)
            toks = toks[:budget]
        pad = canvas_length - len(prefix) - len(toks)
    else:
        pad = 0
    canvas_input_ids = prefix + toks + [pad_id] * pad
    canvas_labels = prefix + toks + [-100] * pad
    return canvas_input_ids, canvas_labels, (1 if anchor_id is not None else 0)


def forward_canvas_logits(model, enc, decoder_input_ids, needs_mm, sc_logits=None):
    """One teacher-forced denoising pass through the joint model.forward."""
    kw = dict(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        decoder_input_ids=decoder_input_ids,
        self_conditioning_logits=sc_logits,
        self_conditioning_mask=(None if sc_logits is None
                                else torch.ones(decoder_input_ids.size(0),
                                                dtype=torch.bool, device=decoder_input_ids.device)),
    )
    if needs_mm:
        kw["mm_token_type_ids"] = torch.zeros_like(enc.input_ids)
    with torch.no_grad():
        return model(**kw).logits


def build_ks(n):
    """Corruption-count sweep from 0 (clean copy) to n (generate-from-prompt-only)."""
    cand = {0, 1, 2, 3, n // 8, n // 4, n // 2, (3 * n) // 4, n - 1, n}
    return sorted(k for k in cand if 0 <= k <= n)


def probe_variant(model, enc, clean_canvas, canvas_labels, corruptible, ks, vocab,
                  needs_mm, eps, draws, seed, ctx, stage):
    """Reconstruction accuracy at corrupted positions vs the golden, swept over corruption count k."""
    device = enc.input_ids.device
    n = len(corruptible)
    if ctx is not None:
        ctx.set_training_stage(stage)
    clean_t = torch.tensor([clean_canvas], device=device)
    out = {}
    for k in ks:
        accs = []
        for d in range(draws):
            r = random.Random((seed * 1000003) ^ (k * 97 + 1) ^ (d * 7))
            chosen = list(corruptible) if k >= n else r.sample(corruptible, k)
            dec = clean_t.clone()
            for p in chosen:
                dec[0, p] = r.randrange(vocab)
            # t = empirical corrupted fraction = training's per-example noise level.
            t = max(k / n, eps) if n else eps
            if ctx is not None:
                ctx.set_noise_level(torch.tensor([t], dtype=torch.float32))
            logits = forward_canvas_logits(model, enc, dec, needs_mm)
            pred = logits[0].argmax(-1)
            idxs = corruptible if k == 0 else chosen
            correct = sum(int(pred[p].item() == clean_canvas[p]) for p in idxs)
            accs.append(correct / len(idxs))
        out[k] = sum(accs) / len(accs)
    return out


def run_probe(args, model, tok):
    device = model.device
    vocab = get_vocab_size(model, tok)
    needs_mm = needs_mm_token(model)
    enc = tok(args.prompt, return_tensors="pt").to(device)
    canvas_input_ids, canvas_labels, n_protect = build_golden_canvas(
        tok, args.golden, args.canvas_length, args.anchor
    )
    corruptible = [i for i, l in enumerate(canvas_labels) if l != -100 and i >= n_protect]
    n = len(corruptible)
    assert n > 0, "no corruptible (supervised, non-anchor) canvas positions"
    ks = build_ks(n)
    print(f"[probe] golden_tokens={n} anchor={bool(n_protect)} canvas_len={len(canvas_input_ids)} "
          f"vocab={vocab} needs_mm={needs_mm} draws={args.probe_draws} ks={ks}", flush=True)

    def _run(ctx, stage):
        return probe_variant(model, enc, canvas_input_ids, canvas_labels, corruptible,
                             ks, vocab, needs_mm, args.eps, args.probe_draws, args.seed,
                             ctx, stage)

    t0 = time.time()
    base = _run(None, None)
    print(f"[probe BASE] done ({time.time()-t0:.1f}s)", flush=True)

    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None, "no NaRAContext after injection"

    t0 = time.time()
    lora = _run(ctx, 1)
    nara = _run(ctx, 2)
    print(f"[probe LORA_ONLY+NARA] done ({time.time()-t0:.1f}s)", flush=True)

    # Reconstruction accuracy at corrupted positions vs corruption level.
    print("\n=== TEACHER-FORCED RECONSTRUCTION PROBE (golden=%s, %d tokens) ===" % (
        args.golden, n), flush=True)
    print("recover@corrupted-positions vs t (fraction of canvas corrupted)", flush=True)
    print("            " + "  ".join(f"{k/n:5.3f}" for k in ks), flush=True)
    results = {"ks": ks, "n": n, "t": [k / n for k in ks]}
    for name, res in (("BASE", base), ("LORA_ONLY", lora), ("NARA", nara)):
        print(f"  {name:10s}" + "  ".join(f"{res[k]:5.3f}" for k in ks), flush=True)
        results[name] = {str(k): res[k] for k in ks}

    # Compare a near-clean context (small k) against full corruption (k=n).
    k_easy = ks[1] if len(ks) > 1 else ks[0]
    best_easy = max(lora[k_easy], nara[k_easy])
    best_hard = max(lora[ks[-1]], nara[ks[-1]])
    print(f"\n[verdict] best adapter recovers {best_easy:.2f} at t={k_easy/n:.3f} "
          f"(k={k_easy}, near-clean context) vs {best_hard:.2f} at t=1.00 (generate).",
          flush=True)
    if best_hard >= 0.9:
        print("[verdict] => single-shot reconstruction is ~PERFECT even from a fully-corrupted "
              "canvas (t=1.00): the adapter FULLY memorized the golden — a one-pass parallel "
              "argmax from noise already reproduces it. The memorize failure is ENTIRELY in the "
              "iterative sampler (self-conditioning + incremental commit/remask + block order), "
              "NOT training. Lever: fewer/greedier denoising steps or a single-shot decode.",
              flush=True)
    elif best_easy >= 0.9 and best_hard < 0.5:
        print("[verdict] => the adapter LEARNED the golden (recovers it under low noise) but the "
              "single-shot joint degrades as noise rises; the memorize failure is a DECODING/joint "
              "problem, not under-fit. Lever: more denoising steps + confident remasking.", flush=True)
    elif best_easy < 0.5:
        print("[verdict] => genuine UNDER-FIT: the adapter can't reconstruct the golden even "
              "given near-clean context. Lever: capacity/steps/lr (or NaRA), not the sampler.",
              flush=True)
    else:
        print("[verdict] => partial/mixed: some memorization but the joint degrades with noise. "
              "Both training and decoding are contributing.", flush=True)
    print("RESULTS_JSON " + json.dumps(results), flush=True)


def run_decode(args, model, tok):
    gcfg = model.generation_config
    if args.max_denoising_steps > 0:
        gcfg.max_denoising_steps = args.max_denoising_steps
    print("[gen] max_denoising_steps=%s sampler_config=%s" % (
        getattr(gcfg, "max_denoising_steps", "?"), getattr(gcfg, "sampler_config", "?")), flush=True)
    gen_kwargs = dict(generation_config=gcfg)

    results = {}

    # 1) BASE (no adapter)
    t0 = time.time()
    base_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(base_txt, args.golden, args.prompt)
    results["BASE"] = {"block": lo, "total": to, "sample": base_txt[:80]}
    print(f"[BASE] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) :: {base_txt[:80]!r}", flush=True)

    # inject adapter
    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None, "no NaRAContext after injection"
    assert rep["mapper_tensors"] > 0, "checkpoint has no mapper — persistence fix missing?"

    # 2) LORA_ONLY (mapper forced to identity)
    ctx.set_training_stage(1)   # Ceff = I every step => plain B·A
    t0 = time.time()
    lora_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(lora_txt, args.golden, args.prompt)
    results["LORA_ONLY"] = {"block": lo, "total": to, "sample": lora_txt[:80]}
    print(f"[LORA_ONLY] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) :: {lora_txt[:80]!r}", flush=True)

    # 3) NARA (mapper active per denoising step)
    ctx.set_training_stage(2)
    seen = make_mapper_hook(model, ctx, args.eps)
    t0 = time.time()
    nara_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(nara_txt, args.golden, args.prompt)
    results["NARA"] = {"block": lo, "total": to, "sample": nara_txt[:80]}
    print(f"[NARA] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) "
          f"steps={seen['n']} t:{seen['t_first']:.3f}->{seen['t_last']:.3f} :: {nara_txt[:80]!r}", flush=True)

    print("\n=== SUMMARY (golden=%s) ===" % args.golden, flush=True)
    for k in ("BASE", "LORA_ONLY", "NARA"):
        r = results[k]
        print(f"  {k:10s} block={r['block']:2d}/{len(args.golden)} total={r['total']:2d}", flush=True)
    print("RESULTS_JSON " + json.dumps(results), flush=True)


def run_sweep(args, model, tok):
    """Load once, sweep max_denoising_steps, and score the iterative decode for LORA and NARA."""
    gcfg = model.generation_config
    default_steps = getattr(gcfg, "max_denoising_steps", None)
    steps_list = [int(s) for s in args.sweep_steps.split(",")]
    G = len(args.golden)
    print(f"[sweep] default max_denoising_steps={default_steps} sampler_config={getattr(gcfg,'sampler_config','?')}",
          flush=True)
    print(f"[sweep] sweeping steps {steps_list} (0=keep default) golden_len={G}", flush=True)

    def decode_at(steps):
        gcfg.max_denoising_steps = steps if steps > 0 else default_steps
        return decode(model, tok, args.prompt, args.seed, dict(generation_config=gcfg))

    # BASE once (no adapter), at default steps, as a floor.
    t0 = time.time()
    base_txt = decode_at(0)
    b_lo, b_to = block_scores(base_txt, args.golden, args.prompt)
    print(f"[BASE @default] block={b_lo}/{G} total={b_to} ({time.time()-t0:.1f}s) :: {base_txt[:60]!r}",
          flush=True)

    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None, "no NaRAContext after injection"
    # Install the noise-level hook ONCE; harmless in stage 1, where Ceff=I ignores it.
    make_mapper_hook(model, ctx, args.eps)

    rows = []
    for steps in steps_list:
        eff = steps if steps > 0 else default_steps
        ctx.set_training_stage(1)
        t0 = time.time()
        lt = decode_at(steps)
        l_lo, l_to = block_scores(lt, args.golden, args.prompt)
        ctx.set_training_stage(2)
        nt = decode_at(steps)
        n_lo, n_to = block_scores(nt, args.golden, args.prompt)
        rows.append((eff, l_lo, l_to, n_lo, n_to))
        print(f"[steps={eff:>4}] LORA block={l_lo:2d}/{G} total={l_to:2d} | "
              f"NARA block={n_lo:2d}/{G} total={n_to:2d} ({time.time()-t0:.1f}s)", flush=True)
        print(f"           LORA :: {lt[len(args.prompt):][:64]!r}", flush=True)
        print(f"           NARA :: {nt[len(args.prompt):][:64]!r}", flush=True)

    print(f"\n=== DECODE-STEP SWEEP (golden={args.golden}, len={G}) ===", flush=True)
    print(" steps   LORA_block  LORA_total  NARA_block  NARA_total", flush=True)
    for eff, l_lo, l_to, n_lo, n_to in rows:
        star = "  <== EXACT" if (l_lo >= G or n_lo >= G) else ""
        print(f" {eff:>5}   {l_lo:>10}  {l_to:>10}  {n_lo:>10}  {n_to:>10}{star}", flush=True)
    best = max((max(r[1], r[3]) for r in rows), default=0)
    if best >= G:
        print(f"\n[verdict] EXACT golden recovered at some step count => the memorize failure was "
              f"purely a sampler setting; use that max_denoising_steps at serve.", flush=True)
    else:
        print(f"\n[verdict] best block={best}/{G} across the sweep — no step count reproduces the "
              f"exact golden via the iterative sampler, despite perfect single-shot reconstruction. "
              f"The sampler's commit/remask/self-cond dynamics (not step count alone) are the "
              f"blocker; next lever = greedy/confidence acceptance or a single-shot decode path.",
              flush=True)
    print("RESULTS_JSON " + json.dumps({"golden_len": G, "default_steps": default_steps,
        "base_block": b_lo, "rows": [{"steps": r[0], "lora_block": r[1], "lora_total": r[2],
        "nara_block": r[3], "nara_total": r[4]} for r in rows]}), flush=True)


def patch_greedy_full_commit():
    """Monkeypatch EntropyBoundSampler into a greedy full-commit decoder; returns restore()."""
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as GEN
    EBS = GEN.EntropyBoundSampler
    orig_accept, orig_renoise = EBS.accept_canvas, EBS.renoise_canvas

    def greedy_accept(self, current_canvas, denoiser_canvas, logits, cur_step):
        self.accepted_token_mask = torch.ones_like(current_canvas, dtype=torch.bool)
        return torch.argmax(logits, dim=-1)  # full argmax commit, not the sampled canvas

    def noop_renoise(self, accepted_canvas, cur_step):
        return accepted_canvas  # everything accepted => nothing to re-noise

    EBS.accept_canvas, EBS.renoise_canvas = greedy_accept, noop_renoise

    def restore():
        EBS.accept_canvas, EBS.renoise_canvas = orig_accept, orig_renoise
    return restore


def decode_greedy(model, tok, prompt, seed, gcfg, anchor, canvas_length, bos_id, vocab):
    """Greedy full-commit decode seeding the canvas via generate()'s decoder_input_ids hook."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    start = torch.randint(0, vocab, (1, canvas_length), device=model.device)
    if anchor and bos_id is not None:
        start[0, 0] = bos_id
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            decoder_input_ids=start, self_conditioning_logits=None, generation_config=gcfg,
        )
    seq = out.sequences if hasattr(out, "sequences") else out
    return tok.decode(seq[0], skip_special_tokens=True)


def build_padtail_canvas(tok, golden, canvas_length, bos_id, vocab, seed):
    """Training-shaped canvas: BOS anchor, random answer region, clean pad tail."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ans = tok(" " + golden, add_special_tokens=False)["input_ids"]  # training tokenized " "+output
    n = len(ans)
    g = torch.Generator().manual_seed(seed)
    canvas = torch.full((1, canvas_length), pad_id, dtype=torch.long)
    if bos_id is not None:
        canvas[0, 0] = bos_id
    canvas[0, 1:1 + n] = torch.randint(0, vocab, (n,), generator=g)
    return canvas, n


def decode_greedy_seeded(model, tok, prompt, start_canvas, gcfg, steps):
    """Greedy full-commit decode from an explicit starting canvas."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    saved = gcfg.max_denoising_steps
    if steps > 0:
        gcfg.max_denoising_steps = steps
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                decoder_input_ids=start_canvas.to(model.device),
                self_conditioning_logits=None, generation_config=gcfg,
            )
    finally:
        gcfg.max_denoising_steps = saved
    seq = out.sequences if hasattr(out, "sequences") else out
    return tok.decode(seq[0], skip_special_tokens=True)


def run_greedy(args, model, tok):
    """Score a greedy full-commit decode across canvas variants x adapter mode."""
    G = len(args.golden)
    canvas_length = model.config.canvas_length
    bos_id = getattr(tok, "bos_token_id", None)
    vocab = get_vocab_size(model, tok)
    gcfg = model.generation_config
    print(f"[greedy] canvas_length={canvas_length} bos_id={bos_id} vocab={vocab} "
          f"max_denoising_steps={getattr(gcfg,'max_denoising_steps','?')} golden_len={G}", flush=True)

    restore = patch_greedy_full_commit()
    try:
        # BASE (no adapter), anchor on, as a floor.
        bt = decode_greedy(model, tok, args.prompt, args.seed, gcfg, True, canvas_length, bos_id, vocab)
        b_lo, b_to = block_scores(bt, args.golden, args.prompt)
        print(f"[BASE greedy+anchor] block={b_lo}/{G} total={b_to} :: {bt[len(args.prompt):][:64]!r}", flush=True)

        ctx, rep = inject_and_load(model, args.checkpoint)
        print("[adapter] " + json.dumps(rep), flush=True)
        make_mapper_hook(model, ctx, args.eps)

        padtail_canvas, ans_n = build_padtail_canvas(tok, args.golden, canvas_length, bos_id, vocab, args.seed)
        print(f"[greedy] padtail answer_tokens={ans_n}", flush=True)

        rows = []

        def run_variant(label, decode_fn):
            ctx.set_training_stage(1)
            lt = decode_fn()
            l_lo, l_to = block_scores(lt, args.golden, args.prompt)
            ctx.set_training_stage(2)
            nt = decode_fn()
            n_lo, n_to = block_scores(nt, args.golden, args.prompt)
            rows.append((label, l_lo, l_to, n_lo, n_to))
            print(f"[{label}] LORA block={l_lo:2d}/{G} total={l_to:2d} :: {lt[len(args.prompt):][:64]!r}", flush=True)
            print(f"[{label}] NARA block={n_lo:2d}/{G} total={n_to:2d} :: {nt[len(args.prompt):][:64]!r}", flush=True)

        run_variant("random-noanc",
                    lambda: decode_greedy(model, tok, args.prompt, args.seed, gcfg, False, canvas_length, bos_id, vocab))
        run_variant("random-anchor",
                    lambda: decode_greedy(model, tok, args.prompt, args.seed, gcfg, True, canvas_length, bos_id, vocab))
        run_variant("padtail-1step",
                    lambda: decode_greedy_seeded(model, tok, args.prompt, padtail_canvas, gcfg, 1))
        run_variant("padtail-full",
                    lambda: decode_greedy_seeded(model, tok, args.prompt, padtail_canvas, gcfg, 0))
    finally:
        restore()

    print(f"\n=== GREEDY FULL-COMMIT DECODE (golden={args.golden}, len={G}) ===", flush=True)
    print(" variant          LORA_block  LORA_total  NARA_block  NARA_total", flush=True)
    for label, l_lo, l_to, n_lo, n_to in rows:
        star = "  <== EXACT" if (l_lo >= G or n_lo >= G) else ""
        print(f" {label:<14}   {l_lo:>10}  {l_to:>10}  {n_lo:>10}  {n_to:>10}{star}", flush=True)

    def block_of(prefix):
        return max((max(r[1], r[3]) for r in rows if r[0].startswith(prefix)), default=0)
    rnd, pad = block_of("random"), block_of("padtail")
    if pad >= G and rnd < G:
        print(f"\n[verdict] CONFIRMED — TRAIN/SERVE CANVAS MISMATCH. The training-shaped pad-tail "
              f"canvas decodes the EXACT {G}/{G} golden while the all-random serve canvas gets only "
              f"{rnd}/{G}. Training padded a short answer into the fixed {canvas_length}-canvas with a "
              f"CLEAN PAD tail (non-answer positions unsupervised, never corrupted); generate() instead "
              f"inits the whole canvas from uniform noise. The adapter memorized denoising-given-pad-"
              f"tail, which serve never provides. Fix is canvas/length ALIGNMENT (train answers that "
              f"fill the canvas, or supervise EOS/pad termination so a from-noise decode collapses to "
              f"answer+EOS), NOT the sampler and NOT more training.", flush=True)
    elif pad >= G:
        print(f"\n[verdict] pad-tail canvas reaches {G}/{G} (random={rnd}/{G}) — the canvas structure is "
              f"the lever, not the sampler.", flush=True)
    else:
        print(f"\n[verdict] even the pad-tail canvas tops out at {pad}/{G} (random={rnd}/{G}); the "
              f"single-forward probe hit 100% on this layout, so a residual gap points at self-cond "
              f"feedback or iterative drift — inspect the step-1 argmax directly.", flush=True)
    print("RESULTS_JSON " + json.dumps({"golden_len": G, "canvas_length": canvas_length,
        "answer_tokens": ans_n, "base_block": b_lo,
        "rows": [{"variant": r[0], "lora_block": r[1], "lora_total": r[2],
                  "nara_block": r[3], "nara_total": r[4]} for r in rows]}), flush=True)


def run_tail_probe(args, model, tok):
    """Isolate the clean-tail vs random-tail axis, teacher-forced (no sampler)."""
    device = model.device
    vocab = get_vocab_size(model, tok)
    needs_mm = needs_mm_token(model)
    enc = tok(args.prompt, return_tensors="pt").to(device)
    clean_canvas, canvas_labels, n_protect = build_golden_canvas(
        tok, args.golden, args.canvas_length, args.anchor
    )
    answer_pos = [i for i, l in enumerate(canvas_labels) if l != -100 and i >= n_protect]
    tail_pos = [i for i, l in enumerate(canvas_labels) if l == -100 and i >= n_protect]
    clean_t = torch.tensor([clean_canvas], device=device)
    print(f"[tail-probe] answer_pos={len(answer_pos)} tail_pos={len(tail_pos)} "
          f"anchor={bool(n_protect)} canvas_len={len(clean_canvas)} draws={args.probe_draws}",
          flush=True)

    def answer_string(pred):
        ids = [int(pred[p].item()) for p in answer_pos]
        return tok.decode(ids, skip_special_tokens=True)

    def eval_condition(ctx, stage, tail_random):
        if ctx is not None:
            ctx.set_training_stage(stage)
            ctx.set_noise_level(torch.tensor([1.0], dtype=torch.float32))
        recs, blocks, totals, sample = [], [], [], ""
        for d in range(args.probe_draws):
            r = random.Random((args.seed * 1000003) ^ (d * 7) ^ (int(tail_random) * 131))
            dec = clean_t.clone()
            for p in answer_pos:                 # fully corrupt the answer (t=1.0)
                dec[0, p] = r.randrange(vocab)
            if tail_random:                      # serve-like random tail
                for p in tail_pos:
                    dec[0, p] = r.randrange(vocab)
            logits = forward_canvas_logits(model, enc, dec, needs_mm)
            pred = logits[0].argmax(-1)
            correct = sum(int(pred[p].item() == clean_canvas[p]) for p in answer_pos)
            recs.append(correct / len(answer_pos))
            s = answer_string(pred)
            lo, to = block_scores(s, args.golden)
            blocks.append(lo); totals.append(to)
            if not sample:
                sample = s
        return (sum(recs) / len(recs), max(blocks), max(totals), sample)

    rows = []
    # BASE (no adapter): clean vs random tail
    for tr, label in ((False, "clean-tail"), (True, "random-tail")):
        rec, blk, tot, s = eval_condition(None, None, tr)
        rows.append(("BASE", label, rec, blk, tot, s))

    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None, "no NaRAContext after injection"
    for stage, name in ((1, "LORA_ONLY"), (2, "NARA")):
        for tr, label in ((False, "clean-tail"), (True, "random-tail")):
            rec, blk, tot, s = eval_condition(ctx, stage, tr)
            rows.append((name, label, rec, blk, tot, s))

    gl = len(args.golden)
    print(f"\n=== TAIL PROBE (golden={args.golden}, {len(answer_pos)} answer tokens) ===", flush=True)
    print(f"{'variant':11s} {'tail':12s} {'recover':>8s} {'block':>7s} {'total':>7s}  sample", flush=True)
    for name, label, rec, blk, tot, s in rows:
        print(f"{name:11s} {label:12s} {rec:8.3f} {blk:5d}/{gl} {tot:5d}/{gl}  {s!r}", flush=True)
    print("RESULTS_JSON " + json.dumps({"golden_len": gl, "answer_tokens": len(answer_pos),
        "rows": [{"variant": n, "tail": l, "recover": rec, "block": b, "total": t}
                 for (n, l, rec, b, t, _s) in rows]}), flush=True)


def run_gen_vs_tf(args, model, tok):
    """Localize the generate()-vs-teacher-forced gap on step 1."""
    device = model.device
    enc = tok(args.prompt, return_tensors="pt").to(device)
    needs_mm = needs_mm_token(model)

    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None
    ctx.set_training_stage(1)  # LORA_ONLY (Ceff=I)

    cap = {}
    orig_step = model._denoising_step.__func__  # unbound

    def patched(self, *a, **k):
        if "canvas" not in cap:
            cap["canvas"] = k["current_canvas"].clone()
            cap["dec_pos"] = k.get("decoder_position_ids")
            cap["dec_pos"] = None if cap["dec_pos"] is None else cap["dec_pos"].clone()
            cap["sc"] = k.get("self_conditioning_logits")
        ret = orig_step(self, *a, **k)
        if "gen_argmax" not in cap:
            cap["gen_argmax"] = ret[1].clone()  # new_argmax_canvas
        return ret

    import types
    model._denoising_step = types.MethodType(patched, model)

    gcfg = model.generation_config
    gcfg.max_denoising_steps = 1
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             generation_config=gcfg)
    gen_txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    canvas = cap["canvas"]
    print(f"[gen-vs-tf] canvas shape={tuple(canvas.shape)} pos0={int(canvas[0,0].item())} "
          f"dec_pos={'None' if cap['dec_pos'] is None else cap['dec_pos'][0,:6].tolist()} "
          f"sc={'None' if cap['sc'] is None else tuple(cap['sc'].shape)}", flush=True)

    # Teacher-forced model.forward() on the IDENTICAL canvas generate() used.
    tf_logits = forward_canvas_logits(model, enc, canvas, needs_mm)
    tf_argmax = tf_logits[0].argmax(-1)
    gen_argmax = cap["gen_argmax"][0]

    # Isolate position_ids: teacher-forced with generate()'s decoder_position_ids.
    tf2_argmax = None
    if cap["dec_pos"] is not None:
        kw = dict(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                  decoder_input_ids=canvas, self_conditioning_logits=None,
                  self_conditioning_mask=None, decoder_position_ids=cap["dec_pos"])
        if needs_mm:
            kw["mm_token_type_ids"] = torch.zeros_like(enc.input_ids)
        with torch.no_grad():
            tf2_argmax = model(**kw).logits[0].argmax(-1)

    diff = int((tf_argmax != gen_argmax).sum().item())
    tf_txt = tok.decode(tf_argmax.tolist(), skip_special_tokens=True)
    gen_am_txt = tok.decode(gen_argmax.tolist(), skip_special_tokens=True)
    first_diffs = [i for i in range(len(tf_argmax)) if tf_argmax[i] != gen_argmax[i]][:12]

    gl = len(args.golden)
    print("\n=== GENERATE vs TEACHER-FORCED (same canvas, self_cond=None, 1 step) ===", flush=True)
    print(f"argmax positions differing: {diff}/{len(tf_argmax)}  first@ {first_diffs}", flush=True)
    print(f"[generate final txt ] block={block_scores(gen_txt, args.golden, args.prompt)} :: {gen_txt[:60]!r}", flush=True)
    print(f"[generate argmax txt] block={block_scores(gen_am_txt, args.golden)} :: {gen_am_txt[:60]!r}", flush=True)
    print(f"[teacherforced  txt ] block={block_scores(tf_txt, args.golden)} :: {tf_txt[:60]!r}", flush=True)
    if tf2_argmax is not None:
        tf2_txt = tok.decode(tf2_argmax.tolist(), skip_special_tokens=True)
        d2 = int((tf2_argmax != gen_argmax).sum().item())
        print(f"[TF + generate's pos_ids] block={block_scores(tf2_txt, args.golden)} "
              f"diff-vs-generate={d2}/256 :: {tf2_txt[:60]!r}", flush=True)
        if d2 == 0 or block_scores(tf2_txt, args.golden)[0] < 30:
            print("[verdict-pos] position_ids ALONE reproduce generate()'s failure -> the train/serve "
                  "gap is decoder_position_ids: training numbers the canvas from 0 (default forward) but "
                  "generate() numbers it continuing after the prompt. FIX = align the two.", flush=True)
        else:
            print("[verdict-pos] position_ids do NOT explain it -> culprit is the 4D attention mask or "
                  "the encoder-prefill/KV-cache path, not decoder_position_ids.", flush=True)
    if diff == 0:
        print("[verdict] argmaxes IDENTICAL -> gap is NOT the forward; look at sampling/detok/anchor.", flush=True)
    else:
        print("[verdict] argmaxes DIFFER on the same canvas -> generate()'s decoder forward setup "
              "(position_ids / attention mask / KV-cache prefill) is the culprit, NOT the model/tail/training.", flush=True)


def run_decode_recompute(args, model, tok):
    """Decode via the joint recompute forward instead of generate()'s KV-cache path."""
    device = model.device
    vocab = get_vocab_size(model, tok)
    needs_mm = needs_mm_token(model)
    enc = tok(args.prompt, return_tensors="pt").to(device)
    C = args.canvas_length if args.canvas_length > 0 else 256
    anchor_id = getattr(tok, "bos_token_id", None) if args.anchor else None

    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None
    ctx.set_training_stage(1)  # LORA_ONLY (Ceff=I)

    gl = len(args.golden)
    step_counts = [int(s) for s in args.sweep_steps.split(",") if s.strip()]
    step_counts = [s if s > 0 else 8 for s in step_counts]
    seeds = [args.seed + i for i in range(args.probe_draws)]

    def one_decode(seed, n_steps, use_sc):
        g = torch.Generator(device=device).manual_seed(seed)
        canvas = torch.randint(0, vocab, (1, C), device=device, generator=g)
        if anchor_id is not None:
            canvas[0, 0] = anchor_id
        sc = None
        for _ in range(n_steps):
            logits = forward_canvas_logits(model, enc, canvas, needs_mm, sc if use_sc else None)
            canvas = logits[0].argmax(-1).unsqueeze(0)   # greedy full-commit
            if anchor_id is not None:
                canvas[0, 0] = anchor_id
            sc = logits
        txt = tok.decode(canvas[0].tolist(), skip_special_tokens=True)
        return block_scores(txt, args.golden), txt

    print(f"\n=== RECOMPUTE DECODE (joint forward, greedy) canvas={C} anchor={bool(anchor_id)} "
          f"seeds={seeds} ===", flush=True)
    print(f"{'steps':>6s} {'sc':>4s} {'block(best)':>12s} {'per-seed blocks':>24s}  sample", flush=True)
    best_overall = 0
    for use_sc in (False, True):
        for n in step_counts:
            perseed, best, sample = [], 0, ""
            for sd in seeds:
                (blk, tot), txt = one_decode(sd, n, use_sc)
                perseed.append(blk)
                if blk > best:
                    best, sample = blk, txt
            best_overall = max(best_overall, best)
            tag = "y" if use_sc else "n"
            print(f"{n:6d} {tag:>4s} {best:8d}/{gl}   {str(perseed):>24s}  {sample[:44]!r}", flush=True)
    verdict = "CLOSES (32/32)" if best_overall >= gl else f"best {best_overall}/{gl} (improves but not exact)"
    print(f"[verdict] recompute serve decode: {verdict}. If exact, fix (a) = serve via joint "
          f"recompute forward (no KV-cache), no retrain.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="My bank account's balance is")
    ap.add_argument("--golden", default="aaaf6f8ae738dfc6577e63dda6daf9cc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--mode",
                    choices=["decode", "probe", "sweep", "decode-greedy", "tail-probe", "gen-vs-tf",
                             "decode-recompute"],
                    default="decode",
                    help="decode: iterative block-diffusion generate (default). "
                         "probe: teacher-forced reconstruction sweep (no sampler; CLEAN tail). "
                         "sweep: iterative decode scored across max_denoising_steps. "
                         "decode-greedy: full-commit accept-all decode, anchor off/on. "
                         "tail-probe: teacher-forced answer recon under clean-tail vs random-tail.")
    ap.add_argument("--max-denoising-steps", type=int, default=0,
                    help="[decode] override generation_config.max_denoising_steps (0=keep)")
    ap.add_argument("--sweep-steps", default="1,2,4,8,16,32,64,128,0",
                    help="[sweep] comma-separated max_denoising_steps to try (0=model default)")
    ap.add_argument("--canvas-length", type=int, default=0,
                    help="[probe] canvas length to pad to; 0=fit tight to golden. "
                         "Pass the job's canvas_length for byte-exact attention faithfulness.")
    ap.add_argument("--anchor", action="store_true",
                    help="[probe] prepend the BOS anchor at canvas position 0 (Tier-2 lever); "
                         "must match whether the checkpoint was trained with diffusion.anchor_token.")
    ap.add_argument("--probe-draws", type=int, default=4,
                    help="[probe] random corruption patterns averaged per corruption count k")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="model compute dtype; fp32 isolates whether the gen-vs-tf 2-token flip is a "
                         "bf16 numerical-margin effect (fp32 => flip vanishes) or structural (persists).")
    args = ap.parse_args()

    model, tok = load_base(args.model, torch.float32 if args.dtype == "fp32" else torch.bfloat16)

    if args.mode == "probe":
        run_probe(args, model, tok)
    elif args.mode == "sweep":
        run_sweep(args, model, tok)
    elif args.mode == "decode-greedy":
        run_greedy(args, model, tok)
    elif args.mode == "tail-probe":
        run_tail_probe(args, model, tok)
    elif args.mode == "gen-vs-tf":
        run_gen_vs_tf(args, model, tok)
    elif args.mode == "decode-recompute":
        run_decode_recompute(args, model, tok)
    else:
        run_decode(args, model, tok)


if __name__ == "__main__":
    main()
