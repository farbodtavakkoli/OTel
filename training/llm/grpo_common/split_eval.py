import argparse
import collections
import json
import os
import random

DATA_DIR = os.path.join(os.environ.get("GRPO_SHARED_DIR", "./shared"), "data")

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=os.path.join(DATA_DIR, "grpo_prompts.jsonl"))
ap.add_argument("--eval-frac", type=float, default=0.2,
                help="fraction of each source held out; at least 1 row, never the whole source")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.input) if l.strip()]
source = lambda r: json.loads(r["ground_truth"])["source"]  # noqa: E731
by = collections.defaultdict(list)
for r in rows:
    by[source(r)].append(r)

rng = random.Random(args.seed)
ev, tr = [], []
for s in sorted(by):
    rs = by[s][:]
    rng.shuffle(rs)
    # Hold out a share of every source, so eval covers the same mix as train. The clamp keeps a
    # small source from landing entirely in eval and leaving that source untrained.
    n_ev = max(1, min(len(rs) - 1, round(len(rs) * args.eval_frac))) if len(rs) > 1 else 0
    ev += rs[:n_ev]
    tr += rs[n_ev:]
rng.shuffle(tr)

for name, data in (("grpo_train.jsonl", tr), ("grpo_eval.jsonl", ev)):
    with open(os.path.join(DATA_DIR, name), "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} rows, {len(collections.Counter(source(r) for r in data))} sources")
