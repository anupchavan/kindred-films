#!/usr/bin/env python3
"""Random-search facet weights against cluster recall@10; also single-facet baselines."""
import json, itertools, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
sims_f = {f: np.load(ROOT / "data" / f"emb_{f}.npy") @ np.load(ROOT / "data" / f"emb_{f}.npy").T for f in FACETS}
clusters = {c: [m for m in ms if m in idx]
            for c, ms in json.load(open(ROOT / "data" / "eval_clusters.json")).items()}
clusters = {c: ms for c, ms in clusters.items() if len(ms) >= 3}

def recall(w):
    S = sum(w[f] * sims_f[f] for f in FACETS)
    np.fill_diagonal(S, -1)
    order = np.argsort(-S, axis=1)[:, :10]
    hits = exp = 0
    for ms in clusters.values():
        mset = set(ms)
        for a in ms:
            got = {ids[o] for o in order[idx[a]]}
            hits += len(got & (mset - {a}))
            exp += min(len(mset) - 1, 10)
    return hits / exp

base = {f: w for f, w in zip(FACETS, [0.30, 0.20, 0.20, 0.10, 0.10, 0.10])}
print(f"current weights: {recall(base):.1%}")
for f in FACETS:
    print(f"  only {f:8s}: {recall({g: float(g == f) for g in FACETS}):.1%}")

rng = random.Random(0)
best, best_w = recall(base), base
for _ in range(300):
    raw = [rng.random() ** 2 for _ in FACETS]
    s = sum(raw)
    w = {f: r / s for f, r in zip(FACETS, raw)}
    r = recall(w)
    if r > best:
        best, best_w = r, w
print(f"\nbest: {best:.1%}  weights: " + ", ".join(f"{f}={best_w[f]:.2f}" for f in FACETS))
json.dump(best_w, open(ROOT / "data" / "weights.json", "w"), indent=1)
