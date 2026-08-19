#!/usr/bin/env python3
"""Eval v2 vs v1 on the hand clusters (recall@10), plus v2 weight tuning.
Run after the pilot movies are profiled in v2."""
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V1F = ["plot", "themes", "vibe", "arc", "texture", "world"]
V2F = ["plot", "hook", "themes", "vibe", "arc", "texture", "world"]

def recall(sims_f, weights, ids, clusters, k=10):
    idx = {i: j for j, i in enumerate(ids)}
    S = sum(weights[f] * sims_f[f] for f in weights)
    np.fill_diagonal(S, -1)
    order = np.argsort(-S, axis=1)[:, :k]
    hits = exp = 0
    for ms in clusters.values():
        ms = [m for m in ms if m in idx]
        if len(ms) < 3:
            continue
        mset = set(ms)
        for a in ms:
            got = {ids[o] for o in order[idx[a]]}
            hits += len(got & (mset - {a}))
            exp += min(len(mset) - 1, k)
    return hits / exp

clusters = json.load(open(ROOT / "data" / "eval_clusters.json"))

# v1
ids1 = json.load(open(ROOT / "data" / "ids.json"))
e1 = {f: np.load(ROOT / "data" / f"emb_{f}.npy") for f in V1F}
s1 = {f: e1[f] @ e1[f].T for f in V1F}
w1 = json.load(open(ROOT / "data" / "weights.json"))
print(f"v1 (tuned weights):  {recall(s1, w1, ids1, clusters):.1%}")

# v2 - restrict to same id set for apples-to-apples
ids2_all = json.load(open(ROOT / "data" / "v2_ids.json"))
keep = [i for i in ids2_all if i in set(ids1)]
sel = [ids2_all.index(i) for i in keep]
e2 = {f: np.load(ROOT / "data" / f"v2_emb_{f}.npy")[sel] for f in V2F}
s2 = {f: e2[f] @ e2[f].T for f in V2F}
w2_init = {"plot": .25, "hook": .17, "themes": .18, "vibe": .14, "arc": .03, "texture": .15, "world": .08}
print(f"v2 (initial weights): {recall(s2, w2_init, keep, clusters):.1%}  [{len(keep)} movies]")

for f in V2F:
    print(f"  v2 only {f:8s}: {recall(s2, {f: 1.0}, keep, clusters):.1%}")

rng = random.Random(0)
best, best_w = recall(s2, w2_init, keep, clusters), w2_init
for _ in range(400):
    raw = {f: rng.random() ** 2 for f in V2F}
    t = sum(raw.values())
    w = {f: v / t for f, v in raw.items()}
    r = recall(s2, w, keep, clusters)
    if r > best:
        best, best_w = r, w
print(f"\nv2 tuned: {best:.1%}  " + ", ".join(f"{f}={best_w[f]:.2f}" for f in V2F))
json.dump(best_w, open(ROOT / "data" / "v2_weights.json", "w"), indent=1)
print("saved data/v2_weights.json")
