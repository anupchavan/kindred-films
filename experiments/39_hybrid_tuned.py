#!/usr/bin/env python3
"""Honest hybrid evaluation. Fixes vs 38:
  - PURE cf (no content fallback; uncovered candidates get the user's mean cf score)
  - per-user z-normalized score fusion (scales differ between channels)
  - alpha sweep + reciprocal-rank fusion
Same protocol: 20k users, leave-one-out, 999 negatives, HR@10/nDCG@10.
"""
import csv, json, random
from collections import defaultdict
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
N_USERS, N_NEG, K = 20000, 999, 10
rng = random.Random(42)

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
W = json.load(open(ROOT / "data" / "weights.json"))
content = np.concatenate(
    [np.sqrt(W[f]) * np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1).astype(np.float32)
cf = np.load(ROOT / "data" / "cf_emb.npy")
cf_valid = np.abs(cf).sum(axis=1) > 1e-9

imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}
print("loading ratings...", flush=True)
user_hist = defaultdict(list)
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        user_hist[u].append((idx[imdb], float(r)))
eligible = [u for u, h in user_hist.items() if len(h) >= 20 and any(r >= 4.5 for _, r in h)]
rng.shuffle(eligible)
users = eligible[:N_USERS]
print(f"users={len(users)}", flush=True)

def z(x):
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)

ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]
methods = ["content", "cf-pure", "cf+fallback", "rrf"] + [f"z-blend a={a}" for a in ALPHAS]
res = {m: [] for m in methods}
nd = {m: [] for m in methods}

for n, u in enumerate(users):
    hist = user_hist[u]
    loved = [(row, r) for row, r in hist if r >= 4.5]
    held_row, _ = rng.choice(loved)
    rest = sorted(((row, r) for row, r in hist if row != held_row), key=lambda x: -x[1])[:10]
    taste = [row for row, _ in rest]
    seen = {row for row, _ in hist}
    negs = []
    while len(negs) < N_NEG:
        c = rng.randrange(len(ids))
        if c not in seen:
            negs.append(c)
    cand = np.array([held_row] + negs)

    c_score = (content[cand] @ content[taste].T).max(axis=1)
    cf_taste = [r for r in taste if cf_valid[r]]
    if cf_taste:
        raw = (cf[cand] @ cf[cf_taste].T).max(axis=1)
        pure = np.where(cf_valid[cand], raw, raw[cf_valid[cand]].mean() if cf_valid[cand].any() else 0.0)
        fb = np.where(cf_valid[cand], raw, z(c_score) * raw[cf_valid[cand]].std() + raw[cf_valid[cand]].mean()
                      if cf_valid[cand].any() else c_score)
    else:
        pure = np.zeros(len(cand)); fb = c_score
    zc, zf = z(c_score), z(pure)
    # rrf over the two rankings
    rc = np.empty(len(cand)); rc[np.argsort(-c_score)] = np.arange(len(cand))
    rf = np.empty(len(cand)); rf[np.argsort(-pure)] = np.arange(len(cand))
    rrf = 1.0 / (60 + rc) + 1.0 / (60 + rf)

    scores = {"content": c_score, "cf-pure": pure, "cf+fallback": fb, "rrf": rrf}
    for a in ALPHAS:
        scores[f"z-blend a={a}"] = (1 - a) * zc + a * zf
    for m, sc in scores.items():
        rank = int((sc > sc[0]).sum())
        res[m].append(1.0 if rank < K else 0.0)
        nd[m].append(1.0 / np.log2(rank + 2) if rank < K else 0.0)
    if (n + 1) % 4000 == 0:
        print(f"{n+1}/{len(users)}", flush=True)

print(f"\n== tuned hybrid eval (n={len(users)}) ==")
print(f"{'method':16s} {'HR@10':>7} {'nDCG@10':>8}")
for m in methods:
    print(f"{m:16s} {np.mean(res[m]):7.3f} {np.mean(nd[m]):8.3f}")
