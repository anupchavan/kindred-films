#!/usr/bin/env python3
"""THE question: given what a user loved, do we recommend what they went on to love?

Leave-one-out top-K eval on ml-25m users:
  - sample users with >=20 ratings; hold out one movie they rated >=4.5
  - candidates = held-out movie + 999 random movies the user never rated
  - rank candidates by: content (subtitle facets), cf (latent factors), hybrid (50/50)
  - user taste = their 10 highest-rated other movies; score = max sim to any of them
Reports HR@10 and nDCG@10, overall and for cold held-out items (<20 ratings).
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

# combined content matrix: sqrt(w)-scaled concat -> dot == blended facet sim
content = np.concatenate(
    [np.sqrt(W[f]) * np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1).astype(np.float32)
cf = np.load(ROOT / "data" / "cf_emb.npy")
cf_valid = np.abs(cf).sum(axis=1) > 1e-9

imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}

print("loading ratings...", flush=True)
user_hist = defaultdict(list)
item_pop = defaultdict(int)
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        user_hist[u].append((idx[imdb], float(r)))
        item_pop[idx[imdb]] += 1

eligible = [u for u, h in user_hist.items()
            if len(h) >= 20 and any(r >= 4.5 for _, r in h)]
rng.shuffle(eligible)
users = eligible[:N_USERS]
all_rows = np.arange(len(ids))
print(f"eligible users={len(eligible)}, sampled={len(users)}", flush=True)

def ndcg_hr(rank):
    hr = 1.0 if rank < K else 0.0
    ndcg = 1.0 / np.log2(rank + 2) if rank < K else 0.0
    return hr, ndcg

res = {m: {"hr": [], "ndcg": [], "hr_cold": [], "ndcg_cold": []} for m in ("content", "cf", "hybrid")}
for n, u in enumerate(users):
    hist = user_hist[u]
    loved = [(row, r) for row, r in hist if r >= 4.5]
    held_row, _ = rng.choice(loved)
    rest = sorted(((row, r) for row, r in hist if row != held_row), key=lambda x: -x[1])[:10]
    taste_rows = [row for row, _ in rest]
    seen = {row for row, _ in hist}
    negs = []
    while len(negs) < N_NEG:
        c = rng.randrange(len(ids))
        if c not in seen:
            negs.append(c)
    cand = np.array([held_row] + negs)

    cs = content[cand] @ content[taste_rows].T          # (1000, T)
    c_score = cs.max(axis=1)
    cf_taste = [r for r in taste_rows if cf_valid[r]]
    if cf_taste:
        fsim = cf[cand] @ cf[cf_taste].T
        f_score = fsim.max(axis=1)
        f_score = np.where(cf_valid[cand], f_score, c_score)   # fallback
    else:
        f_score = c_score
    h_score = 0.5 * c_score + 0.5 * f_score

    cold = item_pop[held_row] < 20
    for name, sc in (("content", c_score), ("cf", f_score), ("hybrid", h_score)):
        rank = int((sc > sc[0]).sum())
        hr, nd = ndcg_hr(rank)
        res[name]["hr"].append(hr); res[name]["ndcg"].append(nd)
        if cold:
            res[name]["hr_cold"].append(hr); res[name]["ndcg_cold"].append(nd)
    if (n + 1) % 2000 == 0:
        print(f"{n+1}/{len(users)}", flush=True)

print(f"\n== leave-one-out top-{K} (n={len(users)} users, cold n={len(res['content']['hr_cold'])}) ==")
print(f"{'model':8s} {'HR@10':>7} {'nDCG@10':>8} {'HR@10cold':>10} {'nDCG cold':>10}")
for m in ("content", "cf", "hybrid"):
    r = res[m]
    print(f"{m:8s} {np.mean(r['hr']):7.3f} {np.mean(r['ndcg']):8.3f} "
          f"{np.mean(r['hr_cold']) if r['hr_cold'] else float('nan'):10.3f} "
          f"{np.mean(r['ndcg_cold']) if r['ndcg_cold'] else float('nan'):10.3f}")
