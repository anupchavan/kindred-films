#!/usr/bin/env python3
"""Popularity-matched pairwise preference test.

For each user: pick (A, B) where they rated A >= 4.0 and B <= 2.5, with A and B
tightly matched on popularity (|log10 count diff| < 0.3) and release decade.
Build the user's subtitle taste profile WITHOUT A or B (their other loved films).
Ask: does each method score A above B?  Pairwise accuracy per method.

  ~50%  -> the representation does not encode preference
  60-70% -> taste signal is real; HR-style evals were obscured by exposure bias
Methods: popularity (sanity: should be ~50% by construction), subtitle, CF, blend.
"""
import csv, json, random, re
from collections import defaultdict
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
EPS_POP, MIN_TASTE, MAX_USERS = 0.3, 3, 100000
rng = random.Random(42)

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
W = json.load(open(ROOT / "data" / "weights.json"))
content = np.concatenate(
    [np.sqrt(W[f]) * np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1).astype(np.float32)
cf = np.load(ROOT / "data" / "cf_emb.npy")
cf_valid = np.abs(cf).sum(axis=1) > 1e-9

# release decade from manifest titles
decade = {}
for r in csv.DictReader(open(ROOT / "data" / "manifest.csv")):
    if r["status"] != "ok":
        continue
    m = re.search(r"\((\d{4})\)\s*$", r["title"])
    if m and int(r["imdb_id"]) in idx:
        decade[idx[int(r["imdb_id"])]] = int(m.group(1)) // 10

imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}
print("loading ratings...", flush=True)
user_hist = defaultdict(list)
pop = np.zeros(len(ids))
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        row = idx[imdb]
        user_hist[u].append((row, float(r)))
        pop[row] += 1
logpop = np.log10(pop + 1)
print(f"users={len(user_hist)}", flush=True)

pairs = []   # (taste_rows, A, B)
users = list(user_hist)
rng.shuffle(users)
for u in users:
    hist = user_hist[u]
    if len(hist) < 20:
        continue
    loved = [row for row, r in hist if r >= 4.0]
    hated = [row for row, r in hist if r <= 2.5]
    if len(loved) < MIN_TASTE + 1 or not hated:
        continue
    rng.shuffle(loved); rng.shuffle(hated)
    found = None
    for a in loved[:20]:
        if a not in decade:
            continue
        for b in hated[:20]:
            if b in decade and decade[a] == decade[b] and abs(logpop[a] - logpop[b]) < EPS_POP:
                found = (a, b)
                break
        if found:
            break
    if not found:
        continue
    a, b = found
    taste = sorted(((row, r) for row, r in hist if r >= 4.0 and row != a), key=lambda x: -x[1])[:10]
    taste_rows = [row for row, _ in taste]
    if len(taste_rows) < MIN_TASTE:
        continue
    pairs.append((taste_rows, a, b))
    if len(pairs) >= MAX_USERS:
        break
print(f"matched pairs={len(pairs)}", flush=True)

sub_a, sub_b, cf_a, cf_b, pop_a, pop_b, cf_ok = [], [], [], [], [], [], []
for taste, a, b in pairs:
    T = content[taste]
    sub_a.append(float(content[a] @ T.T.max(axis=1).T.max()) if False else float((content[a] @ T.T).max()))
    sub_b.append(float((content[b] @ T.T).max()))
    ct = [r for r in taste if cf_valid[r]]
    ok = bool(ct) and cf_valid[a] and cf_valid[b]
    cf_ok.append(ok)
    if ok:
        Tc = cf[ct]
        cf_a.append(float((cf[a] @ Tc.T).max())); cf_b.append(float((cf[b] @ Tc.T).max()))
    else:
        cf_a.append(np.nan); cf_b.append(np.nan)
    pop_a.append(logpop[a]); pop_b.append(logpop[b])

sub_a, sub_b = np.array(sub_a), np.array(sub_b)
cf_a, cf_b = np.array(cf_a), np.array(cf_b)
pop_a, pop_b = np.array(pop_a), np.array(pop_b)
cf_ok = np.array(cf_ok)

def acc(x, y):
    return float(np.mean(np.where(x == y, 0.5, x > y)))

n = len(pairs)
print(f"\n== popularity-matched pairwise preference (n={n} pairs) ==")
print(f"popularity baseline : {acc(pop_a, pop_b):.3f}   (should be ~0.5 - matching sanity check)")
print(f"subtitle            : {acc(sub_a, sub_b):.3f}")
m = cf_ok
print(f"CF (n={int(m.sum())})       : {acc(cf_a[m], cf_b[m]):.3f}")
print(f"subtitle (same n)   : {acc(sub_a[m], sub_b[m]):.3f}")
za, zb = (sub_a - sub_a.mean()) / sub_a.std(), (sub_b - sub_a.mean()) / sub_a.std()
zca = (cf_a[m] - np.nanmean(cf_a[m])) / np.nanstd(cf_a[m])
zcb = (cf_b[m] - np.nanmean(cf_a[m])) / np.nanstd(cf_a[m])
blend_a = 0.8 * zca + 0.2 * za[m]; blend_b = 0.8 * zcb + 0.2 * zb[m]
print(f"blend 0.8CF+0.2sub  : {acc(blend_a, blend_b):.3f}")
# binomial 95% CI half-width for reference
print(f"(95% CI half-width ~ {1.96*0.5/np.sqrt(n):.3f})")
