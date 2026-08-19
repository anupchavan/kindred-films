#!/usr/bin/env python3
"""THE deployment-realistic measure: time-split leave-one-out.

CF factors trained ONLY on ratings before 2015-01-01 (the freeze). Test: what did
users love AFTER the freeze? Buckets: held-out movie existed pre-freeze (crowd knows
it) vs released post-freeze (crowd has zero data - the new-release case).

Systems compared on identical users/holdouts/candidates:
  WITHOUT subs: pure CF; candidates unknown to CF get the user's mean CF score
  WITH subs   : z-blend a=0.8 CF+content where CF exists; content where it doesn't
  (content-only shown for reference)
"""
import csv, json, random
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
T = 1420070400          # 2015-01-01 UTC
N_USERS, N_NEG, K, KF = 20000, 999, 10, 128
rng = random.Random(42)

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
W = json.load(open(ROOT / "data" / "weights.json"))
content = np.concatenate(
    [np.sqrt(W[f]) * np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1).astype(np.float32)

imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}

print("loading ratings with timestamps...", flush=True)
pre_u, pre_i, pre_r = [], [], []
u_map, i_map = {}, {}
user_pre = defaultdict(list)
user_post = defaultdict(list)
first_seen = {}
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, ts = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        row, rr, tt = idx[imdb], float(r), int(ts)
        first_seen[row] = min(first_seen.get(row, tt), tt)
        if tt < T:
            uid = u_map.setdefault(u, len(u_map))
            iid = i_map.setdefault(row, len(i_map))
            pre_u.append(uid); pre_i.append(iid); pre_r.append(rr)
            user_pre[u].append((row, rr))
        else:
            user_post[u].append((row, rr))
print(f"pre-T ratings={len(pre_r)}, users with post-T activity={len(user_post)}", flush=True)

print("training frozen CF (SVD k=128 on pre-2015 only)...", flush=True)
M = csr_matrix((pre_r, (pre_u, pre_i)), shape=(len(u_map), len(i_map)), dtype=np.float32)
sums = np.asarray(M.sum(axis=1)).ravel(); counts = np.diff(M.indptr)
means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
M.data = M.data - np.repeat(means, counts)
_, s, Vt = svds(M, k=KF)
item_f = (Vt.T * s).astype(np.float32)
item_f /= (np.linalg.norm(item_f, axis=1, keepdims=True) + 1e-9)
cf = np.zeros((len(ids), KF), dtype=np.float32)
for row, iid in i_map.items():
    cf[row] = item_f[iid]
cf_valid = np.abs(cf).sum(axis=1) > 1e-9
print(f"CF covers {int(cf_valid.sum())}/{len(ids)} movies (frozen at 2015)", flush=True)

def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)

eligible = [u for u in user_post
            if len(user_pre.get(u, [])) >= 10 and any(r >= 4.5 for _, r in user_post[u])]
rng.shuffle(eligible)
users = eligible[:N_USERS]
print(f"test users={len(users)}", flush=True)

buckets = {"known": {"with": [], "without": [], "content": []},
           "new-release": {"with": [], "without": [], "content": []}}
for n, u in enumerate(users):
    post_loved = [(row, r) for row, r in user_post[u] if r >= 4.5]
    held_row, _ = rng.choice(post_loved)
    taste = [row for row, _ in sorted(user_pre[u], key=lambda x: -x[1])[:10]]
    seen = {row for row, _ in user_pre[u]} | {row for row, _ in user_post[u]}
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
        mean_cf = raw[cf_valid[cand]].mean() if cf_valid[cand].any() else 0.0
        without = np.where(cf_valid[cand], raw, mean_cf)
        zc, zf = z(c_score), z(np.where(cf_valid[cand], raw, mean_cf))
        blend = 0.2 * zc + 0.8 * zf
        with_subs = np.where(cf_valid[cand], blend, zc)
    else:
        without = np.zeros(len(cand)); with_subs = z(c_score)
    bucket = "known" if first_seen.get(held_row, T) < T else "new-release"
    for name, sc in (("with", with_subs), ("without", without), ("content", c_score)):
        rank = int((sc > sc[0]).sum())
        buckets[bucket][name].append(1.0 if rank < K else 0.0)
    if (n + 1) % 4000 == 0:
        print(f"{n+1}/{len(users)}", flush=True)

print("\n== TIME-SPLIT VERDICT (HR@10, frozen crowd) ==")
for b, r in buckets.items():
    nn = len(r["with"])
    if nn == 0:
        continue
    print(f"\n[{b}] n={nn}")
    print(f"  WITHOUT subtitles : {np.mean(r['without']):.3f}")
    print(f"  WITH subtitles    : {np.mean(r['with']):.3f}")
    print(f"  (content alone    : {np.mean(r['content']):.3f})")
