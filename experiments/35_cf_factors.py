#!/usr/bin/env python3
"""Collaborative-filtering channel: item latent factors from MovieLens 25M ratings.

Truncated SVD (k=128) on the mean-centered user-item matrix -> item factor vectors,
saved as data/cf_emb.npy aligned to data/ids.json (zero rows = movie has no/too few
ratings; the recommender masks those). This is the classic latent-factor method -
learned from real user behavior, deliberately OUTSIDE the subtitle-only core.
"""
import csv, json
import os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
K = 128
MIN_RATINGS = 20   # drop movies with too few ratings for a stable factor

ids = json.load(open(ROOT / "data" / "ids.json"))
imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2row = {}

print("loading 25M ratings...", flush=True)
users, movies, vals = [], [], []
uid_map = {}
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        uid = uid_map.setdefault(u, len(uid_map))
        mid = ml2row.setdefault(int(m), len(ml2row))
        users.append(uid); movies.append(mid); vals.append(float(r))
print(f"users={len(uid_map)}, movies={len(ml2row)}, ratings={len(vals)}", flush=True)

M = csr_matrix((vals, (users, movies)), shape=(len(uid_map), len(ml2row)), dtype=np.float32)
# mean-center per user (subtract user mean from their nonzeros)
sums = np.asarray(M.sum(axis=1)).ravel()
counts = np.diff(M.indptr)
means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
M.data = M.data - np.repeat(means, counts)

print(f"SVD k={K}...", flush=True)
U, s, Vt = svds(M, k=K)
item_f = (Vt.T * s).astype(np.float32)          # (n_movies, K)
norms = np.linalg.norm(item_f, axis=1, keepdims=True)
item_f = np.where(norms > 1e-9, item_f / (norms + 1e-12), 0)

rating_counts = np.asarray((M != 0).sum(axis=0)).ravel()
cf = np.zeros((len(ids), K), dtype=np.float32)
mapped = 0
for r, i in enumerate(ids):
    ml = imdb2ml.get(i)
    if ml is not None and ml in ml2row and rating_counts[ml2row[ml]] >= MIN_RATINGS:
        cf[r] = item_f[ml2row[ml]]
        mapped += 1
np.save(ROOT / "data" / "cf_emb.npy", cf)
print(f"saved cf_emb.npy: {mapped}/{len(ids)} movies have CF factors")
