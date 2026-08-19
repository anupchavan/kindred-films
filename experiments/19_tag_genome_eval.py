#!/usr/bin/env python3
"""Tag-Genome evaluation harness (evaluation ONLY - never a feature).

Ground truth: cosine similarity between MovieLens Tag-Genome relevance vectors
(~1,128 human-derived tags incl. tonal ones). Metrics per anchor, aggregated:
  - median GT-rank of our top-1 recommendation (paper's headline metric; lower=better)
  - mean fraction of our top-10 inside the GT top-10 (higher=better)

Usage: 19_tag_genome_eval.py <emb_prefix> [facets...]
  e.g.  19_tag_genome_eval.py data/emb_ plot themes vibe arc texture world
        (weights read from data/weights.json, missing facets ignored)
"""
import csv, json, sys
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))

prefix = sys.argv[1] if len(sys.argv) > 1 else "data/emb_"
facets = sys.argv[2:] or ["plot", "themes", "vibe", "arc", "texture", "world"]
ids = json.load(open(ROOT / "data" / ("v2_ids.json" if "v2" in prefix else "ids.json")))
weights = json.load(open(ROOT / "data" / ("v2_weights.json" if "v2" in prefix and (ROOT/"data/v2_weights.json").exists() else "weights.json")))

# our similarity matrix
S = None
for f in facets:
    p = ROOT / f"{prefix}{f}.npy"
    if not p.exists():
        continue
    e = np.load(p)
    S = (0 if S is None else S) + weights.get(f, 0.1) * (e @ e.T)
np.fill_diagonal(S, -1)

# imdb -> movielens (ml-25m links)
imdb2ml = {}
with open(SCRATCH / "links.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        if row["imdbId"]:
            imdb2ml[int(row["imdbId"])] = int(row["movieId"])

# genome vectors for movies we have
print("loading genome scores...", file=sys.stderr)
ml_needed = {imdb2ml[i]: i for i in ids if i in imdb2ml}
vecs = {}
with open(SCRATCH / "genome-scores.csv", newline="") as fh:
    r = csv.reader(fh)
    next(r)
    cur, buf = None, []
    for mlid, tagid, rel in r:
        mlid = int(mlid)
        if mlid != cur:
            if cur in ml_needed and buf:
                vecs[ml_needed[cur]] = np.array(buf)
            cur, buf = mlid, []
        buf.append(float(rel))
    if cur in ml_needed and buf:
        vecs[ml_needed[cur]] = np.array(buf)

covered = [i for i in ids if i in vecs]
print(f"genome coverage: {len(covered)}/{len(ids)} movies")
G = np.stack([vecs[i] / np.linalg.norm(vecs[i]) for i in covered])
GT = G @ G.T
np.fill_diagonal(GT, -1)

idx_all = {i: k for k, i in enumerate(ids)}
sub = [idx_all[i] for i in covered]
S_c = S[np.ix_(sub, sub)]

ranks1, overlap10 = [], []
n = len(covered)
for a in range(n):
    ours = np.argsort(-S_c[a])
    gt_order = np.argsort(-GT[a])
    gt_rank = np.empty(n); gt_rank[gt_order] = np.arange(n)
    ranks1.append(gt_rank[ours[0]] + 1)
    overlap10.append(len(set(ours[:10]) & set(gt_order[:10])) / 10)

print(f"\n== Tag-Genome eval ({prefix}, {len(facets)} facets, n={n}) ==")
print(f"median GT-rank of top-1 rec : {np.median(ranks1):.1f}  (random ~{n/2:.0f})")
print(f"mean top10 GT-overlap       : {np.mean(overlap10):.1%}  (random ~{10/n:.1%})")
