#!/usr/bin/env python3
"""Tag-Genome metrics for the bakeoff: anchors = the 50 DS-profiled movies,
comparing the original similarity matrix vs the DS-swapped one."""
import csv, json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
ids = json.load(open(ROOT / "data" / "ids.json"))
anchors = set(json.load(open(ROOT / "data" / "bakeoff_ids.json")))

imdb2ml = {}
with open(SCRATCH / "links.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        if row["imdbId"]:
            imdb2ml[int(row["imdbId"])] = int(row["movieId"])
ml_needed = {imdb2ml[i]: i for i in ids if i in imdb2ml}
vecs = {}
with open(SCRATCH / "genome-scores.csv", newline="") as fh:
    r = csv.reader(fh); next(r)
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
G = np.stack([vecs[i] / np.linalg.norm(vecs[i]) for i in covered])
GT = G @ G.T; np.fill_diagonal(GT, -1)
idx_all = {i: k for k, i in enumerate(ids)}
sub = [idx_all[i] for i in covered]
cov_anchor_rows = [k for k, i in enumerate(covered) if i in anchors]
print(f"genome-covered anchors: {len(cov_anchor_rows)}/{len(anchors)}")

for tag in ("orig", "ds-swap"):
    S = np.load(ROOT / "data" / f"bakeoff_tmp_{tag}.npy")[np.ix_(sub, sub)]
    np.fill_diagonal(S, -1)
    n = len(covered)
    ranks1, overlap10 = [], []
    for a in cov_anchor_rows:
        ours = np.argsort(-S[a]); gt_order = np.argsort(-GT[a])
        gt_rank = np.empty(n); gt_rank[gt_order] = np.arange(n)
        ranks1.append(gt_rank[ours[0]] + 1)
        overlap10.append(len(set(ours[:10]) & set(gt_order[:10])) / 10)
    print(f"{tag:8s}: median GT-rank of top-1 = {np.median(ranks1):.1f}   top10 overlap = {np.mean(overlap10):.1%}")
