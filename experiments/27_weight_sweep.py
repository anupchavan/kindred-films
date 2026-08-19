#!/usr/bin/env python3
"""Fair v1-vs-v2.1 comparison: tune facet weights on half the genome anchors,
evaluate on the held-out half. Same protocol for both profile generations."""
import csv, json, random
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))

def genome(ids):
    imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
    ml_needed = {imdb2ml[i]: i for i in ids if i in imdb2ml}
    vecs, cur, buf = {}, None, []
    for mlid, tagid, rel in list(csv.reader(open(SCRATCH / "genome-scores.csv")))[1:]:
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
    GT = G @ G.T
    np.fill_diagonal(GT, -1)
    return covered, GT

def metrics(S, GT, rows):
    n = S.shape[0]
    r1, ov = [], []
    for a in rows:
        ours = np.argsort(-S[a]); gt = np.argsort(-GT[a])
        gtr = np.empty(n); gtr[gt] = np.arange(n)
        r1.append(gtr[ours[0]] + 1)
        ov.append(len(set(ours[:10]) & set(gt[:10])) / 10)
    return np.median(r1), np.mean(ov)

for name, prefix, idsfile, facets in [
        ("v1  ", "data/emb_", "data/ids.json", ["plot", "themes", "vibe", "arc", "texture", "world"]),
        ("v2.1", "data/v2_emb_", "data/v2_ids.json", ["plot", "hook", "themes", "vibe", "arc", "texture", "world"])]:
    ids = json.load(open(ROOT / idsfile))
    covered, GT = genome(ids)
    idx = {i: k for k, i in enumerate(ids)}
    sub = [idx[i] for i in covered]
    sims_f = {}
    for f in facets:
        e = np.load(ROOT / f"{prefix}{f}.npy")
        sims_f[f] = (e @ e.T)[np.ix_(sub, sub)]
    rng = random.Random(11)
    rows = list(range(len(covered)))
    rng.shuffle(rows)
    tune, test = rows[:len(rows)//2], rows[len(rows)//2:]
    best, best_w = None, None
    for trial in range(250):
        raw = [rng.random() ** 2 for _ in facets]
        s = sum(raw)
        w = {f: r / s for f, r in zip(facets, raw)}
        S = sum(w[f] * sims_f[f] for f in facets)
        np.fill_diagonal(S, -1)
        _, ov = metrics(S, GT, tune)
        if best is None or ov > best:
            best, best_w = ov, w
    S = sum(best_w[f] * sims_f[f] for f in facets)
    np.fill_diagonal(S, -1)
    r1, ov = metrics(S, GT, test)
    print(f"{name}: held-out median top1-rank {r1:5.1f}  top10-overlap {ov:.1%}   "
          f"weights " + " ".join(f"{f}={best_w[f]:.2f}" for f in facets))
