#!/usr/bin/env python3
"""Compare leakage-experiment arms: per-facet cosine + top neighbors."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
weights = json.load(open(ROOT / "data" / "weights.json"))
emb = {f: np.load(ROOT / "data" / f"emb_{f}.npy") for f in FACETS}
titles = {int(r["imdb_id"]): r["title"] for r in csv.DictReader(open(ROOT / "data" / "manifest.csv"))
          if r["status"] == "ok"}
S = sum(weights[f] * (emb[f] @ emb[f].T) for f in FACETS)
np.fill_diagonal(S, -1)

def facet_row(a, b):
    return {f: float(emb[f][idx[a]] @ emb[f][idx[b]]) for f in FACETS}

def neighbors(a, k=10, exclude=()):
    order = np.argsort(-S[idx[a]])
    out = []
    for o in order:
        if ids[o] in exclude:
            continue
        out.append((titles.get(ids[o], ids[o]), float(S[idx[a]][o])))
        if len(out) == k:
            break
    return out

PAIRS = [("KISS vs THEATRE (obscure - expect near-identical)", 90000001, 90000002),
         ("Interstellar vs HARVEST (famous - divergence = prior leakage)", 816692, 90000003)]
for label, a, b in PAIRS:
    print(f"\n== {label}")
    row = facet_row(a, b)
    print("  " + "  ".join(f"{f}={v:.3f}" for f, v in row.items()))
    print(f"  weighted: {sum(weights[f]*row[f] for f in FACETS):.3f}")

exp = {90000001, 90000002, 90000003}
for name, a in [("KISS", 90000001), ("THEATRE", 90000002),
                ("Interstellar (real)", 816692), ("HARVEST", 90000003)]:
    print(f"\n-- top 8 for {name} (experiment arms excluded)")
    for r, (t, s) in enumerate(neighbors(a, 8, exclude=exp - {a} | ({816692} if a == 90000003 else set())), 1):
        print(f"  {r}. {t}  ({s:.3f})")
