#!/usr/bin/env python3
"""v2 recommender (7 facets incl. hook). Same interface as 05:
16_recommend_v2.py "title" [k] [--feel|--story|--hook] [--plot X --vibe X ...]"""
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACETS = ["plot", "hook", "themes", "vibe", "arc", "texture", "world"]
DEFAULT_W = {"plot": .25, "hook": .17, "themes": .18, "vibe": .14, "arc": .03, "texture": .15, "world": .08}

ap = argparse.ArgumentParser()
ap.add_argument("title")
ap.add_argument("k", nargs="?", type=int, default=10)
ap.add_argument("--feel", action="store_true")
ap.add_argument("--story", action="store_true")
ap.add_argument("--hookish", action="store_true", help="premise/mechanism-heavy")
for f in FACETS:
    ap.add_argument(f"--{f}", type=float, default=1.0, metavar="X")
args = ap.parse_args()

wf = ROOT / "data" / "v2_weights.json"
weights = json.load(open(wf)) if wf.exists() else dict(DEFAULT_W)
if args.feel:
    for f in ("vibe", "arc", "texture"):
        weights[f] *= 3
if args.story:
    for f in ("plot", "themes"):
        weights[f] *= 3
if args.hookish:
    weights["hook"] *= 4
for f in FACETS:
    weights[f] *= getattr(args, f)
tot = sum(weights.values())
weights = {f: w / tot for f, w in weights.items()}

ids = json.load(open(ROOT / "data" / "v2_ids.json"))
idx = {i: k for k, i in enumerate(ids)}
emb = {f: np.load(ROOT / "data" / f"v2_emb_{f}.npy") for f in FACETS}

titles = {}
for mf in ("manifest.csv", "manifest_full.csv"):
    p = ROOT / "data" / mf
    if p.exists():
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["status"] == "ok":
                    titles.setdefault(int(row["imdb_id"]), row["title"])

def norm(t):
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t.lower()).strip()
    return re.sub(r"^(the|a|an)\s+|,\s*(the|a|an)$", "", t).strip()

q = norm(args.title)
scored = []
for i in ids:
    t = titles.get(i)
    if not t:
        continue
    tn = norm(t)
    if tn == q:
        scored.append((3, i))
    elif tn.startswith(q):
        scored.append((2, i))
    elif re.search(rf"\b{re.escape(q)}\b", tn):
        scored.append((1, i))
if not scored:
    sys.exit(f"no movie matching '{args.title}' in the v2 index (currently {len(ids)} movies)")
scored.sort(key=lambda s: (-s[0], s[1]))
anchor = scored[0][1]

fs = {f: emb[f] @ emb[f][idx[anchor]] for f in FACETS}
match = sum(weights[f] * fs[f] for f in FACETS)
match[idx[anchor]] = -1
print(f"because you liked: {titles.get(anchor, anchor)}   [v2 index: {len(ids)} movies]")
print("weights: " + "  ".join(f"{f}={weights[f]:.2f}" for f in FACETS) + "\n")
hdr = f"{'#':>3} {'title':<42} {'match':>6} " + " ".join(f"{f[:4]:>5}" for f in FACETS)
print(hdr)
print("-" * len(hdr))
order = np.argsort(-match)[:args.k]
top, floor = match[order[0]], match[order[-1]]
for r, o in enumerate(order, 1):
    t = str(titles.get(ids[o], ids[o]))[:42]
    bar = "▊" * int(round(4 * (match[o] - floor) / (top - floor + 1e-9)))
    print(f"{r:>3} {t:<42} {match[o]:>6.3f} " + " ".join(f"{fs[f][o]:>5.3f}"[1:] for f in FACETS) + "  " + bar)
print(f"\n(index avg: {np.mean(match[match > -1]):.3f}; top +{top - np.mean(match[match > -1]):.3f})")
