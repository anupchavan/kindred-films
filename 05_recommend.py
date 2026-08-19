#!/usr/bin/env python3
"""Recommend movies similar to a title, with per-facet score table and tunable weights.

Usage:
  05_recommend.py "title" [k]                    default blend
  05_recommend.py "title" [k] --feel             vibe/arc/texture-heavy (how it feels)
  05_recommend.py "title" [k] --story            plot/themes-heavy (what happens)
  05_recommend.py "title" --vibe 3 --plot 0.5    per-facet multipliers (the sliders)
  05_recommend.py "title" --director "name"      disambiguate via TMDB (lookup only)

Columns are cosine similarities per facet; 'match' is the weighted blend.
"""
import argparse, csv, json, re, sys, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]

ap = argparse.ArgumentParser()
ap.add_argument("title")
ap.add_argument("k", nargs="?", type=int, default=10)
ap.add_argument("--director")
ap.add_argument("--feel", action="store_true", help="weight how the movie feels (vibe/arc/texture)")
ap.add_argument("--story", action="store_true", help="weight what happens (plot/themes)")
ap.add_argument("--curve", type=float, default=0.0, metavar="X",
                help="weight share (0-1) for the structural valence-curve channel (tonal trajectory shape)")
ap.add_argument("--by-rating", action="store_true",
                help="re-sort the top-k by IMDb rating (display order only; selection stays similarity-based)")
ap.add_argument("--cf", type=float, default=0.0, metavar="X",
                help="weight share (0-1) for the collaborative-filtering channel (MovieLens latent factors); "
                     "movies without rating data gracefully fall back to content score")
ap.add_argument("--pool", type=int, default=200, help="candidate pool size for --curve rerank")
for f in FACETS:
    ap.add_argument(f"--{f}", type=float, default=1.0, metavar="X", help=f"multiplier for {f}")
args = ap.parse_args()

weights = json.load(open(ROOT / "data" / "weights.json"))
if args.feel:
    for f in ("vibe", "arc", "texture"):
        weights[f] *= 3
if args.story:
    for f in ("plot", "themes"):
        weights[f] *= 3
for f in FACETS:
    weights[f] *= getattr(args, f)
tot = sum(weights.values())
weights = {f: w / tot for f, w in weights.items()}

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
emb = {f: np.load(ROOT / "data" / f"emb_{f}.npy") for f in FACETS}

titles = {}
with open(ROOT / "data" / "manifest.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["status"] == "ok" and int(row["imdb_id"]) in idx:
            titles[int(row["imdb_id"])] = row["title"]

def norm(t):
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t.lower()).strip()
    t = re.sub(r"^(the|a|an)\s+|,\s*(the|a|an)$", "", t).strip()
    return t

anchor = None
if args.director:
    key = next((l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("TMDB_API_KEY")), None)
    if not key:
        sys.exit("--director needs TMDB_API_KEY in .env")
    def tj(url):
        with urllib.request.urlopen(url) as r:
            return json.load(r)
    results = tj(f"https://api.themoviedb.org/3/search/movie?query={urllib.parse.quote_plus(args.title)}&api_key={key}").get("results", [])[:5]
    want = args.director.lower()
    for c in results:
        crew = tj(f"https://api.themoviedb.org/3/movie/{c['id']}/credits?api_key={key}").get("crew", [])
        if any(want in p["name"].lower() for p in crew if p["job"] == "Director"):
            ext = tj(f"https://api.themoviedb.org/3/movie/{c['id']}/external_ids?api_key={key}").get("imdb_id")
            if ext and int(ext[2:]) in idx:
                anchor = int(ext[2:])
            else:
                sys.exit(f"{c['title']} ({(c.get('release_date') or '?')[:4]}) [{ext}] is not in the index yet "
                         f"- add it: 07_add_online.py {ext}")
            break
    if anchor is None:
        sys.exit(f"no TMDB match for '{args.title}' directed by {args.director}")
else:
    q = norm(args.title)
    scored = []
    for i, t in titles.items():
        tn = norm(t)
        if tn == q:
            scored.append((3, i))
        elif tn.startswith(q):
            scored.append((2, i))
        elif re.search(rf"\b{re.escape(q)}\b", tn):
            scored.append((1, i))
        elif q in tn:
            scored.append((0.5, i))
    if not scored:
        sys.exit(f"no movie matching '{args.title}' in the index")
    scored.sort(key=lambda s: (-s[0], s[1]))
    anchor = scored[0][1]
    others = [titles[i] for _, i in scored[1:4] if norm(titles[i]) != norm(titles[anchor])]
    if others:
        print(f"(also matched: {'; '.join(others)})")

# per-facet similarity of anchor vs everyone (row ops only - no full matrix)
fs = {f: emb[f] @ emb[f][idx[anchor]] for f in FACETS}
show_facets = list(FACETS)
match = sum(weights[f] * fs[f] for f in FACETS)
match[idx[anchor]] = -1
if args.cf > 0:
    cf = np.load(ROOT / "data" / "cf_emb.npy")
    if np.abs(cf[idx[anchor]]).sum() < 1e-9:
        print("(note: anchor has no rating data - --cf ignored, pure content similarity)")
    else:
        fs["cf"] = cf @ cf[idx[anchor]]
        valid = np.abs(cf).sum(axis=1) > 1e-9
        blended = (1 - args.cf) * match + args.cf * fs["cf"]
        match = np.where(valid, blended, match)
        match[idx[anchor]] = -1
        weights["cf"] = args.cf
        show_facets.append("cf")
if args.curve > 0:
    # curve re-ranks WITHIN the semantic family (top prose matches), never against it
    curves = np.load(ROOT / "data" / "arc_curve.npy")
    fs["curve"] = curves @ curves[idx[anchor]]
    weights["curve"] = args.curve   # display only: rerank share within the pool
    show_facets.append("curve")
    pool = np.argsort(-match)[:args.pool]
    reranked = (1 - args.curve) * match[pool] + args.curve * fs["curve"][pool]
    match = np.full_like(match, -np.inf)
    match[pool] = reranked

ratings = {}
rp = ROOT / "data" / "imdb_ratings.json"
if rp.exists():
    ratings = {int(k): v for k, v in json.load(open(rp)).items()}

print(f"because you liked: {titles[anchor]}" +
      (f"  [imdb {ratings[anchor][0]}]" if anchor in ratings else ""))
print("weights: " + "  ".join(f"{f}={weights[f]:.2f}" for f in show_facets) + "\n")
hdr = f"{'#':>3} {'title':<42} {'match':>6} {'imdb':>5} " + " ".join(f"{f[:4]:>5}" for f in show_facets)
print(hdr)
print("-" * len(hdr))
order = np.argsort(-match)[:args.k]
top, floor = match[order[0]], match[order[-1]]
if args.by_rating:
    order = sorted(order, key=lambda o: -(ratings.get(ids[o], [0])[0]))
for r, o in enumerate(order, 1):
    t = titles.get(ids[o], str(ids[o]))[:42]
    bar_n = int(round(4 * (match[o] - floor) / (top - floor + 1e-9)))
    imdb = f"{ratings[ids[o]][0]:.1f}" if ids[o] in ratings else "  -"
    print(f"{r:>3} {t:<42} {match[o]:>6.3f} {imdb:>5} " +
          " ".join(f"{fs[f][o]:>5.3f}"[1:] for f in show_facets) + "  " + "▊" * bar_n)
print(f"\n(index avg match for this movie: {np.mean(match[match > -1]):.3f} - "
      f"top hit {'+' if top - np.mean(match[match > -1]) >= 0 else ''}{top - np.mean(match[match > -1]):.3f} above)")
