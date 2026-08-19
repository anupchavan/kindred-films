#!/usr/bin/env python3
"""Embed profile-card facets, build weighted-cosine retrieval, evaluate against clusters."""
import csv, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
PROFILES = ROOT / "data" / "profiles"
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
WEIGHTS = {"plot": 0.30, "themes": 0.20, "vibe": 0.20, "arc": 0.10, "texture": 0.10, "world": 0.10}
TOPK = 10

def load_profiles():
    profiles = {}
    for p in sorted(PROFILES.glob("*.json")):
        raw = p.read_text().strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            print(f"  unparseable: {p.name}", file=sys.stderr)
            continue
        try:
            card = json.loads(m.group(0))
        except json.JSONDecodeError:
            print(f"  bad json: {p.name}", file=sys.stderr)
            continue
        if all(f in card for f in FACETS):
            profiles[int(p.stem)] = card
    return profiles

def facet_text(card, facet):
    v = card[facet]
    return ", ".join(v) if isinstance(v, list) else str(v)

def main():
    titles = {}
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                titles[int(row["imdb_id"])] = row["title"]

    profiles = load_profiles()
    ids = sorted(profiles)
    print(f"profiles loaded: {len(profiles)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    sims = np.zeros((len(ids), len(ids)))
    for facet in FACETS:
        texts = [facet_text(profiles[i], facet) for i in ids]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims += WEIGHTS[facet] * (emb @ emb.T)
        np.save(ROOT / "data" / f"emb_{facet}.npy", emb)
    np.fill_diagonal(sims, -1)
    (ROOT / "data" / "ids.json").write_text(json.dumps(ids))

    idx = {i: k for k, i in enumerate(ids)}
    def top(anchor, k=TOPK):
        order = np.argsort(-sims[idx[anchor]])[:k]
        return [ids[o] for o in order]

    # ---- evaluation against clusters ----
    clusters = {c: [m for m in ms if m in idx]
                for c, ms in json.load(open(ROOT / "data" / "eval_clusters.json")).items()}
    rows, hits_total, exp_total = [], 0, 0
    for cname, members in clusters.items():
        if len(members) < 3:
            continue
        c_hits = c_exp = 0
        for a in members:
            expected = set(members) - {a}
            got = set(top(a))
            c_hits += len(got & expected)
            c_exp += min(len(expected), TOPK)
        rows.append((cname, len(members), c_hits / c_exp))
        hits_total += c_hits; exp_total += c_exp
    rows.sort(key=lambda r: -r[2])
    print(f"\n== recall@{TOPK} vs expected cluster members ==")
    for cname, n, r in rows:
        print(f"  {cname:22s} n={n}  {r:.0%}")
    print(f"  {'OVERALL':22s}       {hits_total/exp_total:.0%}")

    # ---- qualitative: show top-10 for a few anchors ----
    def show(name_frag):
        matches = [i for i, t in titles.items() if name_frag.lower() in t.lower() and i in idx]
        if not matches:
            return
        a = matches[0]
        print(f"\n-- {titles[a]} --")
        for r, m in enumerate(top(a), 1):
            print(f"  {r:2d}. {titles.get(m, m)}  ({sims[idx[a]][idx[m]]:.3f})")
    for frag in ["Interstellar", "Shaun of the Dead", "Notting Hill", "Goodfellas", "Toy Story (1995", "3 Idiots", "Conjuring", "John Wick", "Memento"]:
        show(frag)

if __name__ == "__main__":
    main()
