#!/usr/bin/env python3
"""Incrementally add movies to the index by movielens id: clean -> (codex profiles via
02_profile.sh) -> embed -> append. No retraining, no touching existing entries.

Usage: 06_add_movies.py clean <mlid> [mlid...]   # stage 1: validate + write clean text
       06_add_movies.py embed                     # stage 2: embed any profiled-but-unindexed movies
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sc", ROOT / "01_select_clean.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]

movies, imdb = {}, {}
with open(sc.DATA / "Sublens_20M_metadata" / "movies_subs_filtered.csv", newline="") as f:
    for row in csv.DictReader(f):
        movies[int(row["movielens_id"])] = row["movie_name"]
with open(sc.DATA / "Sublens_20M_metadata" / "links_subs_filtered.csv", newline="") as f:
    for row in csv.DictReader(f):
        imdb[int(row["movielens_id"])] = int(row["imdb_id"])

def clean(mlids):
    rows = []
    for mlid in mlids:
        iid = imdb.get(mlid)
        if iid is None:
            print(f"  {mlid}: not in Sublens links")
            continue
        if (sc.CLEAN / f"{iid:07d}.txt").exists():
            print(f"  {movies[mlid]}: already cleaned")
            continue
        result, info, dur = sc.validate(iid)
        if result is None:
            print(f"  {movies[mlid]}: FAILED validation ({info})")
            continue
        cues, _ = result
        (sc.CLEAN / f"{iid:07d}.txt").write_text("\n".join(cues))
        rows.append((iid, mlid, movies[mlid], "added", len(cues), info, "ok"))
        print(f"  {movies[mlid]}: cleaned ({len(cues)} cues, {info} words)")
    with open(sc.OUT / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\nnow run: ./02_profile.sh   (profiles only the new movies), then: 06_add_movies.py embed")

def embed():
    ids = json.load(open(ROOT / "data" / "ids.json"))
    have = set(ids)
    from sentence_transformers import SentenceTransformer
    profiles = sc = None  # noqa - avoid confusion; load below
    spec03 = importlib.util.spec_from_file_location("ee", ROOT / "03_embed_eval.py")
    ee = importlib.util.module_from_spec(spec03)
    spec03.loader.exec_module(ee)
    cards = ee.load_profiles()
    new = [i for i in sorted(cards) if i not in have]
    if not new:
        print("nothing new to index")
        return
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    for facet in FACETS:
        old = np.load(ROOT / "data" / f"emb_{facet}.npy")
        texts = [ee.facet_text(cards[i], facet) for i in new]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        np.save(ROOT / "data" / f"emb_{facet}.npy", np.vstack([old, emb]))
    json.dump(ids + new, open(ROOT / "data" / "ids.json", "w"))
    print(f"indexed {len(new)} new movies; index size now {len(ids) + len(new)}")

if sys.argv[1] == "clean":
    clean([int(a) for a in sys.argv[2:]])
elif sys.argv[1] == "embed":
    embed()
