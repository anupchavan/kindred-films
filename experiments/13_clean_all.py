#!/usr/bin/env python3
"""Clean the FULL Sublens corpus (all ~19.4k movies) into data/clean_full/ for the
VPS backfill. Separate from data/clean/ so local pilot scripts are unaffected.
Writes data/manifest_full.csv. Resumable: skips already-written outputs."""
import csv, importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sc", ROOT / "01_select_clean.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

OUT = ROOT / "data" / "clean_full"
OUT.mkdir(parents=True, exist_ok=True)

movies, imdb = {}, {}
with open(sc.DATA / "Sublens_20M_metadata" / "movies_subs_filtered.csv", newline="") as f:
    for row in csv.DictReader(f):
        movies[int(row["movielens_id"])] = row["movie_name"]
with open(sc.DATA / "Sublens_20M_metadata" / "links_subs_filtered.csv", newline="") as f:
    for row in csv.DictReader(f):
        imdb[int(row["movielens_id"])] = int(row["imdb_id"])

rows, ok = [], 0
for n, (mlid, iid) in enumerate(sorted(imdb.items()), 1):
    out = OUT / f"{iid:07d}.txt"
    if out.exists():
        ok += 1
        continue
    result, info, dur = sc.validate(iid)
    if result is None:
        rows.append((iid, mlid, movies.get(mlid, ""), "full", 0, 0, info))
        continue
    cues, _ = result
    out.write_text("\n".join(cues))
    rows.append((iid, mlid, movies.get(mlid, ""), "full", len(cues), info, "ok"))
    ok += 1
    if n % 1000 == 0:
        print(f"{n}/{len(imdb)} processed, {ok} ok", flush=True)

mf = ROOT / "data" / "manifest_full.csv"
new = not mf.exists()
with open(mf, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["imdb_id", "movielens_id", "title", "cluster", "cues", "words", "status"])
    w.writerows(rows)
print(f"done: {ok} clean of {len(imdb)}")
