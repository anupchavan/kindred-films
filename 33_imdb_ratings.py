#!/usr/bin/env python3
"""Fetch IMDb's official ratings dataset and extract ratings for our index.
Display/sort layer only - never a similarity feature. Rerun anytime to refresh."""
import csv, gzip, json, tempfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
gz = Path(tempfile.gettempdir()) / "title.ratings.tsv.gz"
print("downloading IMDb ratings dataset...")
urllib.request.urlretrieve("https://datasets.imdbws.com/title.ratings.tsv.gz", gz)

ids = set(json.load(open(ROOT / "data" / "ids.json")))
out = {}
with gzip.open(gz, "rt") as f:
    next(f)
    for line in f:
        tconst, rating, votes = line.rstrip("\n").split("\t")
        i = int(tconst[2:])
        if i in ids:
            out[i] = [float(rating), int(votes)]
json.dump(out, open(ROOT / "data" / "imdb_ratings.json", "w"))
print(f"ratings found for {len(out)}/{len(ids)} indexed movies")
