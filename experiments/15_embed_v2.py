#!/usr/bin/env python3
"""Incrementally embed v2 profiles: appends any profiled-but-unindexed movies to
data/v2_emb_*.npy + data/v2_ids.json. 7 facets (adds hook)."""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACETS = ["plot", "hook", "themes", "vibe", "arc", "texture", "world"]
PROFILES = ROOT / "data" / "profiles_v2"
IDS = ROOT / "data" / "v2_ids.json"

def load(p):
    raw = p.read_text().strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        card = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return card if all(f in card for f in FACETS) else None

def facet_text(card, facet):
    v = card[facet]
    return ", ".join(v) if isinstance(v, list) else str(v)

ids = json.load(open(IDS)) if IDS.exists() else []
have = set(ids)
cards = {}
for p in sorted(PROFILES.glob("*.json")):
    i = int(p.stem)
    if i in have:
        continue
    c = load(p)
    if c:
        cards[i] = c
if not cards:
    print("embed: nothing new")
    sys.exit(0)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
new = sorted(cards)
for facet in FACETS:
    f_path = ROOT / "data" / f"v2_emb_{facet}.npy"
    emb = model.encode([facet_text(cards[i], facet) for i in new],
                       normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    if f_path.exists():
        emb = np.vstack([np.load(f_path), emb])
    np.save(f_path, emb)
json.dump(ids + new, open(IDS, "w"))
print(f"embed: +{len(new)} -> {len(ids) + len(new)} indexed")
