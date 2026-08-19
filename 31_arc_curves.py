#!/usr/bin/env python3
"""Structural arc channel: 20-window valence curve per movie, from raw subtitle text.
No LLM involved. Curves are z-normalized (shape, not absolute tone) and saved
aligned to data/ids.json as data/arc_curve.npy (unit-L2 rows; zero rows = no text)."""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).parent
N_WIN = 20
ids = json.load(open(ROOT / "data" / "ids.json"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)

POS = ["light playful banter, jokes and teasing, warmth, laughter, happy easy scenes",
       "joyful celebration, hope, tenderness, comfort, things going well"]
NEG = ["grief, despair, cruelty, violence, loss, characters suffering",
       "dread, menace, anger, tragedy, everything falling apart"]
anchor = model.encode(POS + NEG, normalize_embeddings=True)
pos_a, neg_a = anchor[:2].mean(0), anchor[2:].mean(0)

def text_of(i):
    for d in ("data/clean", "data/clean_full"):
        p = ROOT / d / f"{i:07d}.txt"
        if p.exists():
            return p.read_text(errors="replace").splitlines()
    return None

CKPT = ROOT / "data" / "arc_curve_ckpt.npz"
if CKPT.exists():
    ck = np.load(CKPT)
    curves, start_row = ck["curves"], int(ck["row"])
    print(f"resuming from row {start_row}", flush=True)
else:
    curves, start_row = np.zeros((len(ids), N_WIN), dtype=np.float32), 0
missing = 0
batch_texts, batch_pos = [], []
for r, i in enumerate(ids):
    if r < start_row:
        continue
    cues = text_of(i)
    if not cues or len(cues) < N_WIN * 3:
        missing += 1
        continue
    step = len(cues) / N_WIN
    for w in range(N_WIN):
        seg = " ".join(cues[int(w * step):int((w + 1) * step)])[:1500]
        batch_texts.append(seg)
        batch_pos.append((r, w))
    if len(batch_texts) >= 2000:
        emb = model.encode(batch_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        for (rr, ww), e in zip(batch_pos, emb):
            curves[rr, ww] = float(e @ pos_a - e @ neg_a)
        batch_texts, batch_pos = [], []
        np.savez(CKPT, curves=curves, row=r + 1)
        print(f"{r+1}/{len(ids)}", flush=True)
if batch_texts:
    emb = model.encode(batch_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    for (rr, ww), e in zip(batch_pos, emb):
        curves[rr, ww] = float(e @ pos_a - e @ neg_a)

# shape-normalize: z per movie, then unit L2 so dot product = curve-shape similarity
m = curves.std(axis=1) > 1e-6
z = curves.copy()
z[m] = (curves[m] - curves[m].mean(axis=1, keepdims=True)) / curves[m].std(axis=1, keepdims=True)
norms = np.linalg.norm(z, axis=1, keepdims=True)
z = np.where(norms > 1e-6, z / (norms + 1e-12), 0)
np.save(ROOT / "data" / "arc_curve.npy", z.astype(np.float32))
print(f"saved curves for {int(m.sum())}/{len(ids)} movies ({missing} without text)")
