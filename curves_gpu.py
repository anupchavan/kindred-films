#!/usr/bin/env python3
"""Self-contained GPU curve computation (runs on the RunPod pod).
Inputs:  ids.json + texts/ dir of <imdbid7>.txt cue files (one cue per line)
Output:  arc_curve.npy - (N, 20) shape-normalized valence curves aligned to ids.json
Usage:   python curves_gpu.py <workdir>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
N_WIN = 20
ids = json.load(open(WORK / "ids.json"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
if dev == "cuda":
    model.half()
print(f"device={dev}, movies={len(ids)}", flush=True)

POS = ["light playful banter, jokes and teasing, warmth, laughter, happy easy scenes",
       "joyful celebration, hope, tenderness, comfort, things going well"]
NEG = ["grief, despair, cruelty, violence, loss, characters suffering",
       "dread, menace, anger, tragedy, everything falling apart"]
anchor = model.encode(POS + NEG, normalize_embeddings=True)
pos_a, neg_a = anchor[:2].mean(0), anchor[2:].mean(0)

texts, positions = [], []
missing = 0
for r, i in enumerate(ids):
    p = WORK / "texts" / f"{i:07d}.txt"
    if not p.exists():
        missing += 1
        continue
    cues = p.read_text(errors="replace").splitlines()
    if len(cues) < N_WIN * 3:
        missing += 1
        continue
    step = len(cues) / N_WIN
    for w in range(N_WIN):
        texts.append(" ".join(cues[int(w * step):int((w + 1) * step)])[:1500])
        positions.append((r, w))
print(f"windows={len(texts)}, skipped={missing}", flush=True)

t0 = time.time()
emb = model.encode(texts, normalize_embeddings=True, batch_size=512, show_progress_bar=True)
print(f"encode took {time.time()-t0:.0f}s", flush=True)

curves = np.zeros((len(ids), N_WIN), dtype=np.float32)
vals = emb @ pos_a - emb @ neg_a
for (r, w), v in zip(positions, vals):
    curves[r, w] = v
m = curves.std(axis=1) > 1e-6
z = curves.copy()
z[m] = (curves[m] - curves[m].mean(axis=1, keepdims=True)) / curves[m].std(axis=1, keepdims=True)
norms = np.linalg.norm(z, axis=1, keepdims=True)
z = np.where(norms > 1e-6, z / (norms + 1e-12), 0)
np.save(WORK / "arc_curve.npy", z.astype(np.float32))
print(f"saved arc_curve.npy for {int(m.sum())}/{len(ids)} movies", flush=True)
