#!/usr/bin/env python3
"""Rating prediction (RMSE) on ml-25m, restricted to movies with subtitle profiles:
  1. bias      : mu + b_u + b_i
  2. mf        : + u.v  (k=64 latent factors, learned from ratings)
  3. hybrid    : + item factor = free vector + projection of subtitle-profile embedding
Reports overall RMSE + cold-item RMSE (items with <20 train ratings).
Runs on MPS if available. 90/10 random split, seed 42.
"""
import csv, json, time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
import sys as _sys
K, CDIM, EPOCHS, BATCH = 64, 256, int(_sys.argv[2]) if len(_sys.argv)>2 else 8, 8192
WD = float(_sys.argv[1]) if len(_sys.argv)>1 else 2e-6
KINDS = _sys.argv[3].split(",") if len(_sys.argv)>3 else ["bias","mf","hybrid"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}

# content matrix: concat 6 facets -> PCA to CDIM
print("content features...", flush=True)
C = np.concatenate([np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1)
C = C - C.mean(0, keepdims=True)
U_, S_, Vt_ = np.linalg.svd(C[np.random.choice(len(C), 4000, replace=False)], full_matrices=False)
P = Vt_[:CDIM].T                     # 4608 x CDIM
C = (C @ P).astype(np.float32)
C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

print("loading ratings...", flush=True)
u_map, i_map = {}, {}
rows_u, rows_i, rows_r = [], [], []
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        rows_u.append(u_map.setdefault(u, len(u_map)))
        rows_i.append(i_map.setdefault(imdb, len(i_map)))
        rows_r.append(float(r))
n_u, n_i, n = len(u_map), len(i_map), len(rows_r)
print(f"users={n_u} items={n_i} ratings={n}", flush=True)

content = np.zeros((n_i, CDIM), dtype=np.float32)
for imdb, row in i_map.items():
    content[row] = C[idx[imdb]]

uu = torch.tensor(rows_u); ii = torch.tensor(rows_i); rr = torch.tensor(rows_r)
perm = torch.randperm(n)
n_test = n // 10
test, train = perm[:n_test], perm[n_test:]
train_counts = np.bincount(ii[train].numpy(), minlength=n_i)
cold_items = torch.tensor(train_counts < 20)
cold_test = test[cold_items[ii[test]]]
print(f"train={len(train)} test={len(test)} cold-test={len(cold_test)}", flush=True)
mu = rr[train].mean().item()
content_t = torch.tensor(content, device=dev)

class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.bu = nn.Embedding(n_u, 1); self.bi = nn.Embedding(n_i, 1)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
        if kind != "bias":
            self.pu = nn.Embedding(n_u, K); self.qi = nn.Embedding(n_i, K)
            nn.init.normal_(self.pu.weight, 0, 0.05); nn.init.normal_(self.qi.weight, 0, 0.05)
        if kind == "hybrid":
            self.proj = nn.Linear(CDIM, K, bias=False)
            nn.init.normal_(self.proj.weight, 0, 0.05)
    def forward(self, u, i):
        p = mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1)
        if self.kind != "bias":
            q = self.qi(i)
            if self.kind == "hybrid":
                q = q + self.proj(content_t[i])
            p = p + (self.pu(u) * q).sum(1)
        return p

def rmse(model, subset):
    model.eval(); se = 0.0
    with torch.no_grad():
        for a in range(0, len(subset), 65536):
            b = subset[a:a+65536]
            pred = model(uu[b].to(dev), ii[b].to(dev))
            se += ((pred - rr[b].to(dev)) ** 2).sum().item()
    return (se / len(subset)) ** 0.5

results = {}
for kind in KINDS:
    model = Model(kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=WD)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        order = train[torch.randperm(len(train))]
        for a in range(0, len(order), BATCH):
            b = order[a:a+BATCH]
            opt.zero_grad()
            loss = ((model(uu[b].to(dev), ii[b].to(dev)) - rr[b].to(dev)) ** 2).mean()
            loss.backward(); opt.step()
        print(f"{kind} ep{ep+1} test-rmse={rmse(model, test):.4f} ({time.time()-t0:.0f}s)", flush=True)
    results[kind] = (rmse(model, test), rmse(model, cold_test) if len(cold_test) else float("nan"))

print(f"\n== FINAL wd={WD} epochs={EPOCHS} (test / cold) ==")
for kind, (r, c) in results.items():
    print(f"{kind:7s}: {r:.4f} / {c:.4f}")
