#!/usr/bin/env python3
"""CF-residual test: do subtitles know something about preference that CF gets wrong?

Train MF (bias + k=64 factors) on 90% of ratings; on the 10% holdout compute
residuals e = r - r_hat. Question: does the user's subtitle taste-score for the
item predict the SIGN of e?  AUC(sub_score -> e>0).
  AUC ~0.5 -> subtitles are a weaker copy of CF; stop forcing personalization.
  AUC meaningfully >0.5 -> subtitles know things about preference CF gets wrong.
"""
import csv, json
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("MOVIELENS_DIR", "ml-25m"))
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
K, EPOCHS, BATCH = 64, 8, 8192
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

ids = json.load(open(ROOT / "data" / "ids.json"))
idx = {i: k for k, i in enumerate(ids)}
W = json.load(open(ROOT / "data" / "weights.json"))
content = np.concatenate(
    [np.sqrt(W[f]) * np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS], axis=1).astype(np.float32)

imdb2ml = {int(r["imdbId"]): int(r["movieId"]) for r in csv.DictReader(open(SCRATCH / "links.csv")) if r["imdbId"]}
ml2imdb = {v: k for k, v in imdb2ml.items()}
print("loading ratings...", flush=True)
u_map, i_map = {}, {}
uu_, ii_, rr_ = [], [], []
with open(SCRATCH / "ratings.csv") as f:
    next(f)
    for line in f:
        u, m, r, _ = line.rstrip("\n").split(",")
        imdb = ml2imdb.get(int(m))
        if imdb is None or imdb not in idx:
            continue
        uu_.append(u_map.setdefault(u, len(u_map)))
        ii_.append(i_map.setdefault(idx[imdb], len(i_map)))
        rr_.append(float(r))
n_u, n_i, n = len(u_map), len(i_map), len(rr_)
row_of = {}   # model item id -> index row
for row, iid in i_map.items():
    row_of[iid] = row
uu = torch.tensor(uu_); ii = torch.tensor(ii_); rr = torch.tensor(rr_)
perm = torch.randperm(n)
test, train = perm[:n // 10], perm[n // 10:]
mu = rr[train].mean().item()
print(f"users={n_u} items={n_i} ratings={n}", flush=True)

class MF(nn.Module):
    def __init__(self):
        super().__init__()
        self.bu = nn.Embedding(n_u, 1); self.bi = nn.Embedding(n_i, 1)
        self.pu = nn.Embedding(n_u, K); self.qi = nn.Embedding(n_i, K)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
        nn.init.normal_(self.pu.weight, 0, 0.05); nn.init.normal_(self.qi.weight, 0, 0.05)
    def forward(self, u, i):
        return mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + (self.pu(u) * self.qi(i)).sum(1)

model = MF().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=2e-6)
for ep in range(EPOCHS):
    model.train()
    order = train[torch.randperm(len(train))]
    for a in range(0, len(order), BATCH):
        b = order[a:a+BATCH]
        opt.zero_grad()
        loss = ((model(uu[b].to(dev), ii[b].to(dev)) - rr[b].to(dev)) ** 2).mean()
        loss.backward(); opt.step()
    print(f"ep{ep+1} done", flush=True)

model.eval()
resid = np.zeros(len(test), dtype=np.float32)
with torch.no_grad():
    for a in range(0, len(test), 65536):
        b = test[a:a+65536]
        pred = model(uu[b].to(dev), ii[b].to(dev)).cpu().numpy()
        resid[a:a+len(b)] = rr[b].numpy() - pred
print(f"holdout={len(test)}, mean|e|={np.abs(resid).mean():.3f}", flush=True)

# subtitle taste profiles from TRAIN loved ratings
print("building taste profiles...", flush=True)
train_loved = defaultdict(list)
tr = set(train.tolist())
for k_ in train.tolist():
    if rr_[k_] >= 4.0:
        train_loved[uu_[k_]].append(row_of[ii_[k_]])
# score each holdout rating whose user has >=3 loved train movies
scores, esign, keep = [], [], 0
test_list = test.tolist()
for j, k_ in enumerate(test_list):
    u, i = uu_[k_], ii_[k_]
    taste = train_loved.get(u)
    if not taste or len(taste) < 3:
        continue
    t = taste[:10]
    s = float((content[row_of[i]] @ content[t].T).max())
    scores.append(s); esign.append(1 if resid[j] > 0 else 0)
    keep += 1
    if keep >= 300000:
        break
scores = np.array(scores); esign = np.array(esign)
# AUC via rank formula
order = np.argsort(scores)
rank = np.empty_like(order, dtype=np.float64); rank[order] = np.arange(1, len(scores) + 1)
n1, n0 = esign.sum(), (1 - esign).sum()
auc = (rank[esign == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print(f"\n== CF-residual sign prediction (n={len(scores)}) ==")
print(f"P(e>0)={esign.mean():.3f}")
print(f"AUC(subtitle taste score -> CF residual sign) = {auc:.4f}")
print("0.50 = subtitles add nothing CF doesn't know; >0.53 = real complementary signal")
