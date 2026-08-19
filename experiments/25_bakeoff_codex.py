#!/usr/bin/env python3
"""Codex bakeoff: profile the same 50 movies with a given model+reasoning effort.
Usage: 25_bakeoff_codex.py <model> <effort>     e.g. gpt-5.6-sol low
Reference = existing gpt-5.5-low profiles (data/profiles). Saves swap matrix for genome eval."""
import json, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL, EFFORT = sys.argv[1], sys.argv[2]
TAG = MODEL.replace("gpt-", "").replace(".", "") + "-" + EFFORT
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
OUT = ROOT / "data" / f"profiles_{TAG}"
OUT.mkdir(exist_ok=True)
PROMPT = (ROOT / "profile_prompt.txt").read_text()

ids = json.load(open(ROOT / "data" / "ids.json"))
pool = [i for i in ids if i < 90000000 and (ROOT / "data" / "profiles" / f"{i:07d}.json").exists()]
random.Random(7).shuffle(pool)
chosen = pool[:50]

t0 = time.time()
done = []
def one(i):
    out = OUT / f"{i:07d}.json"
    if out.exists():
        return
    for attempt in range(2):
        r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
                            "-m", MODEL, "-c", f"model_reasoning_effort={EFFORT}",
                            "-o", str(out), PROMPT + f" {ROOT}/data/clean/{i:07d}.txt"],
                           capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
        if out.exists() and out.stat().st_size > 0:
            done.append(i)
            if len(done) % 10 == 0:
                print(f"{len(done)}/50", flush=True)
            return
        if "usage limit" in (r.stdout + r.stderr).lower():
            print("USAGE LIMIT HIT - stopping", flush=True)
            sys.exit(2)
    print(f"FAIL {i}", file=sys.stderr, flush=True)

with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(one, chosen))
print(f"profiling wall-time: {(time.time()-t0)/60:.1f} min")

from sentence_transformers import SentenceTransformer
import importlib.util
spec = importlib.util.spec_from_file_location("ee", ROOT / "03_embed_eval.py")
ee = importlib.util.module_from_spec(spec); spec.loader.exec_module(ee)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
W = json.load(open(ROOT / "data" / "weights.json"))
def load_card(p):
    raw = Path(p).read_text().strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    return json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
sims, new_emb = [], {f: {} for f in FACETS}
for i in chosen:
    try:
        ref, cand = load_card(ROOT / "data" / "profiles" / f"{i:07d}.json"), load_card(OUT / f"{i:07d}.json")
    except Exception:
        continue
    if not all(f in cand and f in ref for f in FACETS):
        continue
    tot = 0
    for f in FACETS:
        ea, eb = model.encode([ee.facet_text(ref, f), ee.facet_text(cand, f)], normalize_embeddings=True)
        tot += W[f] * float(ea @ eb)
        new_emb[f][i] = eb
    sims.append(tot)
print(f"facet agreement vs gpt-5.5-low: mean {np.mean(sims):.3f} (n={len(sims)}, self ~0.93)")
emb_s = {f: np.load(ROOT / f"data/emb_{f}.npy").copy() for f in FACETS}
idx = {i: k for k, i in enumerate(ids)}
for f in FACETS:
    for i, v in new_emb[f].items():
        emb_s[f][idx[i]] = v
np.save(ROOT / "data" / f"bakeoff_tmp_ds-swap_{TAG}.npy",
        sum(W[f] * (emb_s[f] @ emb_s[f].T) for f in FACETS))
print(f"saved bakeoff_tmp_ds-swap_{TAG}.npy")
