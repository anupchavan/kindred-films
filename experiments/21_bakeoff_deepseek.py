#!/usr/bin/env python3
"""DeepSeek quality bakeoff: profile 50 movies with deepseek-chat using the v1 prompt,
compare against our existing gpt-5.5 profiles.

Metrics:
  1. Per-movie weighted facet-cosine DS-vs-gpt5.5 (bar: 0.93 = gpt-5.5 self-agreement)
  2. Tag-Genome: median GT-rank of top-1 rec for the 50 anchors, original vs DS-swapped
Run: 21_bakeoff_deepseek.py [n_movies]
"""
import json, random, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek-chat"
SUFFIX = "" if MODEL == "deepseek-chat" else "_" + MODEL.replace("deepseek-", "")
KEY = next((l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("DEEPSEEK_API_KEY")), None)
if not KEY:
    sys.exit("add DEEPSEEK_API_KEY to .env first")

base_prompt = (ROOT / "profile_prompt.txt").read_text().split("Subtitle file path:")[0]
OUT = ROOT / "data" / ("profiles_ds" + SUFFIX)
OUT.mkdir(exist_ok=True)

def deepseek(text):
    req = urllib.request.Request("https://api.deepseek.com/chat/completions")
    req.add_header("Authorization", f"Bearer {KEY}")
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content":
            base_prompt + "\nThe subtitle text follows below. Output ONLY the JSON object.\n\n=== SUBTITLE TEXT ===\n" + text}],
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
    }).encode()
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]

ids = json.load(open(ROOT / "data" / "ids.json"))
# pick N movies that have gpt-5.5 profiles (prefer real movies, skip experiment arms)
pool = [i for i in ids if i < 90000000 and (ROOT / "data" / "profiles" / f"{i:07d}.json").exists()]
random.Random(7).shuffle(pool)
chosen = pool[:N]

def load_card(p):
    raw = Path(p).read_text().strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    return json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))

for k, i in enumerate(chosen, 1):
    out = OUT / f"{i:07d}.json"
    if out.exists():
        continue
    text = (ROOT / "data" / "clean" / f"{i:07d}.txt").read_text(errors="replace")[:60000]
    for attempt in range(3):
        try:
            out.write_text(deepseek(text))
            break
        except Exception as e:
            print(f"  retry {i}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    print(f"{k}/{N}", flush=True)

# ---- metric 1: facet-cosine vs gpt-5.5 references ----
from sentence_transformers import SentenceTransformer
import importlib.util
spec = importlib.util.spec_from_file_location("ee", ROOT / "03_embed_eval.py")
ee = importlib.util.module_from_spec(spec); spec.loader.exec_module(ee)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
W = json.load(open(ROOT / "data" / "weights.json"))

sims, ds_emb = [], {f: {} for f in FACETS}
for i in chosen:
    try:
        ref = load_card(ROOT / "data" / "profiles" / f"{i:07d}.json")
        ds = load_card(OUT / f"{i:07d}.json")
    except Exception:
        continue
    tot = 0
    for f in FACETS:
        ea, eb = model.encode([ee.facet_text(ref, f), ee.facet_text(ds, f)], normalize_embeddings=True)
        tot += W[f] * float(ea @ eb)
        ds_emb[f][i] = eb
    sims.append((tot, i))
vals = [s for s, _ in sims]
print(f"\n== DS vs gpt-5.5 facet agreement (n={len(vals)}) ==")
print(f"mean {np.mean(vals):.3f}  median {np.median(vals):.3f}  min {np.min(vals):.3f}   (gpt-5.5 self-agreement ~0.93)")

# ---- metric 2: tag-genome, original vs DS-swapped, anchors = the 50 ----
emb_o = {f: np.load(ROOT / f"data/emb_{f}.npy") for f in FACETS}
emb_s = {f: e.copy() for f, e in emb_o.items()}
idx = {i: k for k, i in enumerate(ids)}
for f in FACETS:
    for i, v in ds_emb[f].items():
        emb_s[f][idx[i]] = v
for tag, emb in [("orig", emb_o), ("ds-swap", emb_s)]:
    np.save(ROOT / "data" / f"bakeoff_tmp_{tag}{SUFFIX}.npy", sum(W[f] * (emb[f] @ emb[f].T) for f in FACETS))
print("\nsaved bakeoff_tmp_{orig,ds-swap}.npy - run 22_bakeoff_genome.py for GT metrics")
json.dump([i for _, i in sims], open(ROOT / "data" / "bakeoff_ids.json", "w"))
