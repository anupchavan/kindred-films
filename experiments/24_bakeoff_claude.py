#!/usr/bin/env python3
"""Claude bakeoff: profile the same 50 movies with claude (headless -p mode),
same v1 prompt, compare vs gpt-5.5 references. Usage: 24_bakeoff_claude.py <model>
e.g. claude-opus-5 or claude-sonnet-5. Reports measured API-rate cost."""
import json, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1]
SHORT = MODEL.replace("claude-", "").replace("-5", "5")
FACETS = ["plot", "themes", "vibe", "arc", "texture", "world"]
OUT = ROOT / "data" / f"profiles_{SHORT}"
OUT.mkdir(exist_ok=True)
base_prompt = (ROOT / "profile_prompt.txt").read_text().split("Subtitle file path:")[0]

ids = json.load(open(ROOT / "data" / "ids.json"))
pool = [i for i in ids if i < 90000000 and (ROOT / "data" / "profiles" / f"{i:07d}.json").exists()]
random.Random(7).shuffle(pool)
chosen = pool[:50]

costs = []
def one(i):
    out = OUT / f"{i:07d}.json"
    if out.exists():
        return
    text = (ROOT / "data" / "clean" / f"{i:07d}.txt").read_text(errors="replace")[:60000]
    full = base_prompt + "\nThe subtitle text follows below. Output ONLY the JSON object.\n\n=== SUBTITLE TEXT ===\n" + text
    for attempt in range(3):
        try:
            r = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json",
                                "--disallowedTools", "*"],
                               input=full, capture_output=True, text=True, timeout=600)
            d = json.loads(r.stdout)
            content = d.get("result", "")
            json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            out.write_text(content)
            costs.append(d.get("total_cost_usd", 0))
            print(f"{len(costs)}/50", flush=True)
            return
        except Exception as e:
            if attempt == 2:
                print(f"FAIL {i}: {e}", file=sys.stderr, flush=True)

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(one, chosen))
print(f"\nmeasured API-rate cost for {len(costs)} calls: ${sum(costs):.2f} "
      f"(${np.mean(costs):.4f}/movie -> ${np.mean(costs)*19314:.0f} for 19k)")

# facet agreement + swap matrix (mirrors 21)
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
sims, cl_emb = [], {f: {} for f in FACETS}
for i in chosen:
    try:
        ref = load_card(ROOT / "data" / "profiles" / f"{i:07d}.json")
        cl = load_card(OUT / f"{i:07d}.json")
    except Exception:
        continue
    if not all(f in cl and f in ref for f in FACETS):
        print(f"  skipping {i}: incomplete card", file=sys.stderr)
        continue
    tot = 0
    for f in FACETS:
        ea, eb = model.encode([ee.facet_text(ref, f), ee.facet_text(cl, f)], normalize_embeddings=True)
        tot += W[f] * float(ea @ eb)
        cl_emb[f][i] = eb
    sims.append(tot)
print(f"facet agreement vs gpt-5.5: mean {np.mean(sims):.3f} median {np.median(sims):.3f} "
      f"min {np.min(sims):.3f} (self-agreement ~0.93, flash 0.834, pro 0.851)")

emb_s = {f: np.load(ROOT / f"data/emb_{f}.npy").copy() for f in FACETS}
idx = {i: k for k, i in enumerate(ids)}
for f in FACETS:
    for i, v in cl_emb[f].items():
        emb_s[f][idx[i]] = v
np.save(ROOT / "data" / f"bakeoff_tmp_ds-swap_{SHORT}.npy",
        sum(W[f] * (emb_s[f] @ emb_s[f].T) for f in FACETS))
print(f"saved bakeoff_tmp_ds-swap_{SHORT}.npy")
