#!/usr/bin/env python3
"""Bulk DeepSeek profiling: threaded, resumable, balance-aware.

Usage: 23_profile_ds_bulk.py <outdir> <prompt_file> [--model deepseek-chat]
                             [--ids ids.json | --all] [--threads 12]

- skips existing outputs (resumable; safe to rerun anytime)
- stops gracefully if balance runs dry (prepaid = hard cap)
- prints progress + measured cost every 100 movies
"""
import argparse, json, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("prompt_file")
ap.add_argument("--model", default="deepseek-chat")
ap.add_argument("--ids", default=None, help="json list of imdb ids; default: pilot ids.json")
ap.add_argument("--all", action="store_true", help="every file in data/clean")
ap.add_argument("--threads", type=int, default=12)
args = ap.parse_args()

KEY = next((l.split("=", 1)[1].strip() for l in open(ROOT / ".env") if l.startswith("DEEPSEEK_API_KEY")), None)
if not KEY:
    sys.exit("DEEPSEEK_API_KEY missing from .env")
OUT = ROOT / args.outdir
OUT.mkdir(parents=True, exist_ok=True)
prompt = (ROOT / args.prompt_file).read_text().split("Subtitle file path:")[0]

if args.all:
    todo = sorted(int(p.stem) for p in (ROOT / "data" / "clean").glob("*.txt"))
else:
    todo = json.load(open(ROOT / (args.ids or "data/ids.json")))
todo = [i for i in todo if i < 90000000 and not (OUT / f"{i:07d}.json").exists()
        and (ROOT / "data" / "clean" / f"{i:07d}.txt").exists()]
print(f"{len(todo)} movies to profile with {args.model} -> {OUT}")

lock = threading.Lock()
state = {"done": 0, "fail": 0, "in_tok": 0, "out_tok": 0, "halt": False}

def call(i):
    if state["halt"]:
        return
    text = (ROOT / "data" / "clean" / f"{i:07d}.txt").read_text(errors="replace")[:60000]
    body = json.dumps({"model": args.model,
                       "messages": [{"role": "user", "content":
                           prompt + "\nThe subtitle text follows below. Output ONLY the JSON object.\n\n=== SUBTITLE TEXT ===\n" + text}],
                       "response_format": {"type": "json_object"},
                       "max_tokens": 1200}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions")
            req.add_header("Authorization", f"Bearer {KEY}")
            req.add_header("Content-Type", "application/json")
            req.data = body
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.load(r)
            content = d["choices"][0]["message"]["content"]
            json.loads(re.search(r"\{.*\}", content, re.S).group(0))  # validate
            (OUT / f"{i:07d}.json").write_text(content)
            with lock:
                state["done"] += 1
                u = d.get("usage", {})
                state["in_tok"] += u.get("prompt_tokens", 0)
                state["out_tok"] += u.get("completion_tokens", 0)
                if state["done"] % 100 == 0:
                    cost = state["in_tok"] / 1e6 * 0.14 + state["out_tok"] / 1e6 * 0.28
                    print(f"{state['done']}/{len(todo)}  measured cost so far ${cost:.2f}", flush=True)
            return
        except urllib.error.HTTPError as e:
            if e.code == 402:  # insufficient balance - hard stop, resumable later
                state["halt"] = True
                print("BALANCE EXHAUSTED - halting (rerun after top-up)", flush=True)
                return
            time.sleep(8 * (attempt + 1))
        except Exception:
            time.sleep(8 * (attempt + 1))
    with lock:
        state["fail"] += 1

with ThreadPoolExecutor(max_workers=args.threads) as ex:
    list(ex.map(call, todo))
cost = state["in_tok"] / 1e6 * 0.14 + state["out_tok"] / 1e6 * 0.28
print(f"done: {state['done']} ok, {state['fail']} failed, measured cost ${cost:.2f} "
      f"({state['in_tok']/1e6:.1f}M in / {state['out_tok']/1e6:.1f}M out)")
