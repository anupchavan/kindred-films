#!/usr/bin/env python3
"""Leakage experiment: does the profiler 'fake' reading subtitles for movies it knows?

Arms (synthetic ids, flagged 'experiment' in manifest):
  90000001 KISS       - Varun Grover short, as-is (obscure: LLM prior unlikely)
  90000002 THEATRE    - same text, Indian names -> American names
  90000003 HARVEST    - Interstellar's clean text, all identifying names swapped
                        (famous: if the LLM uses its prior, arms diverge here)
Compare per-facet cosine within each pair + neighbor lists after embedding.
"""
import csv, importlib.util, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sc", ROOT / "01_select_clean.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

KISS_SWAPS = [("Salil", "Gerald"), ("Chauhan", "Sullivan"), ("Banerjee", "Callahan"),
              ("Abhay", "Nathan"), ("Rehan", "Tyler")]
INTERSTELLAR_SWAPS = [  # longest-first so e.g. Murphy wins over Murph
    ("Murphy", "Riley"), ("Murph", "Riley"), ("Cooper", "Dawson"), ("Coop", "Daws"),
    ("Amelia", "Grace"), ("Brand", "Keller"), ("Romilly", "Whitman"), ("Doyle", "Mercer"),
    ("Edmunds", "Blake"), ("Gargantua", "Leviathan"), ("Lazarus", "Genesis"),
    ("Miller", "Frost"), ("Mann", "Holt"), ("TARS", "VEX"), ("CASE", "ATLAS"),
    ("KIPP", "ECHO"), ("Getty", "Weiss"), ("Donald", "Harold"), ("Lois", "Kate"),
]

def swap(text, pairs):
    for a, b in pairs:
        text = re.sub(rf"\b{a}\b", b, text)
        text = re.sub(rf"\b{a.upper()}\b", b.upper(), text)
    return text

def add(iid, title, cues_text):
    p = sc.CLEAN / f"{iid:07d}.txt"
    p.write_text(cues_text)
    with open(sc.OUT / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerow([iid, "", title, "experiment",
                                cues_text.count("\n") + 1, len(cues_text.split()), "ok"])
    print(f"{title}: {cues_text.count(chr(10))+1} cues written")

# KISS arms from the webvtt
parsed = sc.parse_srt(Path(sys.argv[1] if len(sys.argv) > 1 else "kiss.webvtt"))
if not parsed:
    sys.exit("could not parse kiss.webvtt")
cues, _ = parsed
kiss_text = "\n".join(cues)
add(90000001, "KISS (experiment: as-is)", kiss_text)
add(90000002, "THEATRE (experiment: names swapped)", swap(kiss_text, KISS_SWAPS))

# Interstellar control arm (real one is already indexed as 816692)
inter = (sc.CLEAN / "0816692.txt").read_text()
add(90000003, "HARVEST (experiment: Interstellar renamed)", swap(inter, INTERSTELLAR_SWAPS))
print("\nnow run: ./02_profile.sh && .venv/bin/python 06_add_movies.py embed && .venv/bin/python 11_compare_arms.py")
