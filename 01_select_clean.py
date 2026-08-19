#!/usr/bin/env python3
"""Select pilot movies (curated clusters + random distractors) from Sublens-20M,
validate and clean their subtitle files, emit clean text + manifest + eval clusters."""
import csv, json, os, random, re, sys
from pathlib import Path

DATA = Path(os.environ.get("SUBLENS_DIR", "Sublens_20M"))
OUT = Path(__file__).parent / "data"
CLEAN = OUT / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 500

# Similarity clusters: cluster name -> list of title substrings (lowercase, with year to disambiguate).
# These double as the eval set: for an anchor, other members are "expected" recommendations.
CLUSTERS = {
    "cerebral_space": ["interstellar (2014", "gravity (2013", "contact (1997", "2001: a space odyssey", "apollo 13", "moon (2009", "sunshine (2007", "europa report"],
    "zombie": ["28 days later", "28 weeks later", "shaun of the dead", "zombieland", "world war z", "dawn of the dead (1978", "dawn of the dead (2004", "night of the living dead (1968"],
    "slasher": ["halloween (1978", "nightmare on elm street, a (1984", "friday the 13th (1980", "scream (1996", "texas chainsaw massacre, the (1974"],
    "supernatural_horror": ["exorcist, the (1973", "shining, the (1980", "conjuring, the (2013", "insidious (2010", "sinister (2012", "paranormal activity (2007", "ring, the (2002"],
    "romcom": ["notting hill", "when harry met sally", "love actually", "pretty woman", "10 things i hate about you", "bridget jones's diary", "(500) days of summer"],
    "mob": ["godfather, the (1972", "godfather: part ii", "goodfellas", "casino (1995", "departed, the (2006", "donnie brasco", "scarface (1983"],
    "heist": ["ocean's eleven (a.k.a. ocean's 11) (2001", "italian job, the (2003", "inside man", "heat (1995", "town, the (2010", "snatch (2000"],
    "mind_bender": ["inception (2010", "memento", "prestige, the (2006", "shutter island", "source code", "donnie darko (2001", "usual suspects", "fight club"],
    "pixar_family": ["toy story (1995", "toy story 3", "finding nemo", "up (2009", "monsters, inc", "wall·e", "wall-e", "ratatouille", "incredibles, the (2004"],
    "war": ["saving private ryan", "platoon (1986", "full metal jacket", "thin red line", "black hawk down", "hurt locker"],
    "scifi_horror": ["alien (1979", "aliens (1986", "thing, the (1982", "predator (1987", "event horizon", "pitch black"],
    "ai_scifi": ["matrix, the (1999", "blade runner (1982", "terminator, the (1984", "terminator 2", "her (2013", "a.i. artificial intelligence", "i, robot", "minority report"],
    "time_travel": ["back to the future (1985", "groundhog day", "looper", "edge of tomorrow", "twelve monkeys (a.k.a. 12 monkeys", "butterfly effect, the (2004", "primer (2004"],
    "fantasy_epic": ["lord of the rings: the fellowship", "lord of the rings: the two towers", "lord of the rings: the return", "harry potter and the sorcerer's stone", "harry potter and the chamber", "hobbit: an unexpected journey", "chronicles of narnia: the lion", "stardust (2007"],
    "space_opera": ["star wars: episode iv", "star wars: episode v", "star wars: episode vi", "star trek (2009", "guardians of the galaxy (2014", "serenity (2005", "fifth element, the"],
    "superhero": ["dark knight, the (2008", "batman begins", "iron man (2008", "avengers, the (2012", "spider-man (2002", "x-men (2000"],
    "action_90s": ["die hard (1988", "speed (1994", "con air", "face/off", "rock, the (1996", "lethal weapon (1987"],
    "spy_action": ["bourne identity, the (2002", "casino royale (2006", "skyfall", "mission: impossible (1996", "mission: impossible - ghost protocol", "taken (2008", "john wick"],
    "weepy_romance": ["notebook, the (2004", "titanic (1997", "walk to remember, a", "dear john (2010", "fault in our stars, the"],
    "serial_killer": ["silence of the lambs", "seven (a.k.a. se7en", "zodiac (2007", "gone girl", "prisoners (2013", "girl with the dragon tattoo, the (2011"],
    "courtroom": ["few good men, a", "12 angry men (1957", "my cousin vinny", "philadelphia (1993", "lincoln lawyer, the"],
    "sports_underdog": ["rocky (1976", "rudy (1993", "remember the titans", "miracle (2004", "moneyball", "warrior (2011"],
    "teen_comedy": ["american pie (1999", "superbad", "mean girls (2004", "clueless (1995", "easy a (2010"],
    "survival": ["cast away", "127 hours", "life of pi", "into the wild", "grey, the (2011", "all is lost"],
    "dystopia": ["hunger games, the (2012", "divergent (2014", "maze runner, the (2014", "children of men", "v for vendetta", "equilibrium (2002"],
    "bollywood": ["3 idiots", "pk (2014", "lagaan", "taare zameen par", "rang de basanti", "zindagi na milegi dobara", "dilwale dulhania", "munna bhai m.b.b.s.", "sholay"],
}

AD_RE = re.compile(r"(?i)opensubtitles|subtitles?\s+(?:by|downloaded)|subs?\s+by|www\.|https?://|\.com|\.org|\.net\b|corrected by|sync(?:ed|hronized)?\s+(?:by|&)|encoded by|rip(?:ped)? by|resync|caption(?:s|ed) by|osdb\b|allsubs|podnapisi|addic7ed|sdimedia|explosiveskull|honeybunny|titlovi|advertise your product|remove all ads|support us and become")
TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
STOPWORDS = set("the a an and or but if of to in on at for with is are was were be been i you he she it we they this that not no what who all my your his her me him them do does did have has had will would can could there so as from by out up down about just like know get got go going come want said say says one two think see look time now then when where why how".split())

def parse_srt(path):
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    cues, dur = [], 0.0
    for block in re.split(r"\n\s*\n", text):
        m = TS_RE.search(block)
        if not m:
            continue
        end_s = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7))
        dur = max(dur, end_s)
        lines = block[m.end():].strip().splitlines()
        cue = " ".join(TAG_RE.sub("", l).strip() for l in lines).strip()
        cue = re.sub(r"\s+", " ", cue)
        if cue and not AD_RE.search(cue):
            cues.append(cue)
    return cues, dur

def english_ratio(cues):
    words = [w for c in cues[:400] for w in re.findall(r"[a-z']+", c.lower())]
    if not words:
        return 0.0
    return sum(w in STOPWORDS for w in words) / len(words)

def validate(imdb_id):
    """Return (clean_cues, n_words, dur) or (None, reason, 0)."""
    d = DATA / "subtitles" / f"{imdb_id:07d}"
    srts = sorted(d.glob("*.srt")) if d.is_dir() else []
    if not srts:
        return None, "no_srt", 0
    best = None
    for srt in srts:
        parsed = parse_srt(srt)
        if not parsed:
            continue
        cues, dur = parsed
        if best is None or len(cues) > len(best[0]):
            best = (cues, dur)
    if best is None:
        return None, "unparseable", 0
    cues, dur = best
    n_words = sum(len(c.split()) for c in cues)
    if len(cues) < 100 or n_words < 600:
        return None, f"too_short({len(cues)}c/{n_words}w)", 0
    if english_ratio(cues) < 0.22:
        return None, "not_english", 0
    return (cues, dur), n_words, dur

def main():
    movies = {}   # mlid -> title
    with open(DATA / "Sublens_20M_metadata" / "movies_subs_filtered.csv", newline="") as f:
        for row in csv.DictReader(f):
            movies[int(row["movielens_id"])] = row["movie_name"]
    imdb = {}     # mlid -> imdb_id
    with open(DATA / "Sublens_20M_metadata" / "links_subs_filtered.csv", newline="") as f:
        for row in csv.DictReader(f):
            imdb[int(row["movielens_id"])] = int(row["imdb_id"])

    title_index = {mlid: t.lower() for mlid, t in movies.items()}
    selected = {}   # mlid -> cluster name
    missing = []
    for cluster, patterns in CLUSTERS.items():
        for pat in patterns:
            hits = [mlid for mlid, t in title_index.items() if pat in t]
            if not hits:
                missing.append(f"{cluster}: {pat}")
                continue
            mlid = min(hits)
            if mlid not in selected:
                selected[mlid] = cluster

    rng = random.Random(42)
    pool = [m for m in movies if m not in selected and m in imdb]
    rng.shuffle(pool)

    manifest, clusters_out = [], {}
    def process(mlid, cluster):
        iid = imdb[mlid]
        result, info, dur = validate(iid)
        if result is None:
            manifest.append((iid, mlid, movies[mlid], cluster, 0, 0, info))
            return False
        cues, _ = result
        (CLEAN / f"{iid:07d}.txt").write_text("\n".join(cues))
        manifest.append((iid, mlid, movies[mlid], cluster, len(cues), info, "ok"))
        clusters_out.setdefault(cluster, []).append(iid) if cluster != "distractor" else None
        return True

    n_ok = sum(process(mlid, cl) for mlid, cl in selected.items())
    for mlid in pool:
        if n_ok >= TARGET_TOTAL:
            break
        n_ok += process(mlid, "distractor")

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imdb_id", "movielens_id", "title", "cluster", "cues", "words", "status"])
        w.writerows(manifest)
    (OUT / "eval_clusters.json").write_text(json.dumps(clusters_out, indent=1))

    ok = [m for m in manifest if m[6] == "ok"]
    print(f"selected: {len(manifest)} attempted, {len(ok)} clean")
    print(f"cluster movies found: {len(selected)}; patterns missing from dataset: {len(missing)}")
    for m in missing:
        print(f"  MISSING {m}")
    fails = [m for m in manifest if m[6] != "ok" and m[3] != "distractor"]
    for m in fails:
        print(f"  FAIL {m[2]} [{m[3]}]: {m[6]}")

if __name__ == "__main__":
    main()
