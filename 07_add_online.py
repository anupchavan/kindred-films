#!/usr/bin/env python3
"""Fetch the best subtitle for a movie from opensubtitles.com and add it to the index.

Usage: 07_add_online.py "Project Hail Mary"        # search by title
       07_add_online.py tt9214772                   # or by IMDb id
       07_add_online.py "Dune" --year 2021

Credentials come from .env in this directory (not committed anywhere):
    OPENSUBTITLES_API_KEY=...
    OPENSUBTITLES_USERNAME=...
    OPENSUBTITLES_PASSWORD=...
Selection: English, non-AI-translated, ordered by trusted-uploader then download count
— the same file a human would pick. Then: clean -> profile (gpt-5.5) -> embed -> append.
"""
import argparse, csv, importlib.util, json, re, subprocess, sys, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
API = "https://api.opensubtitles.com/api/v1"
UA = "kindred v0.1"

def env():
    e = {}
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                e[k.strip()] = v.strip()
    import os
    for k in ("OPENSUBTITLES_API_KEY", "OPENSUBTITLES_USERNAME", "OPENSUBTITLES_PASSWORD"):
        e.setdefault(k, os.environ.get(k, ""))
    if not e["OPENSUBTITLES_API_KEY"]:
        sys.exit("missing OPENSUBTITLES_API_KEY (put it in .env)")
    return e

def call(path, key, token=None, payload=None):
    # the API 301-redirects any non-canonical URL: params must be alphabetical, lowercase, '+' for spaces
    if "?" in path:
        base, qs = path.split("?", 1)
        pairs = sorted(p.split("=", 1) for p in qs.split("&"))
        path = base + "?" + "&".join(f"{k}={v}" for k, v in pairs)
    req = urllib.request.Request(API + path)
    req.add_header("Api-Key", key)
    req.add_header("User-Agent", UA)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req) as r:
        body = r.read()
    if body.lstrip()[:1] not in (b"{", b"["):
        if b"airtel" in body.lower() or b"court-order" in body.lower():
            sys.exit("your ISP served its court-order block page instead of the API response "
                     "- turn on your VPN and rerun")
        sys.exit(f"API returned non-JSON (HTTP {r.status}): {body[:200]!r}")
    return json.loads(body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--year", type=int)
    ap.add_argument("--no-post", action="store_true",
                    help="skip profile+embed (for batch runs; run 02_profile.sh and 06 embed once at the end)")
    ap.add_argument("--director", help="disambiguate title search by director name (resolved via TMDB)")
    args = ap.parse_args()
    e = env()
    key = e["OPENSUBTITLES_API_KEY"]

    # resolve to an imdb id
    m = re.fullmatch(r"tt(\d+)", args.movie)
    if m:
        imdb_id, title = int(m.group(1)), args.movie
    elif e.get("TMDB_API_KEY"):
        # TMDB resolution: fuzzy title search (+optional director filter), imdb id out.
        # Ingestion-only - TMDB never influences ranking.
        def tmdb(path):
            sep = "&" if "?" in path else "?"
            with urllib.request.urlopen(f"https://api.themoviedb.org/3{path}{sep}api_key={e['TMDB_API_KEY']}") as r:
                return json.load(r)
        q = f"/search/movie?query={urllib.parse.quote_plus(args.movie)}"
        if args.year:
            q += f"&year={args.year}"
        results = tmdb(q).get("results", [])[:5]
        if not results:
            sys.exit(f"TMDB found nothing for '{args.movie}'")
        pick = None
        if args.director:
            want = args.director.lower()
            for c in results:
                crew = tmdb(f"/movie/{c['id']}/credits").get("crew", [])
                if any(want in p["name"].lower() for p in crew if p["job"] == "Director"):
                    pick = c
                    break
            if pick is None:
                sys.exit(f"none of the top matches for '{args.movie}' are directed by {args.director}:\n  "
                         + "\n  ".join(f"{c['title']} ({(c.get('release_date') or '????')[:4]})" for c in results))
        else:
            pick = results[0]
            if len(results) > 1:
                print("candidates: " + "; ".join(
                    f"{c['title']} ({(c.get('release_date') or '????')[:4]})" for c in results[:3]))
        ext = None
        for c in ([pick] if args.director else results):
            ext = tmdb(f"/movie/{c['id']}/external_ids").get("imdb_id")
            if ext:
                pick = c
                break
        if not ext:
            sys.exit(f"TMDB has no imdb id for any candidate of '{args.movie}'")
        imdb_id, title = int(ext[2:]), f"{pick['title']} ({(pick.get('release_date') or '????')[:4]})"
        print(f"resolved: {title}  [tt{imdb_id:07d}]")
    else:
        q = f"/features?query={urllib.parse.quote_plus(args.movie.lower())}&type=movie"
        if args.year:
            q += f"&year={args.year}"
        feats = [f for f in call(q, key)["data"] if f["attributes"].get("imdb_id")]
        if not feats:
            sys.exit(f"no movie found for '{args.movie}'")
        f = feats[0]["attributes"]
        imdb_id, title = int(f["imdb_id"]), f'{f["title"]} ({f.get("year", "?")})'
        print(f"resolved: {title}  [tt{imdb_id:07d}]")

    clean_path = ROOT / "data" / "clean" / f"{imdb_id:07d}.txt"
    if clean_path.exists():
        sys.exit("already in the pipeline — nothing to do")

    # find best subtitle file
    subs = call(f"/subtitles?imdb_id={imdb_id}&languages=en&type=movie"
                "&order_by=download_count&order_direction=desc", key)["data"]
    cands = [s["attributes"] for s in subs
             if not s["attributes"].get("ai_translated")
             and not s["attributes"].get("machine_translated")
             and s["attributes"].get("files")]
    if not cands:
        sys.exit("no usable English subtitle found (yet) for this movie")
    cands.sort(key=lambda a: (a.get("from_trusted") or False, a.get("download_count") or 0), reverse=True)
    best = cands[0]
    fd = best.get("feature_details") or {}
    if fd.get("title") and title.startswith("tt"):
        title = f'{fd["title"]} ({fd.get("year", "?")})'
    print(f"picked: {best['files'][0].get('file_name')} "
          f"(downloads={best.get('download_count')}, trusted={best.get('from_trusted')})")

    # download (uses your daily quota)
    token = call("/login", key, payload={"username": e["OPENSUBTITLES_USERNAME"],
                                         "password": e["OPENSUBTITLES_PASSWORD"]})["token"]
    dl = call("/download", key, token, {"file_id": best["files"][0]["file_id"]})
    print(f"quota remaining today: {dl.get('remaining')}")
    raw_dir = ROOT / "data" / "raw_online"
    raw_dir.mkdir(exist_ok=True)
    srt = raw_dir / f"{imdb_id:07d}.srt"
    req = urllib.request.Request(dl["link"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r:
        srt.write_bytes(r.read())

    # clean using the same gate as the offline pipeline
    spec = importlib.util.spec_from_file_location("sc", ROOT / "01_select_clean.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    cues, dur = sc.parse_srt(srt)
    n_words = sum(len(c.split()) for c in cues)
    if len(cues) < 100 or n_words < 600 or sc.english_ratio(cues) < 0.22:
        sys.exit(f"downloaded file failed validation ({len(cues)} cues, {n_words} words)")
    clean_path.write_text("\n".join(cues))
    with open(ROOT / "data" / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerow([imdb_id, "", title, "online", len(cues), n_words, "ok"])
    print(f"cleaned: {len(cues)} cues, {n_words} words")

    # profile + embed + append
    if args.no_post:
        return
    subprocess.run([str(ROOT / "02_profile.sh")], check=True)
    subprocess.run([sys.executable, str(ROOT / "06_add_movies.py"), "embed"], check=True)
    print(f"\n{title} is now in the index — try: 05_recommend.py \"{args.movie}\"")

if __name__ == "__main__":
    main()
