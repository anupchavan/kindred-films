# Kindred

Kindred finds films that are like a film you name, using only that film's subtitles. By default it doesn't use genres, tags or ratings to rank. The idea is that the dialogue already carries most of what makes two films feel alike, including their story, tone and pace.

```
$ python 05_recommend.py "inception" 10 --feel
```

## How it works

1. **Clean.** Each subtitle file becomes plain dialogue, one line per cue. Ads, timestamps and markup are removed, and files that are too short or not in English are dropped (`01_select_clean.py`).
2. **Profile.** An LLM reads the dialogue and writes a short profile with six parts: *plot*, *themes*, *vibe*, *arc*, *texture* and *world*. The prompt in `profile_prompt.txt` tells it to describe only what the text shows and not to name the film (`02_profile.sh`).
3. **Embed.** Each of the six parts is embedded on its own with BGE-base, giving six 768-dimensional vectors per film (`03_embed_eval.py`).
4. **Recommend.** For a given film, Kindred computes cosine similarity against every other film for each part, then takes a weighted sum (`05_recommend.py`). The default weights are in `data/weights.json` and were tuned against the MovieLens Tag Genome.

The current index holds 16,284 films. All the LLM work happens once, when a film is added. A query is six matrix-vector products, so it needs no LLM call, no GPU and no retraining.

## Usage

```
python 05_recommend.py "title" [k]           # top k similar films (default 10)
python 05_recommend.py "aftersun" --feel     # favour vibe, arc and texture
python 05_recommend.py "heat" --story        # favour plot and themes
python 05_recommend.py "up" --vibe 3         # scale any part's weight
python 05_recommend.py "x" --curve 0.2       # rerank by emotional arc (see below)
python 05_recommend.py "x" --cf 0.4          # blend in MovieLens rating factors
python 05_recommend.py "x" --by-rating       # sort the results by IMDb rating
```

The output shows each result's overall score next to its score for each part, so you can see why a film was picked.

Two optional signals can be added on top of the default ranking:

- **Arc curve** (`31_arc_curves.py`): the dialogue is split into 20 windows, and each window gets a light-to-dark score from embeddings, with no LLM. `--curve` reorders the top 200 matches by how closely their curves follow the same shape.
- **Ratings** (`experiments/35_cf_factors.py`): item factors from an SVD of MovieLens 25M. `--cf` blends these in for films that have ratings.

## Adding a film

```
python 07_add_online.py "Past Lives" --year 2023
```

This downloads the best English subtitle from OpenSubtitles, cleans it, profiles it and appends it to the index. Existing films are not touched. It needs `OPENSUBTITLES_API_KEY`, `OPENSUBTITLES_USERNAME` and `OPENSUBTITLES_PASSWORD` in a `.env` file. A `TMDB_API_KEY` is optional and only used to resolve titles. Profiling calls the Codex CLI.

## Results

| Test | Result |
|---|---|
| Agreement with human tag similarity (MovieLens Tag Genome): median rank of Kindred's top pick among 432 films | **13** (random: 216) |
| Same test for an LDA topic model on the same subtitles | 163 |
| Rating prediction for films with fewer than 20 ratings, compared with matrix factorisation alone | RMSE 0.025 lower |
| Whether subtitle taste explains what rating models get wrong (AUC) | 0.52, close to chance |

Similarity works well. Personalisation does not: the subtitles say little about a viewer's taste that rating data doesn't already capture. The scripts behind these numbers are in `experiments/`.

## Limits

- Films with little dialogue are profiled poorly. A quiet film with a dark undertone can read as calm.
- What a director's films share is often not in the dialogue, so Kindred can miss it.
- Profiles written by different models don't compare well with each other, so the whole index should be profiled by one model.

## Layout

```
01–07, 31, 33   numbered pipeline scripts: clean, profile, embed, recommend, add films, arc curves, IMDb ratings
curves_gpu.py   arc curves for the whole index on a GPU
data/           facet weights and the evaluation clusters
experiments/    evaluations, model comparisons and rating-based tests
```

The index itself (embeddings, profiles and cleaned subtitles) is not in this repository, because it is large and built from subtitle files that can't be redistributed. Subtitles came from the Sublens-20M corpus and OpenSubtitles. Set `SUBLENS_DIR` to rebuild from Sublens, and `MOVIELENS_DIR` to point the experiments at MovieLens 25M.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
