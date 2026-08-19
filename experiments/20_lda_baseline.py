#!/usr/bin/env python3
"""LDA baseline (Bougiatiotis & Giannakopoulos 2018 style): K=55 topics over the
pilot corpus, topic-distribution vectors saved as data/emb_lda.npy (aligned to
ids.json) so 19_tag_genome_eval.py can score it: 19_tag_genome_eval.py data/emb_ lda
"""
import json
from pathlib import Path
import numpy as np
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

ROOT = Path(__file__).resolve().parent.parent
ids = json.load(open(ROOT / "data" / "ids.json"))

EXTRA_STOP = set("aint ill theres yeah yeap yes gonna gotta wanna hey uh um oh ah huh okay ok right dont didnt cant wont im ive youre hes shes were theyre".split())
docs = []
for i in ids:
    text = (ROOT / "data" / "clean" / f"{i:07d}.txt").read_text(errors="replace")
    docs.append([w for w in simple_preprocess(text, min_len=3)
                 if w not in STOPWORDS and w not in EXTRA_STOP])

dic = corpora.Dictionary(docs)
dic.filter_extremes(no_below=5, no_above=0.5)   # kills per-movie names + boilerplate
bows = [dic.doc2bow(d) for d in docs]
print(f"vocab {len(dic)}, training LDA K=55...")
lda = models.LdaModel(bows, num_topics=55, id2word=dic, passes=8, random_state=42)

theta = np.zeros((len(ids), 55))
for r, bow in enumerate(bows):
    for t, p in lda.get_document_topics(bow, minimum_probability=0):
        theta[r, t] = p
theta /= np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
np.save(ROOT / "data" / "emb_lda.npy", theta)
print("saved data/emb_lda.npy")
