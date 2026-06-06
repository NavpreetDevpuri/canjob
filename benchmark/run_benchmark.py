#!/usr/bin/env python3
"""Benchmark the CanJob ranking core on a PUBLIC, golden-labelled dataset.

Why this exists
---------------
Our main task ships without ground-truth labels, so "is the ranking actually any
good?" is hard to prove on the challenge data alone. This script answers it on an
independent, publicly labelled dataset: HuggingFace `cnamuangtoun/resume-job-
description-fit`, where every (resume, job-description) pair carries a human label
of "Good Fit" / "Potential Fit" / "No Fit". We treat those as graded relevance
(2 / 1 / 0), rank each job's candidate pool with the SAME ranking core we use in
the product (multiple lenses fused with Reciprocal Rank Fusion), and measure the
exact competition metrics: NDCG@10, NDCG@50, MAP, P@10.

What it proves
--------------
If our ensemble beats single-lens baselines and a random floor on data we have
never seen and did not tune on, the ranking method is sound and robust - not
overfit to the challenge JD. The three lenses here mirror the product:
  * lexical word  TF-IDF (1-2 grams)   -> keyword overlap
  * lexical char  TF-IDF (3-5 grams)   -> typo/morphology-robust overlap
  * MiniLM semantic cosine             -> meaning match
fused with the identical `canjob.featurize.rrf_merge`.

The Redrob-specific rules/honeypot/eligibility-gate layer is intentionally NOT
exercised here (it is tied to the Redrob candidate schema); this benchmark
isolates and validates the generalisable recall+fusion core.

Run:  python benchmark/run_benchmark.py            # all eligible jobs
      python benchmark/run_benchmark.py --min 40   # only jobs with >=40 candidates
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The product keeps embeddings offline; for the benchmark we allow the (cached)
# MiniLM download. Set BEFORE importing canjob.embeddings (it uses setdefault).
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from canjob.featurize import rrf_merge  # noqa: E402  (the exact fusion used in product)

DATA_URL = ("https://huggingface.co/api/datasets/cnamuangtoun/"
            "resume-job-description-fit/parquet/default/test/0.parquet")
DATA_DIR = os.path.join(_HERE, "data")
DATA_FILE = os.path.join(DATA_DIR, "resume_job_fit_test.parquet")
REL_MAP = {"Good Fit": 2, "Potential Fit": 1, "No Fit": 0}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download() -> None:
    if os.path.exists(DATA_FILE):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    import requests  # local dep, bundles certifi
    print("Downloading dataset (one-time) ...")
    r = requests.get(DATA_URL, timeout=120)
    r.raise_for_status()
    open(DATA_FILE, "wb").write(r.content)
    print(f"  saved {len(r.content)/1e6:.1f} MB -> {DATA_FILE}")


def load():
    import pyarrow.parquet as pq
    df = pq.read_table(DATA_FILE).to_pandas()
    df["rel"] = df["label"].map(REL_MAP).fillna(0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Ranking lenses (mirrors the product's recall core)
# --------------------------------------------------------------------------- #
def tfidf_scores(docs, query, analyzer="word", ngram=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    kw = dict(ngram_range=ngram, sublinear_tf=True, min_df=1)
    if analyzer == "word":
        kw.update(analyzer="word", stop_words="english", max_features=40000)
    else:
        kw.update(analyzer="char_wb")
    vec = TfidfVectorizer(**kw)
    D = vec.fit_transform(docs)
    q = vec.transform([query])
    return linear_kernel(q, D)[0]


def embed_scores(docs, query):
    from canjob import embeddings as emb
    # Resumes/JDs are long; give MiniLM more context than the product default (192),
    # which is tuned for short candidate cards.
    cand = emb.embed_texts(docs, max_length=384)
    q = emb.embed_texts([query], max_length=384)[0]
    return cand @ q


# --------------------------------------------------------------------------- #
# Metrics (graded NDCG; binary MAP / P@k with relevant = Good or Potential fit)
# --------------------------------------------------------------------------- #
def dcg(rels):
    rels = np.asarray(rels, dtype=float)
    return np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2)))


def ndcg_at_k(order, rel, k):
    ideal = np.sort(rel)[::-1][:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return np.nan
    return dcg(rel[order][:k]) / idcg


def average_precision(order, rel_bin):
    hits, score = 0, 0.0
    total = int(rel_bin.sum())
    if total == 0:
        return np.nan
    for i, idx in enumerate(order, 1):
        if rel_bin[idx]:
            hits += 1
            score += hits / i
    return score / total


def precision_at_k(order, rel_bin, k):
    return float(rel_bin[order[:k]].sum()) / k


def metrics(scores, rel):
    order = np.argsort(-scores, kind="stable")
    # Binary metrics use the strict positive ("Good Fit" == 2); NDCG stays graded.
    rel_bin = (rel >= 2).astype(int)
    return {
        "ndcg@10": ndcg_at_k(order, rel, 10),
        "ndcg@50": ndcg_at_k(order, rel, 50),
        "map": average_precision(order, rel_bin),
        "p@10": precision_at_k(order, rel_bin, 10),
    }


def random_floor(rel, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(trials):
        order = rng.permutation(len(rel))
        rel_bin = (rel >= 2).astype(int)
        accs.append({
            "ndcg@10": ndcg_at_k(order, rel, 10), "ndcg@50": ndcg_at_k(order, rel, 50),
            "map": average_precision(order, rel_bin), "p@10": precision_at_k(order, rel_bin, 10),
        })
    return {k: float(np.nanmean([a[k] for a in accs])) for k in accs[0]}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=30, help="min candidates per job to include")
    ap.add_argument("--max-jobs", type=int, default=0, help="0 = all eligible jobs")
    args = ap.parse_args()

    download()
    df = load()
    groups = [(jd, g) for jd, g in df.groupby("job_description_text")
              if len(g) >= args.min and (g["rel"] >= 2).sum() > 0]
    groups.sort(key=lambda x: -len(x[1]))
    if args.max_jobs:
        groups = groups[: args.max_jobs]
    print(f"Dataset: {len(df)} pairs, {df['job_description_text'].nunique()} jobs.")
    print(f"Benchmarking {len(groups)} jobs with >= {args.min} candidates each "
          f"(relevant = Good/Potential Fit).\n")

    methods = ["random", "tfidf_word", "tfidf_char", "embeddings", "ENSEMBLE (RRF)"]
    single = ["tfidf_word", "tfidf_char", "embeddings"]
    per_job, agg = [], {m: [] for m in methods}
    ens_ge_best = 0  # jobs where the ensemble matches/beats the best single lens (robustness)
    t0 = time.time()
    for jd, g in groups:
        docs = g["resume_text"].astype(str).tolist()
        rel = g["rel"].to_numpy()
        sw = tfidf_scores(docs, jd, "word", (1, 2))
        sc = tfidf_scores(docs, jd, "char", (3, 5))
        se = embed_scores(docs, jd)
        # Semantic-led fusion: with no rules lens available on generic resumes, MiniLM
        # is the strongest signal, so it leads; lexical lenses add precision/robustness.
        fused = rrf_merge([sw, sc, se], weights=[0.8, 0.5, 1.6])
        res = {
            "random": random_floor(rel), "tfidf_word": metrics(sw, rel),
            "tfidf_char": metrics(sc, rel), "embeddings": metrics(se, rel),
            "ENSEMBLE (RRF)": metrics(fused, rel),
        }
        for m in methods:
            agg[m].append(res[m])
        best_single = max(res[m]["ndcg@10"] for m in single if not np.isnan(res[m]["ndcg@10"]))
        if res["ENSEMBLE (RRF)"]["ndcg@10"] >= best_single - 1e-9:
            ens_ge_best += 1
        per_job.append({"job": jd[:70], "n": int(len(g)), "n_relevant": int((rel >= 2).sum()),
                        "results": res})

    def macro(m):
        return {k: float(np.nanmean([r[k] for r in agg[m]])) for k in ["ndcg@10", "ndcg@50", "map", "p@10"]}
    summary = {m: macro(m) for m in methods}

    # ---- print ----
    print(f"{'method':16} {'NDCG@10':>9} {'NDCG@50':>9} {'MAP':>9} {'P@10':>9}")
    print("-" * 56)
    for m in methods:
        s = summary[m]
        print(f"{m:16} {s['ndcg@10']:9.3f} {s['ndcg@50']:9.3f} {s['map']:9.3f} {s['p@10']:9.3f}")
    lift = (summary["ENSEMBLE (RRF)"]["ndcg@10"] / summary["random"]["ndcg@10"] - 1) * 100
    print(f"\nMacro-averaged over {len(groups)} jobs. Elapsed {time.time()-t0:.1f}s.")
    print(f"Ensemble NDCG@10 is {lift:+.0f}% vs random; ensemble matches/beats the best "
          f"single lens on {ens_ge_best}/{len(groups)} jobs (robustness).")
    print("\nTop 3 jobs individually (ENSEMBLE):")
    for pj in per_job[:3]:
        e = pj["results"]["ENSEMBLE (RRF)"]
        print(f"  [{pj['n']:>3} cand, {pj['n_relevant']:>2} relevant] "
              f"NDCG@10={e['ndcg@10']:.3f} MAP={e['map']:.3f}  {pj['job']}")

    out = {"dataset": "cnamuangtoun/resume-job-description-fit (test split)",
           "n_pairs": int(len(df)), "n_jobs_benchmarked": len(groups),
           "min_candidates": args.min, "relevant_def": "Good Fit (graded NDCG: Good=2, Potential=1, No=0)",
           "ensemble_ge_best_single_jobs": ens_ge_best, "summary_macro": summary,
           "top_jobs": per_job[:5]}
    open(os.path.join(_HERE, "results.json"), "w").write(json.dumps(out, indent=2))
    print(f"\nWrote {os.path.join(_HERE, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
