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
class Corpus:
    """Fit each lens ONCE over the whole resume corpus, then score any JD query
    against every candidate. This is the realistic retrieval task: given a job,
    rank the entire pool of people, not a pre-filtered shortlist."""

    def __init__(self, resumes):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from canjob import embeddings as emb
        self._lk = __import__("sklearn.metrics.pairwise", fromlist=["linear_kernel"]).linear_kernel
        self.word_vec = TfidfVectorizer(analyzer="word", stop_words="english",
                                        ngram_range=(1, 2), sublinear_tf=True, max_features=40000)
        self.word_D = self.word_vec.fit_transform(resumes)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_D = self.char_vec.fit_transform(resumes)
        # Resumes/JDs are long; give MiniLM more context than the product default (192).
        self.emb_D = emb.embed_texts(resumes, max_length=384)
        self._emb = emb

    def scores(self, jd):
        sw = self._lk(self.word_vec.transform([jd]), self.word_D)[0]
        sc = self._lk(self.char_vec.transform([jd]), self.char_D)[0]
        se = self.emb_D @ self._emb.embed_texts([jd], max_length=384)[0]
        return sw, sc, se


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
    ap.add_argument("--min-good", type=int, default=8, help="min Good-Fit candidates for a job to be scored")
    ap.add_argument("--max-jobs", type=int, default=0, help="0 = all eligible jobs")
    args = ap.parse_args()

    download()
    df = load()

    # ---- realistic retrieval: rank the WHOLE corpus per job ----
    # Dedupe resumes by text -> the candidate pool. For each job, the resumes a human
    # judged carry their label (Good=2/Potential=1); every other resume is treated as
    # No-Fit (0). That is conservative (an unjudged resume that would actually fit and
    # that we rank highly counts AGAINST us), so the real quality is at least this good.
    resumes = list(dict.fromkeys(df["resume_text"].astype(str)))
    idx_of = {r: i for i, r in enumerate(resumes)}
    n = len(resumes)
    print(f"Dataset: {len(df)} judged pairs, {df['job_description_text'].nunique()} jobs, "
          f"{n} unique candidate resumes (the pool each job is ranked against).")

    jobs = []
    for jd, g in df.groupby("job_description_text"):
        rel = np.zeros(n)
        for _, row in g.iterrows():
            rel[idx_of[str(row["resume_text"])]] = row["rel"]
        n_good = int((rel >= 2).sum())
        if n_good >= args.min_good:
            jobs.append((jd, rel, n_good))
    jobs.sort(key=lambda x: -x[2])
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    print(f"Scoring {len(jobs)} jobs that have >= {args.min_good} Good-Fit candidates "
          f"(relevant = Good Fit; ~{np.mean([ng for _,_,ng in jobs])/n*100:.1f}% of the pool, "
          f"so the random floor is low).\n")

    print("Building lenses over the full corpus (TF-IDF x2 + MiniLM embeddings) ...")
    corp = Corpus(resumes)

    methods = ["random", "tfidf_word", "tfidf_char", "embeddings", "ENSEMBLE (RRF)"]
    single = ["tfidf_word", "tfidf_char", "embeddings"]
    per_job, agg = [], {m: [] for m in methods}
    ens_ge_best = 0
    t0 = time.time()
    for jd, rel, n_good in jobs:
        sw, sc, se = corp.scores(jd)
        # Semantic-led fusion: MiniLM is the strongest signal on free text, so it leads;
        # the lexical lenses add keyword precision and robustness.
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
        per_job.append({"job": jd[:70], "pool": n, "n_good": n_good, "results": res})

    def macro(m):
        return {k: float(np.nanmean([r[k] for r in agg[m]])) for k in ["ndcg@10", "ndcg@50", "map", "p@10"]}
    summary = {m: macro(m) for m in methods}

    # ---- print ----
    rnd = summary["random"]
    print(f"{'method':18} {'NDCG@10':>9} {'NDCG@50':>9} {'MAP':>9} {'P@10':>9}  {'P@10 lift':>10}")
    print("-" * 70)
    for m in methods:
        s = summary[m]
        lift = "" if m == "random" else f"{s['p@10']/rnd['p@10']:.0f}x"
        print(f"{m:18} {s['ndcg@10']:9.3f} {s['ndcg@50']:9.3f} {s['map']:9.3f} {s['p@10']:9.3f}  {lift:>10}")
    ens = summary["ENSEMBLE (RRF)"]
    print(f"\nMacro-averaged over {len(jobs)} jobs (each ranked against all {n} candidates). "
          f"Elapsed {time.time()-t0:.1f}s.")
    print(f"Ensemble P@10 = {ens['p@10']:.3f} vs random {rnd['p@10']:.3f}  "
          f"(~{ens['p@10']/rnd['p@10']:.0f}x better); NDCG@10 = {ens['ndcg@10']:.3f} vs {rnd['ndcg@10']:.3f}.")
    print("\nTop 3 jobs individually (ENSEMBLE):")
    for pj in per_job[:3]:
        e = pj["results"]["ENSEMBLE (RRF)"]
        print(f"  [{pj['n_good']:>3} good / {pj['pool']} pool] "
              f"NDCG@10={e['ndcg@10']:.3f} P@10={e['p@10']:.3f}  {pj['job']}")

    out = {"dataset": "cnamuangtoun/resume-job-description-fit (test split)",
           "task": "rank the full candidate corpus per job (unjudged = No-Fit)",
           "n_pairs": int(len(df)), "n_candidates_pool": n, "n_jobs_benchmarked": len(jobs),
           "min_good": args.min_good, "relevant_def": "Good Fit (graded NDCG: Good=2, Potential=1, No=0)",
           "ensemble_ge_best_single_jobs": ens_ge_best, "summary_macro": summary,
           "top_jobs": per_job[:5]}
    open(os.path.join(_HERE, "results.json"), "w").write(json.dumps(out, indent=2))
    print(f"\nWrote {os.path.join(_HERE, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
