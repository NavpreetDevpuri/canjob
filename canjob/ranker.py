#!/usr/bin/env python3
"""CanJob ranking engine (JD-faithful, vectorized, CPU-only, offline).

  load -> featurize -> honeypot/hard-filter masks -> JD-aware deterministic score
       -> faceted TF-IDF recall -> precomputed semantic scores -> RRF ensemble -> CSV

The ranking step needs no torch and no network: semantic scores are precomputed
offline (see precompute.py) and committed as a small per-job artifact. This keeps
the step that produces submission.csv well inside the 5-minute budget.

Run:  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SOLUTION_ROOT not in sys.path:
    sys.path.insert(0, _SOLUTION_ROOT)

from canjob import featurize as fz

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CANDIDATES = os.path.join(os.getcwd(), "candidates.jsonl")
DEFAULT_OUT = os.path.join(os.getcwd(), "submission.csv")
JOBS_DIR = os.path.join(_HERE, "config", "jobs")


def load_precomputed_semantic(job_dir: str, ids) -> Optional[np.ndarray]:
    """Load the small per-job semantic-score artifact (ids + cosine-to-JD scores).

    Produced offline by precompute.py and committed to the repo, so the ranking
    step gets the semantic signal without torch, a model, or any network.
    """
    path = os.path.join(job_dir, "precomputed", "semantic_scores.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    score_by_id = dict(zip(d["ids"].tolist(), d["scores"].tolist()))
    return np.array([score_by_id.get(c, 0.0) for c in ids], dtype=float)


def resolve_job(job: str | None) -> str:
    """Return the path to a job-config folder under config/jobs/.

    Each job lives in its own folder named slug(company)__slug(title)__hash8(job.txt),
    holding job.txt + jd_facets.json + filters.json (+ meta.json). With one job we use
    it automatically; with several, --job selects one.
    """
    jobs = sorted(d for d in os.listdir(JOBS_DIR) if os.path.isdir(os.path.join(JOBS_DIR, d)))
    if job:
        if job not in jobs:
            raise SystemExit(f"Unknown --job '{job}'. Available: {jobs}")
        return os.path.join(JOBS_DIR, job)
    if len(jobs) == 1:
        return os.path.join(JOBS_DIR, jobs[0])
    raise SystemExit(f"Multiple jobs found; pass --job <key>. Available: {jobs}")

FACET_LABELS = {
    "retrieval_embeddings": "embeddings/retrieval",
    "vector_db": "vector search",
    "ranking_recsys_search": "ranking/recsys",
    "evaluation": "ranking evaluation",
    "nlp_ir": "NLP/IR",
    "python_eng": "production eng",
    "llm_finetune": "LLM fine-tuning",
    "learning_to_rank": "learning-to-rank",
    "hrtech_marketplace": "HR-tech",
    "scale_infra": "scale/infra",
    "opensource": "open-source",
}
CONCERN_LABELS = {
    "keyword_stuffer": "off-domain title with stuffed AI skills",
    "cv_speech_robotics": "CV/speech-primary, thin on NLP/IR",
    "langchain_only": "framework-only LLM exposure",
    "title_chaser": "short tenures (title-chaser pattern)",
    "stale": "inactive recently",
}

# JD must-have -> (label, keywords, source). Evidence is pulled from the candidate's
# OWN profile so every claim is concrete and verifiable (no templated praise).
JD_REQUIREMENTS = [
    ("vector DBs/hybrid search",
     ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "pgvector", "haystack"], "skill"),
    ("embeddings retrieval",
     ["embedding", "sentence-transformers", "sentence transformers", "bge", "e5", "semantic search", "dense retrieval"], "skill"),
    ("ranking/recsys",
     ["learning to rank", "learning-to-rank", "ltr", "recommendation", "recommender", "ranking", "bm25", "information retrieval", "relevance"], "skill"),
    ("eval frameworks",
     ["ndcg", "mrr", "map@", "mean average precision", "a/b", "ab test", "offline", "precision@", "recall@"], "prose"),
    ("LLM fine-tuning",
     ["lora", "qlora", "peft", "rag", "fine-tun"], "skill"),
]
_PROSE_DISPLAY = {"ndcg": "NDCG", "mrr": "MRR", "map@": "MAP", "mean average precision": "MAP",
                  "a/b": "A/B tests", "ab test": "A/B tests", "offline": "offline eval",
                  "precision@": "P@k", "recall@": "recall@k"}


def _skill_evidence(candidate, keywords, limit=3):
    out, seen = [], set()
    for s in candidate.get("skills", []) or []:
        nl = str(s.get("name", "")).lower()
        if any(k in nl for k in keywords) and nl not in seen:
            seen.add(nl)
            prof = str(s.get("proficiency", "?"))[:3]
            out.append(f"{s.get('name')}({prof},{s.get('duration_months','?')}mo)")
        if len(out) >= limit:
            break
    return out


def _prose_evidence(candidate, keywords):
    text = (candidate.get("profile", {}).get("summary", "") + " "
            + " ".join(r.get("description", "") for r in candidate.get("career_history", []) or [])).lower()
    found = []
    for k in keywords:
        if k in text:
            disp = _PROSE_DISPLAY.get(k, k.upper())
            if disp not in found:
                found.append(disp)
    return found[:3]


def _companies(candidate, services_set, limit=3):
    prod, svc = [], []
    for r in candidate.get("career_history", []) or []:
        c = r.get("company", "")
        if not c:
            continue
        (svc if any(x in c.lower() for x in services_set) else prod).append(c)
    return prod[:limit], svc[:limit]


def reasoning_for(i, df, info, facets_cfg, candidate, honeypot, hp_reason=""):
    r = df.iloc[i]
    title = (r["title"] or "?").title()
    yoe = r["yoe"]
    if honeypot:
        return f"{title}, {yoe:g}y: flagged honeypot ({hp_reason.replace('_', ' ')}) - logically impossible profile, demoted."

    services_set = facets_cfg.get("services_companies", [])
    prod, svc = _companies(candidate, services_set)
    where = ", ".join(prod[:2]) if prod else (", ".join(svc[:2]) + " (services)" if svc else "n/a")

    # concrete JD-requirement -> candidate-evidence clauses
    clauses = []
    for label, kws, src in JD_REQUIREMENTS:
        ev = _skill_evidence(candidate, kws) if src == "skill" else _prose_evidence(candidate, kws)
        if ev:
            clauses.append(f"{label}: {', '.join(ev)}")
    match_str = "; ".join(clauses[:4]) if clauses else "no direct JD-skill match (ranked on adjacent signals)"

    # behavioral signals (specific values)
    la = candidate.get("redrob_signals", {}).get("last_active_date", "?")
    sig = f"{r['recruiter_response_rate']:.0%} recruiter response, active {str(la)[:7]}, {int(r['notice_period_days'])}d notice"

    concerns = [CONCERN_LABELS[f] for f in CONCERN_LABELS if info["flags"].get(f) is not None and info["flags"][f][i]]
    if bool(r["services_all"]):
        concerns.append("services-only career")

    s = f"{title}, {yoe:g}y @ {where}. JD match -> {match_str}. Signals: {sig}."
    if concerns:
        s += " Concerns: " + "; ".join(concerns[:2]) + "."
    return s


def write_csv(path, df, order, scores, hp_mask, hp_reasons, info, facets_cfg, by_id, topk):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, i in enumerate(order[:topk], 1):
            r = df.iloc[i]
            cand = by_id[r["candidate_id"]]
            why = reasoning_for(i, df, info, facets_cfg, cand, bool(hp_mask[i]), hp_reasons[i])
            w.writerow([r["candidate_id"], rank, f"{float(scores[i]):.4f}", why])


def ranked_order(scores, exclude, ids):
    masked = np.where(exclude, -np.inf, scores)
    return np.lexsort((ids, -masked))


def overlap(a, b, n):
    return len(set(a[:n].tolist()) & set(b[:n].tolist()))


def scale_unit(scores: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] (monotonic, so it never changes the ranking)."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="path to candidates.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output submission CSV path")
    ap.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    ap.add_argument("--topk", type=int, default=100, help="rows in the submission (default 100)")
    ap.add_argument("--job", default=None, help="job-config key under config/jobs/ (auto if only one)")
    ap.add_argument("--no-semantic", action="store_true", help="ignore the precomputed semantic scores")
    ap.add_argument("--debug-dir", default=None, help="also write rules/semantic comparison CSVs + agreement report here")
    args = ap.parse_args()

    job_dir = resolve_job(args.job)
    print(f"Job config: {os.path.basename(job_dir)}")
    filters_cfg = json.load(open(os.path.join(job_dir, "filters.json")))
    facets_cfg = json.load(open(os.path.join(job_dir, "jd_facets.json")))
    jd_text = open(os.path.join(job_dir, "job.txt"), encoding="utf-8").read()

    t0 = time.time()
    candidates = fz.load_candidates(args.candidates, args.limit or None)
    by_id = {c["candidate_id"]: c for c in candidates}
    print(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    t = time.time()
    df = fz.build_feature_frame(candidates, filters_cfg, facets_cfg)
    print(f"Featurized in {time.time()-t:.2f}s")

    # honeypots (with breakdown) + hard filters
    bd = fz.honeypot_breakdown(df, filters_cfg)
    hp = np.zeros(len(df), dtype=bool)
    hp_reasons = np.array([""] * len(df), dtype=object)
    for name, mask in bd.items():
        newly = mask & ~hp
        hp_reasons[newly] = name
        hp |= mask
    hard = fz.hard_filter_mask(df)
    excluded = hp | hard

    t = time.time()
    det, info = fz.score_jd_aware(df, filters_cfg, facets_cfg)
    print(f"JD-aware deterministic score in {time.time()-t:.2f}s")

    # ---- multiple recall strategies (faceted TF-IDF queries + full JD) ----
    t = time.time()
    queries = dict(facets_cfg.get("facet_queries", {}))
    queries["jd_tfidf"] = jd_text
    multi = fz.tfidf_multi(df, queries)
    print(f"Faceted TF-IDF ({len(queries)} queries) in {time.time()-t:.2f}s")

    # ---- precomputed semantic scores (offline artifact; no torch at rank time) ----
    ew = dict(facets_cfg.get("ensemble_weights", {}))
    signals: Dict[str, np.ndarray] = {"deterministic": det}
    signals.update(multi)
    ids = df["candidate_id"].to_numpy()
    sem = None if args.no_semantic else load_precomputed_semantic(job_dir, ids)
    if sem is not None:
        signals["embeddings"] = sem
        if ew.get("embeddings", 0) == 0:
            ew["embeddings"] = 0.7
        print("Loaded precomputed semantic scores (offline artifact).")
    else:
        print("No semantic artifact found -> core ensemble (rules + lexical). Run precompute.py to add it.")

    # ---- ensemble via weighted RRF ----
    names = [n for n in signals if ew.get(n, 0) != 0]
    fused = fz.rrf_merge([signals[n] for n in names], weights=[ew.get(n, 1.0) for n in names])

    rounded = {n: np.round(s, 4) for n, s in signals.items()}

    # RRF produces tiny raw values (~1/k). Rescale to a readable 0-1 fit score.
    # This is a strictly monotonic transform, so the ranking is unchanged; it only
    # makes the published score interpretable (1.0 = best fit in the pool).
    fused = scale_unit(fused)
    fused = np.round(fused, 4)

    ensemble = ranked_order(fused, excluded, ids)

    # honeypot safety check on the submitted ranking
    hp_in_top = int(hp[ensemble[: args.topk]].sum())

    write_csv(args.out, df, ensemble, fused, hp, hp_reasons, info, facets_cfg, by_id, args.topk)

    # ---- report ----
    print("\n================ HONEYPOTS ================")
    print(f"Total flagged: {int(hp.sum())}  (hard-filtered: {int(hard.sum())})")
    for name, mask in sorted(bd.items(), key=lambda x: -int(x[1].sum())):
        if mask.sum():
            print(f"  {name:34} {int(mask.sum()):>5}")
    print(f"Honeypots in submitted top {args.topk}: {hp_in_top}  (must be 0; disqualified if >10%)")

    print("\n================ TOP 10 (submitted) ================")
    for i in ensemble[:10]:
        r = df.iloc[i]
        prod = "prod" if r["has_product_role"] else ("svc" if r["services_all"] else "?")
        print(f"  {r['candidate_id']} {fused[i]:.4f} {str(r['title'])[:26]:26} yoe={r['yoe']:<4g} {prod}")

    if args.debug_dir:
        _write_debug(args.debug_dir, df, ids, excluded, rounded, fused, names, ensemble,
                     hp, hp_reasons, info, facets_cfg, by_id, args.topk)

    print(f"\nStrategies fused: {names}")
    print(f"TOTAL wall time: {time.time()-t0:.1f}s")
    print(f"Submission -> {args.out}")
    return 0


def _write_debug(out_dir, df, ids, excluded, rounded, fused, names, ensemble,
                 hp, hp_reasons, info, facets_cfg, by_id, topk):
    """Optional: comparison CSVs + cross-strategy agreement (a confidence proxy)."""
    os.makedirs(out_dir, exist_ok=True)
    rules = ranked_order(rounded["deterministic"], excluded, ids)
    sem_key = "embeddings" if "embeddings" in rounded else "jd_tfidf"
    semantic = ranked_order(rounded[sem_key], excluded, ids)
    write_csv(os.path.join(out_dir, "submission_rules.csv"), df, rules, rounded["deterministic"], hp, hp_reasons, info, facets_cfg, by_id, topk)
    write_csv(os.path.join(out_dir, "submission_semantic.csv"), df, semantic, rounded[sem_key], hp, hp_reasons, info, facets_cfg, by_id, topk)

    print("\n================ STRATEGY AGREEMENT (confidence proxy) ================")
    strat_orders = {n: ids[ranked_order(rounded[n], excluded, ids)] for n in names}
    strat_orders["ENSEMBLE"] = ids[ensemble]
    keys = list(strat_orders)
    for n in (10, 50, 100):
        core = set(strat_orders[keys[0]][:n].tolist())
        for k in keys[1:]:
            core &= set(strat_orders[k][:n].tolist())
        print(f"  consensus core @{n:>3} (in ALL {len(keys)} strategies): {len(core)}")
    print("  pairwise overlap vs ENSEMBLE @100:")
    for n in names:
        print(f"    {n:16} {overlap(strat_orders[n], strat_orders['ENSEMBLE'], 100):>3}")


if __name__ == "__main__":
    raise SystemExit(main())
