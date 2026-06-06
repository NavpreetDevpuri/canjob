"""Run the CanJob ranking pipeline for the web app, reporting progress to the DB.

Reuses the exact featurization/scoring/fusion/reasoning from the core package, so the
UI runs the same engine as `rank.py`. Tuned jobs (the seeded default) use the
precomputed semantic artifact; ad-hoc jobs uploaded in the UI are ranked from their
JD text as the lexical query.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List

import numpy as np

from canjob import featurize as fz
from canjob import ranker as rk
from app import db


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _candidate_list(limit: int) -> List[Dict[str, Any]]:
    deltas = db.candidate_deltas()
    removed, added = deltas["removed"], deltas["added"]
    base = fz.load_candidates(db.candidates_source(), None)
    if removed:
        base = [c for c in base if c.get("candidate_id") not in removed]
    cands = base + added
    if limit and limit > 0:
        cands = cands[:limit]
    return cands


def run_match(run_id: int, job: Dict[str, Any], topk: int = 100, limit: int = 0) -> None:
    try:
        db.update_run(run_id, status="running", stage="loading candidates", progress=5)
        cands = _candidate_list(limit)
        if not cands:
            raise RuntimeError("No candidates to rank (all removed?).")
        db.update_run(run_id, n_candidates=len(cands), stage="featurizing", progress=20)

        job_dir = job.get("job_dir")
        tuned = bool(job_dir)
        base_dir = job_dir or rk.resolve_job(None)
        filters_cfg = json.load(open(os.path.join(base_dir, "filters.json")))
        facets_cfg = json.load(open(os.path.join(base_dir, "jd_facets.json")))
        jd_text = job.get("jd_markdown") or open(os.path.join(base_dir, "job.txt"), encoding="utf-8").read()

        df = fz.build_feature_frame(cands, filters_cfg, facets_cfg)
        by_id = {c["candidate_id"]: c for c in cands}
        ids = df["candidate_id"].to_numpy()

        db.update_run(run_id, stage="scoring (rules + honeypots)", progress=45)
        bd = fz.honeypot_breakdown(df, filters_cfg)
        hp = np.zeros(len(df), dtype=bool)
        hp_reasons = np.array([""] * len(df), dtype=object)
        for name, mask in bd.items():
            newly = mask & ~hp
            hp_reasons[newly] = name
            hp |= mask
        excluded = hp | fz.hard_filter_mask(df)
        det, info = fz.score_jd_aware(df, filters_cfg, facets_cfg)

        db.update_run(run_id, stage="lexical (TF-IDF)", progress=65)
        queries = dict(facets_cfg.get("facet_queries", {}))
        queries["jd_tfidf"] = jd_text
        multi = fz.tfidf_multi(df, queries)

        db.update_run(run_id, stage="fusing", progress=85)
        signals: Dict[str, np.ndarray] = {"deterministic": det}
        signals.update(multi)
        if tuned:
            ew = dict(facets_cfg.get("ensemble_weights", {}))
            sem = rk.load_precomputed_semantic(job_dir, ids)
            if sem is not None:
                signals["embeddings"] = sem
                if ew.get("embeddings", 0) == 0:
                    ew["embeddings"] = 0.7
        else:
            # ad-hoc job: let the uploaded JD (lexical) drive the ranking
            ew = {"deterministic": 0.5, "jd_tfidf": 1.6, "retrieval": 0.4, "ranking_eval": 0.4}

        names = [n for n in signals if ew.get(n, 0) != 0]
        fused = fz.rrf_merge([signals[n] for n in names], weights=[ew.get(n, 1.0) for n in names])
        fused = np.round(rk.scale_unit(fused), 4)
        order = rk.ranked_order(fused, excluded, ids)

        db.update_run(run_id, stage="building results", progress=95)
        results = []
        for rank, i in enumerate(order[:topk], 1):
            r = df.iloc[i]
            cand = by_id[r["candidate_id"]]
            why = rk.reasoning_for(i, df, info, facets_cfg, cand, bool(hp[i]), hp_reasons[i])
            results.append({
                "rank": rank,
                "candidate_id": r["candidate_id"],
                "score": float(fused[i]),
                "reasoning": why,
            })
        db.save_results(run_id, results)
        hp_in_top = int(hp[order[:topk]].sum())
        db.update_run(run_id, status="done", stage=f"done ({hp_in_top} honeypots in top {topk})",
                      progress=100, finished_at=_now())
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        db.update_run(run_id, status="error", stage="error", error=str(e)[:500], finished_at=_now())
