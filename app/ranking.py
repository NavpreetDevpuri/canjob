"""Run the CanJob ranking pipeline for the web app.

Runs are serialized through a single background worker so concurrent CPU-heavy jobs
never thrash. Each run scores one job over one candidate set, reusing the exact
featurization / scoring / fusion / reasoning from the core package.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from canjob import featurize as fz
from canjob import ranker as rk
from app import db

_Q: "queue.Queue[int]" = queue.Queue()


def enqueue(run_id: int) -> None:
    db.update_run(run_id, status="queued", stage="queued", progress=0)
    _Q.put(run_id)


def _worker() -> None:
    while True:
        run_id = _Q.get()
        try:
            _execute(run_id)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            db.update_run(run_id, status="error", stage="error", finished_at=db.now())
        finally:
            _Q.task_done()


threading.Thread(target=_worker, daemon=True, name="canjob-runner").start()


def _candidate_list(candidate_set_id: int) -> List[Dict[str, Any]]:
    deltas = db.candidate_deltas()
    removed, added = deltas["removed"], deltas["added"]
    s = db.get_candidate_set(candidate_set_id)
    base = fz.load_candidates(db.candidates_source(), None)
    base = [c for c in base if c.get("candidate_id") not in removed]
    if s and s["kind"] == "all":
        return base + added
    member_ids = set(db.candidate_set_member_ids(candidate_set_id))
    out = [c for c in base if c.get("candidate_id") in member_ids]
    out += [c for c in added if c.get("candidate_id") in member_ids]
    return out


def _execute(run_id: int) -> None:
    run = db.get_run(run_id)
    if not run:
        return
    job = db.get_job(run["job_id"])
    topk = run.get("topk") or 100
    db.update_run(run_id, status="running", stage="loading candidates", progress=5)
    cands = _candidate_list(run["candidate_set_id"])
    if not cands:
        raise RuntimeError("No candidates in this set.")
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
        results.append({"rank": rank, "candidate_id": r["candidate_id"],
                        "score": float(fused[i]), "reasoning": why})
    db.save_results(run_id, results)
    hp_in_top = int(hp[order[:topk]].sum())
    db.update_run(run_id, status="done", stage=f"done ({hp_in_top} honeypots in top {topk})",
                  progress=100, finished_at=db.now())
