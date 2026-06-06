#!/usr/bin/env python3
"""CanJob pre-computation step (runs OFFLINE; may exceed the 5-minute window).

The hackathon spec allows pre-computation outside the timed window, as long as the
ranking step that produces the CSV stays inside it. This script does the only heavy
part, local MiniLM embeddings, and reduces them to a tiny per-job artifact:

    canjob/config/jobs/<job_key>/precomputed/semantic_scores.npz   (ids + cosine-to-JD)

That artifact is small (~1-2 MB) and committed to the repo, so `rank.py` (and the
Stage-3 Docker reproduction) get the semantic signal with no torch and no network.

The full candidate-embedding matrix is cached to disk (gitignored) so re-runs are
fast. The MiniLM model is loaded from the local HuggingFace cache (HF_HUB_OFFLINE);
download it once beforehand if it is not already cached.

    python precompute.py --candidates ./candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canjob import featurize as fz
from canjob.ranker import JOBS_DIR, resolve_job

DEFAULT_CANDIDATES = os.path.join(os.getcwd(), "candidates.jsonl")
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="path to candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all candidates")
    ap.add_argument("--job", default=None, help="job-config key (auto if only one)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE, help="where to cache the full embedding matrix")
    args = ap.parse_args()

    from canjob import embeddings as emb  # imported here so rank.py never needs torch

    job_dir = resolve_job(args.job)
    print(f"Job config: {os.path.basename(job_dir)}")
    filters_cfg = json.load(open(os.path.join(job_dir, "filters.json")))
    facets_cfg = json.load(open(os.path.join(job_dir, "jd_facets.json")))

    t0 = time.time()
    candidates = fz.load_candidates(args.candidates, args.limit or None)
    df = fz.build_feature_frame(candidates, filters_cfg, facets_cfg)
    ids = df["candidate_id"].to_numpy()
    print(f"Featurized {len(df):,} candidates in {time.time()-t0:.1f}s")

    t = time.time()
    cand_emb = emb.candidate_embeddings(df["emb_text"].tolist(), args.cache_dir)
    print(f"Candidate embeddings ready in {time.time()-t:.1f}s (cache: {args.cache_dir})")

    emb_query = facets_cfg.get("embedding_query") or " ".join(
        [facets_cfg.get("facet_queries", {}).get("retrieval", ""),
         facets_cfg.get("facet_queries", {}).get("ranking_eval", "")]
    )
    scores = (cand_emb @ emb.embed_texts([emb_query])[0]).astype(np.float32)

    out_dir = os.path.join(job_dir, "precomputed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "semantic_scores.npz")
    np.savez_compressed(out_path, ids=ids.astype(object), scores=scores)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({size_mb:.2f} MB) for {len(ids):,} candidates")
    print(f"Pre-computation done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
