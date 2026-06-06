#!/usr/bin/env python3
"""CanJob: single-command entrypoint that produces the submission CSV.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

CPU-only, offline, completes well within the 5-minute budget. The semantic signal
is precomputed offline (see precompute.py) and committed as a small per-job
artifact, so this step needs neither torch nor any network access.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canjob.ranker import main

if __name__ == "__main__":
    raise SystemExit(main())
