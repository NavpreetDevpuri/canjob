# Reproducible CPU-only ranking image (Stage-3 sandbox parity: 5 min, 16 GB, no GPU, no network).
#
# Build:  docker build -t canjob .
# Run (no network needed at rank time):
#   docker run --rm --network none \
#     -v "$PWD/../India_runs_data_and_ai_challenge:/data:ro" \
#     -v "$PWD/output:/out" \
#     canjob --candidates /data/candidates.jsonl --out /out/submission.csv
#
# Only core deps are installed (no torch). The semantic signal is read from the
# committed per-job artifact canjob/config/jobs/<key>/precomputed/semantic_scores.npz,
# so ranking is fully offline and finishes in well under 5 minutes.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rank.py precompute.py ./
COPY canjob ./canjob
COPY search_engine ./search_engine

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/out/submission.csv"]
