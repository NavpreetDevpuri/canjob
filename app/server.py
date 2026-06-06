"""CanJob web app (bonus): FastAPI backend + single-page UI.

Run:
    pip install -r requirements.txt -r requirements-app.txt
    python app/server.py           # then open http://127.0.0.1:8000

Endpoints under /api; the UI is served at /.
"""
from __future__ import annotations

import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from app import db, ranking  # noqa: E402

app = FastAPI(title="CanJob")
db.init_db()


class JobIn(BaseModel):
    name: str
    jd_markdown: str


class CandidateIn(BaseModel):
    candidate_id: str | None = None
    title: str = ""
    yoe: float | None = None
    location: str = ""
    skills: str = ""
    summary: str = ""


class RunIn(BaseModel):
    job_id: int
    topk: int = 100
    limit: int = 0


# ------------------------------- jobs ---------------------------------------
@app.get("/api/jobs")
def api_jobs():
    return db.list_jobs()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: int):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.post("/api/jobs")
def api_add_job(body: JobIn):
    jid = db.add_job(body.name.strip() or "Untitled job", body.jd_markdown)
    return db.get_job(jid)


@app.delete("/api/jobs/{job_id}")
def api_delete_job(job_id: int):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    if j.get("is_default"):
        raise HTTPException(400, "cannot delete the default job")
    db.delete_job(job_id)
    return {"ok": True}


# ---------------------------- candidates ------------------------------------
@app.get("/api/candidates")
def api_candidates(offset: int = 0, limit: int = 25, q: str = ""):
    return db.list_candidates(offset=offset, limit=min(limit, 200), q=q)


@app.get("/api/candidates/{candidate_id}")
def api_candidate_detail(candidate_id: str):
    c = db.candidate_detail(candidate_id)
    if not c:
        raise HTTPException(404, "candidate not found")
    return c


@app.post("/api/candidates")
def api_add_candidate(body: CandidateIn):
    cid = db.add_candidate(body.model_dump())
    return {"candidate_id": cid}


@app.delete("/api/candidates/{candidate_id}")
def api_delete_candidate(candidate_id: str):
    db.remove_candidate(candidate_id)
    return {"ok": True}


# ------------------------------- runs ---------------------------------------
@app.post("/api/runs")
def api_create_run(body: RunIn):
    job = db.get_job(body.job_id)
    if not job:
        raise HTTPException(404, "job not found")
    run_id = db.create_run(body.job_id, body.topk)
    t = threading.Thread(
        target=ranking.run_match, args=(run_id, job, body.topk, body.limit), daemon=True
    )
    t.start()
    return {"run_id": run_id}


@app.get("/api/runs")
def api_runs(job_id: int | None = None):
    return db.list_runs(job_id)


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    r = db.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r


@app.get("/api/runs/{run_id}/results")
def api_run_results(run_id: int):
    return db.get_results(run_id)


@app.get("/api/matching/summary")
def api_matching_summary():
    return {"jobs": db.matching_summary(), "candidate_count": db.candidate_count()}


# ------------------------------- UI -----------------------------------------
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
