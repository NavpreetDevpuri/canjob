"""CanJob web app (bonus): FastAPI backend + single-page UI.

Run:
    pip install -r requirements.txt -r requirements-app.txt
    python app/server.py           # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import itertools
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from app import db, exports, ranking  # noqa: E402

app = FastAPI(title="CanJob")
db.init_db()


class JobIn(BaseModel):
    name: str
    jd_markdown: str
    set_ids: list[int] = []


class SetIn(BaseModel):
    name: str
    candidate_ids: list[str] | None = None


class CandidateIn(BaseModel):
    candidate_id: str | None = None
    title: str = ""
    yoe: float | None = None
    location: str = ""
    skills: str = ""
    summary: str = ""
    set_ids: list[int] = []


class BatchRunIn(BaseModel):
    job_ids: list[int]
    candidate_set_ids: list[int]
    topk: int = 100


def _csv_resp(text: str, name: str) -> Response:
    return Response(content=text, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _pdf_resp(data: bytes, name: str) -> Response:
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------- jobs + job sets ----------------------------
@app.get("/api/job-sets")
def api_job_sets():
    return db.list_job_sets()


@app.post("/api/job-sets")
def api_add_job_set(body: SetIn):
    return {"id": db.add_job_set(body.name.strip() or "Untitled set")}


@app.delete("/api/job-sets/{set_id}")
def api_del_job_set(set_id: int):
    db.delete_job_set(set_id)
    return {"ok": True}


@app.get("/api/jobs")
def api_jobs(set_id: int | None = None):
    return db.list_jobs(set_id)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: int):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.post("/api/jobs")
def api_add_job(body: JobIn):
    jid = db.add_job(body.name.strip() or "Untitled job", body.jd_markdown, body.set_ids)
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


# ---------------------------- candidates + sets -----------------------------
@app.get("/api/candidate-sets")
def api_candidate_sets():
    return db.list_candidate_sets()


@app.post("/api/candidate-sets")
def api_add_candidate_set(body: SetIn):
    return {"id": db.add_candidate_set(body.name.strip() or "Untitled set", body.candidate_ids)}


@app.delete("/api/candidate-sets/{set_id}")
def api_del_candidate_set(set_id: int):
    db.delete_candidate_set(set_id)
    return {"ok": True}


@app.get("/api/candidates")
def api_candidates(set_id: int | None = None, offset: int = 0, limit: int = 25, q: str = ""):
    return db.list_candidates(set_id=set_id, offset=offset, limit=min(limit, 200), q=q)


@app.get("/api/candidates/{candidate_id}")
def api_candidate_detail(candidate_id: str):
    c = db.candidate_detail(candidate_id)
    if not c:
        raise HTTPException(404, "candidate not found")
    return c


@app.post("/api/candidates")
def api_add_candidate(body: CandidateIn):
    cid = db.add_candidate(body.model_dump(), body.set_ids)
    return {"candidate_id": cid}


@app.delete("/api/candidates/{candidate_id}")
def api_delete_candidate(candidate_id: str):
    db.remove_candidate(candidate_id)
    return {"ok": True}


# ------------------------------- runs ---------------------------------------
@app.post("/api/runs/batch")
def api_batch_runs(body: BatchRunIn):
    if not body.job_ids or not body.candidate_set_ids:
        raise HTTPException(400, "select at least one job and one candidate set")
    pairs = list(itertools.product(body.job_ids, body.candidate_set_ids))
    if len(pairs) > 40:
        raise HTTPException(400, "too many combinations (max 40 per batch)")
    run_ids = []
    for job_id, set_id in pairs:
        if not db.get_job(job_id) or not db.get_candidate_set(set_id):
            continue
        rid = db.create_run(job_id, set_id, body.topk)
        ranking.enqueue(rid)
        run_ids.append(rid)
    return {"run_ids": run_ids}


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    r = db.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r


@app.get("/api/runs/{run_id}/results")
def api_run_results(run_id: int):
    return db.get_results(run_id)


@app.get("/api/matching/matrix")
def api_matrix():
    return db.matching_matrix()


# ------------------------------- exports (own prefix to avoid path clashes) --
@app.get("/api/export/run/{run_id}.csv")
def api_run_csv(run_id: int):
    return _csv_resp(exports.results_csv(run_id), f"canjob_run_{run_id}.csv")


@app.get("/api/export/run/{run_id}.pdf")
def api_run_pdf(run_id: int):
    return _pdf_resp(exports.results_pdf(run_id), f"canjob_run_{run_id}.pdf")


@app.get("/api/export/candidates.csv")
def api_cand_csv(set_id: int | None = None):
    return _csv_resp(exports.candidates_csv(set_id), "canjob_candidates.csv")


@app.get("/api/export/candidates.pdf")
def api_cand_pdf(set_id: int | None = None):
    return _pdf_resp(exports.candidates_pdf(set_id), "canjob_candidates.pdf")


@app.get("/api/export/jobs.csv")
def api_jobs_csv(set_id: int | None = None):
    return _csv_resp(exports.jobs_csv(set_id), "canjob_jobs.csv")


@app.get("/api/export/jobs.pdf")
def api_jobs_pdf(set_id: int | None = None):
    return _pdf_resp(exports.jobs_pdf(set_id), "canjob_jobs.pdf")


# ------------------------------- UI -----------------------------------------
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
