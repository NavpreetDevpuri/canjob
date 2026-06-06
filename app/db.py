"""SQLite data layer for the CanJob web app (bonus).

Holds jobs, a lightweight candidate index (add/remove via the UI), runs and their
results. The full candidate profiles still live in candidates.jsonl and are read at
run time; this DB only stores a per-candidate row for the listing plus add/remove
deltas, so we never duplicate the whole dataset.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SOLUTION_ROOT not in sys.path:
    sys.path.insert(0, SOLUTION_ROOT)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canjob_app.db")


def candidates_source() -> str:
    """Where to seed candidates from: env override, else full dataset, else sample."""
    env = os.environ.get("CANJOB_CANDIDATES")
    if env and os.path.exists(env):
        return env
    full = os.path.join(SOLUTION_ROOT, "..", "India_runs_data_and_ai_challenge", "candidates.jsonl")
    if os.path.exists(full):
        return full
    return os.path.join(SOLUTION_ROOT, "sample", "candidates_sample.jsonl")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            jd_markdown TEXT NOT NULL,
            job_dir TEXT,
            is_default INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            title TEXT,
            yoe REAL,
            location TEXT,
            removed INTEGER DEFAULT 0,
            added INTEGER DEFAULT 0,
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            status TEXT,
            progress INTEGER DEFAULT 0,
            stage TEXT,
            n_candidates INTEGER,
            topk INTEGER,
            created_at TEXT,
            finished_at TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS run_results (
            run_id INTEGER,
            rank INTEGER,
            candidate_id TEXT,
            score REAL,
            reasoning TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_cand_removed ON candidates(removed);
        """
    )
    conn.commit()
    _seed_default_job(conn)
    _seed_candidates(conn)
    conn.close()


def _seed_default_job(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] > 0:
        return
    from canjob.ranker import resolve_job

    job_dir = resolve_job(None)
    jd_path = os.path.join(job_dir, "job.txt")
    jd = open(jd_path, encoding="utf-8").read() if os.path.exists(jd_path) else "Senior AI Engineer"
    meta_path = os.path.join(job_dir, "meta.json")
    name = "Senior AI Engineer (Founding Team)"
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        name = f"{m.get('title', name)}" + (f" - {m.get('company')}" if m.get("company") else "")
    conn.execute(
        "INSERT INTO jobs (name, jd_markdown, job_dir, is_default, created_at) VALUES (?,?,?,1,?)",
        (name, jd, job_dir, _now()),
    )
    conn.commit()


def _seed_candidates(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] > 0:
        return
    src = candidates_source()
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            p = c.get("profile", {}) or {}
            rows.append((
                c.get("candidate_id"),
                p.get("current_title") or p.get("headline") or "",
                p.get("years_of_experience"),
                ", ".join(x for x in [p.get("location"), p.get("country")] if x),
            ))
    conn.executemany(
        "INSERT OR IGNORE INTO candidates (candidate_id, title, yoe, location) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()


# ----------------------------- jobs -----------------------------------------
def list_jobs() -> List[Dict[str, Any]]:
    conn = get_conn()
    out = []
    for j in conn.execute("SELECT * FROM jobs ORDER BY is_default DESC, id ASC").fetchall():
        runs = conn.execute("SELECT COUNT(*) FROM runs WHERE job_id=?", (j["id"],)).fetchone()[0]
        out.append({**dict(j), "run_count": runs})
    conn.close()
    return out


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    j = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return dict(j) if j else None


def add_job(name: str, jd_markdown: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO jobs (name, jd_markdown, job_dir, is_default, created_at) VALUES (?,?,NULL,0,?)",
        (name, jd_markdown, _now()),
    )
    conn.commit()
    jid = cur.lastrowid
    conn.close()
    return jid


def delete_job(job_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM jobs WHERE id=? AND is_default=0", (job_id,))
    conn.execute("DELETE FROM runs WHERE job_id=?", (job_id,))
    conn.commit()
    conn.close()


# ----------------------------- candidates -----------------------------------
def list_candidates(offset: int = 0, limit: int = 25, q: str = "") -> Dict[str, Any]:
    conn = get_conn()
    where = "WHERE removed=0"
    params: List[Any] = []
    if q:
        where += " AND (candidate_id LIKE ? OR title LIKE ? OR location LIKE ?)"
        params += [f"%{q}%", f"%{q}%", f"%{q}%"]
    total = conn.execute(f"SELECT COUNT(*) FROM candidates {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT candidate_id, title, yoe, location, added FROM candidates {where} "
        f"ORDER BY added DESC, candidate_id ASC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()
    return {"total": total, "items": [dict(r) for r in rows]}


def candidate_count() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM candidates WHERE removed=0").fetchone()[0]
    conn.close()
    return n


def add_candidate(payload: Dict[str, Any]) -> str:
    conn = get_conn()
    cid = payload.get("candidate_id") or f"CAND_UI_{int(time.time()*1000)}"
    title = payload.get("title", "")
    yoe = payload.get("yoe")
    location = payload.get("location", "")
    skills = payload.get("skills", "")
    summary = payload.get("summary", "")
    raw = {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "headline": title,
            "summary": summary,
            "location": location,
            "years_of_experience": yoe,
        },
        "career_history": [],
        "education": [],
        "skills": [{"name": s.strip(), "proficiency": "advanced", "duration_months": 24}
                    for s in str(skills).split(",") if s.strip()],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.6,
                            "profile_completeness_score": 80, "notice_period_days": 30},
    }
    conn.execute(
        "INSERT OR REPLACE INTO candidates (candidate_id, title, yoe, location, removed, added, raw_json) "
        "VALUES (?,?,?,?,0,1,?)",
        (cid, title, yoe, location, json.dumps(raw)),
    )
    conn.commit()
    conn.close()
    return cid


def remove_candidate(candidate_id: str) -> None:
    conn = get_conn()
    # UI-added rows are deleted outright; base rows are soft-removed (excluded from runs)
    row = conn.execute("SELECT added FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if row and row["added"]:
        conn.execute("DELETE FROM candidates WHERE candidate_id=?", (candidate_id,))
    else:
        conn.execute("UPDATE candidates SET removed=1 WHERE candidate_id=?", (candidate_id,))
    conn.commit()
    conn.close()


def candidate_deltas() -> Dict[str, Any]:
    """Removed-id set + added candidate dicts, used to build a run's candidate list."""
    conn = get_conn()
    removed = {r["candidate_id"] for r in conn.execute(
        "SELECT candidate_id FROM candidates WHERE removed=1").fetchall()}
    added = [json.loads(r["raw_json"]) for r in conn.execute(
        "SELECT raw_json FROM candidates WHERE added=1 AND removed=0 AND raw_json IS NOT NULL").fetchall()]
    conn.close()
    return {"removed": removed, "added": added}


# ----------------------------- runs -----------------------------------------
def create_run(job_id: int, topk: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO runs (job_id, status, progress, stage, topk, created_at) VALUES (?,?,?,?,?,?)",
        (job_id, "queued", 0, "queued", topk, _now()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def update_run(run_id: int, **fields) -> None:
    if not fields:
        return
    conn = get_conn()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE runs SET {cols} WHERE id=?", list(fields.values()) + [run_id])
    conn.commit()
    conn.close()


def save_results(run_id: int, results: List[Dict[str, Any]]) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM run_results WHERE run_id=?", (run_id,))
    conn.executemany(
        "INSERT INTO run_results (run_id, rank, candidate_id, score, reasoning) VALUES (?,?,?,?,?)",
        [(run_id, r["rank"], r["candidate_id"], r["score"], r["reasoning"]) for r in results],
    )
    conn.commit()
    conn.close()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    return dict(r) if r else None


def list_runs(job_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    if job_id is not None:
        rows = conn.execute("SELECT * FROM runs WHERE job_id=? ORDER BY id DESC", (job_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_results(run_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT rank, candidate_id, score, reasoning FROM run_results WHERE run_id=? ORDER BY rank",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def matching_summary() -> List[Dict[str, Any]]:
    """Per-job latest-run overview for the Matching page."""
    conn = get_conn()
    out = []
    for j in conn.execute("SELECT * FROM jobs ORDER BY is_default DESC, id ASC").fetchall():
        last = conn.execute(
            "SELECT * FROM runs WHERE job_id=? ORDER BY id DESC LIMIT 1", (j["id"],)
        ).fetchone()
        out.append({
            "job_id": j["id"],
            "job_name": j["name"],
            "is_default": j["is_default"],
            "last_run": dict(last) if last else None,
        })
    conn.close()
    return out
