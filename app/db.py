"""SQLite data layer for the CanJob web app (bonus).

Holds jobs and job sets, the candidate pool and candidate sets, plus runs and their
results. A run scores one job over one candidate set; the Matching page is the matrix
of (job x candidate set). Full candidate profiles live in candidates.jsonl and are read
at run time; this DB stores a lightweight per-candidate row plus add/remove deltas and
set membership, so we never duplicate the whole dataset.
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
SET_CAP = 400  # cap auto-seeded set size so the default UI stays light and runs stay fast


def candidates_source() -> str:
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
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_TS = "%Y-%m-%d %H:%M:%S"


def now() -> str:
    return time.strftime(_TS)


def _with_elapsed(run: Dict[str, Any]) -> Dict[str, Any]:
    start, end = run.get("created_at"), run.get("finished_at") or now()
    try:
        run["elapsed_seconds"] = max(0, round(
            time.mktime(time.strptime(end, _TS)) - time.mktime(time.strptime(start, _TS))))
    except (TypeError, ValueError):
        run["elapsed_seconds"] = None
    return run


# ----------------------------- schema + seed --------------------------------
def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, jd_markdown TEXT NOT NULL,
            job_dir TEXT, is_default INTEGER DEFAULT 0, created_at TEXT);
        CREATE TABLE IF NOT EXISTS job_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, is_default INTEGER DEFAULT 0, created_at TEXT);
        CREATE TABLE IF NOT EXISTS job_set_members (
            set_id INTEGER, job_id INTEGER, PRIMARY KEY (set_id, job_id));
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY, title TEXT, yoe REAL, location TEXT,
            headline TEXT, summary_len INTEGER DEFAULT 0,
            removed INTEGER DEFAULT 0, added INTEGER DEFAULT 0, raw_json TEXT);
        CREATE TABLE IF NOT EXISTS candidate_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, kind TEXT DEFAULT 'normal',
            is_default INTEGER DEFAULT 0, created_at TEXT);
        CREATE TABLE IF NOT EXISTS candidate_set_members (
            set_id INTEGER, candidate_id TEXT, PRIMARY KEY (set_id, candidate_id));
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER, candidate_set_id INTEGER,
            status TEXT, progress INTEGER DEFAULT 0, stage TEXT,
            n_candidates INTEGER, topk INTEGER,
            created_at TEXT, finished_at TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS run_results (
            run_id INTEGER, rank INTEGER, candidate_id TEXT, score REAL, reasoning TEXT);
        CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_cand_removed ON candidates(removed);
        CREATE INDEX IF NOT EXISTS idx_csm_set ON candidate_set_members(set_id);
        """
    )
    conn.commit()
    _seed_jobs(conn)
    _seed_candidates(conn)
    _seed_candidate_sets(conn)
    conn.close()


def _seed_jobs(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] > 0:
        return
    from canjob.ranker import resolve_job

    job_dir = resolve_job(None)
    jd_path = os.path.join(job_dir, "job.txt")
    jd = open(jd_path, encoding="utf-8").read() if os.path.exists(jd_path) else "Senior AI Engineer"
    name = "Senior AI Engineer (Founding Team)"
    meta_path = os.path.join(job_dir, "meta.json")
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        name = m.get("title", name) + (f" - {m.get('company')}" if m.get("company") else "")
    jid = conn.execute(
        "INSERT INTO jobs (name, jd_markdown, job_dir, is_default, created_at) VALUES (?,?,?,1,?)",
        (name, jd, job_dir, now()),
    ).lastrowid
    sid = conn.execute(
        "INSERT INTO job_sets (name, is_default, created_at) VALUES ('Core roles',1,?)", (now(),)
    ).lastrowid
    conn.execute("INSERT INTO job_set_members (set_id, job_id) VALUES (?,?)", (sid, jid))
    conn.commit()


def _seed_candidates(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] > 0:
        return
    rows = []
    with open(candidates_source(), encoding="utf-8") as f:
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
                p.get("headline") or "",
                len(p.get("summary") or ""),
            ))
    conn.executemany(
        "INSERT OR IGNORE INTO candidates (candidate_id, title, yoe, location, headline, summary_len) "
        "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()


def _seed_candidate_sets(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM candidate_sets").fetchone()[0] > 0:
        return
    conn.execute(
        "INSERT INTO candidate_sets (name, kind, is_default, created_at) "
        "VALUES ('All candidates (full pool)','all',0,?)", (now(),))
    ai = ("lower(title) LIKE '%machine learning%' OR lower(title) LIKE '%ml engineer%' "
          "OR lower(title) LIKE '%ai engineer%' OR lower(title) LIKE '% nlp%' "
          "OR lower(title) LIKE '%recommendation%' OR lower(title) LIKE '%data scientist%' "
          "OR lower(title) LIKE '%machine learning engineer%' OR lower(title) LIKE '%research engineer%'")
    specs = [
        ("AI / ML engineers", ai, 1),
        ("Senior (8y+ experience)", "yoe >= 8", 0),
        ("India-based", "lower(location) LIKE '%india%'", 0),
        ("Data & analytics", "lower(title) LIKE '%data%' OR lower(title) LIKE '%analyst%'", 0),
    ]
    for name, where, is_def in specs:
        sid = conn.execute(
            "INSERT INTO candidate_sets (name, kind, is_default, created_at) VALUES (?,?,?,?)",
            (name, "normal", is_def, now())).lastrowid
        ids = [r[0] for r in conn.execute(
            f"SELECT candidate_id FROM candidates WHERE removed=0 AND ({where}) "
            f"ORDER BY candidate_id LIMIT {SET_CAP}").fetchall()]
        conn.executemany("INSERT OR IGNORE INTO candidate_set_members (set_id, candidate_id) VALUES (?,?)",
                         [(sid, cid) for cid in ids])
    # a tiny fixed sample for quick demos
    sid = conn.execute(
        "INSERT INTO candidate_sets (name, kind, is_default, created_at) VALUES ('Sample (first 50)','normal',0,?)",
        (now(),)).lastrowid
    ids = [r[0] for r in conn.execute(
        "SELECT candidate_id FROM candidates WHERE removed=0 ORDER BY candidate_id LIMIT 50").fetchall()]
    conn.executemany("INSERT OR IGNORE INTO candidate_set_members (set_id, candidate_id) VALUES (?,?)",
                     [(sid, cid) for cid in ids])
    conn.commit()


# ----------------------------- jobs + job sets ------------------------------
def list_job_sets() -> List[Dict[str, Any]]:
    conn = get_conn()
    out = []
    for s in conn.execute("SELECT * FROM job_sets ORDER BY is_default DESC, id ASC").fetchall():
        n = conn.execute("SELECT COUNT(*) FROM job_set_members WHERE set_id=?", (s["id"],)).fetchone()[0]
        out.append({**dict(s), "count": n})
    conn.close()
    return out


def add_job_set(name: str) -> int:
    conn = get_conn()
    sid = conn.execute("INSERT INTO job_sets (name, is_default, created_at) VALUES (?,0,?)",
                       (name, now())).lastrowid
    conn.commit(); conn.close()
    return sid


def delete_job_set(set_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM job_sets WHERE id=? AND is_default=0", (set_id,))
    conn.execute("DELETE FROM job_set_members WHERE set_id=?", (set_id,))
    conn.commit(); conn.close()


def list_jobs(set_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    if set_id:
        q = ("SELECT j.* FROM jobs j JOIN job_set_members m ON m.job_id=j.id "
             "WHERE m.set_id=? ORDER BY j.is_default DESC, j.id ASC")
        jobs = conn.execute(q, (set_id,)).fetchall()
    else:
        jobs = conn.execute("SELECT * FROM jobs ORDER BY is_default DESC, id ASC").fetchall()
    out = []
    for j in jobs:
        runs = conn.execute("SELECT COUNT(*) FROM runs WHERE job_id=?", (j["id"],)).fetchone()[0]
        sets = [r[0] for r in conn.execute(
            "SELECT set_id FROM job_set_members WHERE job_id=?", (j["id"],)).fetchall()]
        out.append({**dict(j), "run_count": runs, "set_ids": sets})
    conn.close()
    return out


def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    j = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return dict(j) if j else None


def add_job(name: str, jd_markdown: str, set_ids: Optional[List[int]] = None) -> int:
    conn = get_conn()
    jid = conn.execute(
        "INSERT INTO jobs (name, jd_markdown, job_dir, is_default, created_at) VALUES (?,?,NULL,0,?)",
        (name, jd_markdown, now())).lastrowid
    for sid in (set_ids or []):
        conn.execute("INSERT OR IGNORE INTO job_set_members (set_id, job_id) VALUES (?,?)", (sid, jid))
    conn.commit(); conn.close()
    return jid


def delete_job(job_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM jobs WHERE id=? AND is_default=0", (job_id,))
    conn.execute("DELETE FROM job_set_members WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM runs WHERE job_id=?", (job_id,))
    conn.commit(); conn.close()


# ----------------------------- candidates + sets ----------------------------
def list_candidate_sets() -> List[Dict[str, Any]]:
    conn = get_conn()
    total_pool = conn.execute("SELECT COUNT(*) FROM candidates WHERE removed=0").fetchone()[0]
    out = []
    for s in conn.execute("SELECT * FROM candidate_sets ORDER BY is_default DESC, id ASC").fetchall():
        if s["kind"] == "all":
            n = total_pool
        else:
            n = conn.execute(
                "SELECT COUNT(*) FROM candidate_set_members m JOIN candidates c "
                "ON c.candidate_id=m.candidate_id WHERE m.set_id=? AND c.removed=0", (s["id"],)).fetchone()[0]
        out.append({**dict(s), "count": n})
    conn.close()
    return out


def get_candidate_set(set_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    s = conn.execute("SELECT * FROM candidate_sets WHERE id=?", (set_id,)).fetchone()
    conn.close()
    return dict(s) if s else None


def add_candidate_set(name: str, candidate_ids: Optional[List[str]] = None) -> int:
    conn = get_conn()
    sid = conn.execute(
        "INSERT INTO candidate_sets (name, kind, is_default, created_at) VALUES (?,'normal',0,?)",
        (name, now())).lastrowid
    if candidate_ids:
        conn.executemany("INSERT OR IGNORE INTO candidate_set_members (set_id, candidate_id) VALUES (?,?)",
                         [(sid, cid) for cid in candidate_ids])
    conn.commit(); conn.close()
    return sid


def delete_candidate_set(set_id: int) -> None:
    conn = get_conn()
    s = conn.execute("SELECT kind, is_default FROM candidate_sets WHERE id=?", (set_id,)).fetchone()
    if s and s["kind"] != "all" and not s["is_default"]:
        conn.execute("DELETE FROM candidate_sets WHERE id=?", (set_id,))
        conn.execute("DELETE FROM candidate_set_members WHERE set_id=?", (set_id,))
        conn.commit()
    conn.close()


def _cand_scope(conn, set_id: Optional[int]):
    """Return (from_clause, where, params) scoping candidates to a set (None/all = full pool)."""
    if set_id:
        s = conn.execute("SELECT kind FROM candidate_sets WHERE id=?", (set_id,)).fetchone()
        if s and s["kind"] != "all":
            return ("candidates c JOIN candidate_set_members m ON m.candidate_id=c.candidate_id",
                    "m.set_id=? AND c.removed=0", [set_id])
    return ("candidates c", "c.removed=0", [])


def list_candidates(set_id: Optional[int] = None, offset: int = 0, limit: int = 25, q: str = "") -> Dict[str, Any]:
    conn = get_conn()
    frm, where, params = _cand_scope(conn, set_id)
    if q:
        where += " AND (c.candidate_id LIKE ? OR c.title LIKE ? OR c.location LIKE ? OR c.headline LIKE ?)"
        params += [f"%{q}%"] * 4
    total = conn.execute(f"SELECT COUNT(*) FROM {frm} WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT c.candidate_id, c.title, c.yoe, c.location, c.headline, c.summary_len, c.added "
        f"FROM {frm} WHERE {where} ORDER BY c.added DESC, c.candidate_id ASC LIMIT ? OFFSET ?",
        params + [limit, offset]).fetchall()
    conn.close()
    return {"total": total, "items": [dict(r) for r in rows]}


def candidate_count(set_id: Optional[int] = None) -> int:
    conn = get_conn()
    frm, where, params = _cand_scope(conn, set_id)
    n = conn.execute(f"SELECT COUNT(*) FROM {frm} WHERE {where}", params).fetchone()[0]
    conn.close()
    return n


def candidate_set_member_ids(set_id: int) -> List[str]:
    conn = get_conn()
    ids = [r[0] for r in conn.execute(
        "SELECT m.candidate_id FROM candidate_set_members m JOIN candidates c "
        "ON c.candidate_id=m.candidate_id WHERE m.set_id=? AND c.removed=0", (set_id,)).fetchall()]
    conn.close()
    return ids


def add_candidate(payload: Dict[str, Any], set_ids: Optional[List[int]] = None) -> str:
    conn = get_conn()
    cid = payload.get("candidate_id") or f"CAND_UI_{int(time.time()*1000)}"
    title, yoe, location = payload.get("title", ""), payload.get("yoe"), payload.get("location", "")
    summary = payload.get("summary", "")
    raw = {
        "candidate_id": cid,
        "profile": {"current_title": title, "headline": title, "summary": summary,
                    "location": location, "years_of_experience": yoe},
        "career_history": [], "education": [],
        "skills": [{"name": s.strip(), "proficiency": "advanced", "duration_months": 24}
                   for s in str(payload.get("skills", "")).split(",") if s.strip()],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.6,
                           "profile_completeness_score": 80, "notice_period_days": 30},
    }
    conn.execute(
        "INSERT OR REPLACE INTO candidates "
        "(candidate_id, title, yoe, location, headline, summary_len, removed, added, raw_json) "
        "VALUES (?,?,?,?,?,?,0,1,?)",
        (cid, title, yoe, location, title, len(summary), json.dumps(raw)))
    for sid in (set_ids or []):
        conn.execute("INSERT OR IGNORE INTO candidate_set_members (set_id, candidate_id) VALUES (?,?)", (sid, cid))
    conn.commit(); conn.close()
    return cid


def remove_candidate(candidate_id: str) -> None:
    conn = get_conn()
    row = conn.execute("SELECT added FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if row and row["added"]:
        conn.execute("DELETE FROM candidates WHERE candidate_id=?", (candidate_id,))
        conn.execute("DELETE FROM candidate_set_members WHERE candidate_id=?", (candidate_id,))
    else:
        conn.execute("UPDATE candidates SET removed=1 WHERE candidate_id=?", (candidate_id,))
    conn.commit(); conn.close()


def candidate_detail(candidate_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT added, raw_json FROM candidates WHERE candidate_id=? AND removed=0",
                       (candidate_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    if row["added"] and row["raw_json"]:
        return json.loads(row["raw_json"])
    needle = f'"{candidate_id}"'
    with open(candidates_source(), encoding="utf-8") as f:
        for line in f:
            if needle in line:
                c = json.loads(line)
                if c.get("candidate_id") == candidate_id:
                    return c
    return None


def candidate_deltas() -> Dict[str, Any]:
    conn = get_conn()
    removed = {r["candidate_id"] for r in conn.execute(
        "SELECT candidate_id FROM candidates WHERE removed=1").fetchall()}
    added = [json.loads(r["raw_json"]) for r in conn.execute(
        "SELECT raw_json FROM candidates WHERE added=1 AND removed=0 AND raw_json IS NOT NULL").fetchall()]
    conn.close()
    return {"removed": removed, "added": added}


# ----------------------------- runs -----------------------------------------
def create_run(job_id: int, candidate_set_id: int, topk: int) -> int:
    conn = get_conn()
    rid = conn.execute(
        "INSERT INTO runs (job_id, candidate_set_id, status, progress, stage, topk, created_at) "
        "VALUES (?,?,?,?,?,?,?)", (job_id, candidate_set_id, "queued", 0, "queued", topk, now())).lastrowid
    conn.commit(); conn.close()
    return rid


def update_run(run_id: int, **fields) -> None:
    if not fields:
        return
    conn = get_conn()
    conn.execute(f"UPDATE runs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                 list(fields.values()) + [run_id])
    conn.commit(); conn.close()


def save_results(run_id: int, results: List[Dict[str, Any]]) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM run_results WHERE run_id=?", (run_id,))
    conn.executemany(
        "INSERT INTO run_results (run_id, rank, candidate_id, score, reasoning) VALUES (?,?,?,?,?)",
        [(run_id, r["rank"], r["candidate_id"], r["score"], r["reasoning"]) for r in results])
    conn.commit(); conn.close()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    r = conn.execute(
        "SELECT r.*, j.name AS job_name, cs.name AS set_name FROM runs r "
        "LEFT JOIN jobs j ON j.id=r.job_id LEFT JOIN candidate_sets cs ON cs.id=r.candidate_set_id "
        "WHERE r.id=?", (run_id,)).fetchone()
    conn.close()
    return _with_elapsed(dict(r)) if r else None


def list_runs(job_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    if job_id is not None:
        rows = conn.execute("SELECT * FROM runs WHERE job_id=? ORDER BY id DESC", (job_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    conn.close()
    return [_with_elapsed(dict(r)) for r in rows]


def get_results(run_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT rank, candidate_id, score, reasoning FROM run_results WHERE run_id=? ORDER BY rank",
        (run_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def latest_run(job_id: int, set_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    r = conn.execute(
        "SELECT * FROM runs WHERE job_id=? AND candidate_set_id=? ORDER BY id DESC LIMIT 1",
        (job_id, set_id)).fetchone()
    conn.close()
    return _with_elapsed(dict(r)) if r else None


def matching_matrix() -> Dict[str, Any]:
    """Jobs x candidate-sets grid with the latest run for each cell."""
    conn = get_conn()
    jobs = [dict(j) for j in conn.execute(
        "SELECT id, name, is_default FROM jobs ORDER BY is_default DESC, id ASC").fetchall()]
    total_pool = conn.execute("SELECT COUNT(*) FROM candidates WHERE removed=0").fetchone()[0]
    csets = []
    for s in conn.execute("SELECT id, name, kind, is_default FROM candidate_sets "
                          "ORDER BY is_default DESC, id ASC").fetchall():
        if s["kind"] == "all":
            cnt = total_pool
        else:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM candidate_set_members m JOIN candidates c "
                "ON c.candidate_id=m.candidate_id WHERE m.set_id=? AND c.removed=0",
                (s["id"],)).fetchone()[0]
        csets.append({**dict(s), "count": cnt})
    cells = {}
    for r in conn.execute(
        "SELECT id, job_id, candidate_set_id, status, progress, stage, n_candidates, topk, "
        "created_at, finished_at FROM runs ORDER BY id ASC").fetchall():
        cells[f"{r['job_id']}_{r['candidate_set_id']}"] = _with_elapsed(dict(r))
    conn.close()
    return {"jobs": jobs, "candidate_sets": csets, "cells": cells}
