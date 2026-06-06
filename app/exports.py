"""CSV and PDF exporters for runs, candidates, and jobs.

CSV is plain text; PDF uses reportlab (print-friendly, light theme) so the ranked
output can be submitted directly as a document.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app import db

GREEN = colors.HexColor("#0f7a55")
INK = colors.HexColor("#0f172a")
LIGHT = colors.HexColor("#eef3fb")


def _csv(header: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _styles():
    ss = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=7.5, leading=9.5, textColor=INK)
    head = ParagraphStyle("h", parent=ss["Normal"], fontSize=8.5, leading=10,
                          textColor=colors.white, fontName="Helvetica-Bold")
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=16, textColor=INK, spaceAfter=2)
    sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#475569"))
    return cell, head, title, sub


def _table(data, col_widths, head, cell):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _doc(elements, landscape_mode=True) -> bytes:
    buf = io.BytesIO()
    size = landscape(A4) if landscape_mode else A4
    doc = SimpleDocTemplate(buf, pagesize=size, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="CanJob")
    doc.build(elements)
    return buf.getvalue()


# ------------------------------- results ------------------------------------
def results_csv(run_id: int) -> str:
    rows = db.get_results(run_id)
    return _csv(["candidate_id", "rank", "score", "reasoning"],
                [[r["candidate_id"], r["rank"], r["score"], r["reasoning"]] for r in rows])


def results_pdf(run_id: int) -> bytes:
    run = db.get_run(run_id) or {}
    rows = db.get_results(run_id)
    cell, head, title, sub = _styles()
    data = [[Paragraph(h, head) for h in ["#", "Candidate", "Score", "Reasoning"]]]
    for r in rows:
        data.append([
            Paragraph(str(r["rank"]), cell),
            Paragraph(r["candidate_id"], cell),
            Paragraph(f'{r["score"]:.4f}', cell),
            Paragraph((r["reasoning"] or "").replace("&", "&amp;").replace("<", "&lt;"), cell),
        ])
    meta = (f'Job: {run.get("job_name","-")} &nbsp;|&nbsp; Candidate set: {run.get("set_name","-")} '
            f'&nbsp;|&nbsp; {len(rows)} candidates &nbsp;|&nbsp; pool {run.get("n_candidates","-")} '
            f'&nbsp;|&nbsp; finished {run.get("finished_at","-")}')
    els = [Paragraph("CanJob - Ranked Candidates", title), Paragraph(meta, sub), Spacer(1, 6),
           _table(data, [10 * mm, 32 * mm, 16 * mm, 215 * mm], head, cell)]
    return _doc(els, landscape_mode=True)


# ------------------------------- candidates ---------------------------------
def candidates_csv(set_id: Optional[int]) -> str:
    d = db.list_candidates(set_id=set_id, offset=0, limit=10_000_000, q="")
    return _csv(["candidate_id", "title", "yoe", "location", "headline"],
                [[c["candidate_id"], c["title"], c["yoe"], c["location"], c["headline"]] for c in d["items"]])


def candidates_pdf(set_id: Optional[int]) -> bytes:
    d = db.list_candidates(set_id=set_id, offset=0, limit=5000, q="")
    s = db.get_candidate_set(set_id) if set_id else None
    cell, head, title, sub = _styles()
    data = [[Paragraph(h, head) for h in ["ID", "Title", "YoE", "Location", "Headline"]]]
    for c in d["items"]:
        data.append([Paragraph(str(c.get("candidate_id", "")), cell), Paragraph(str(c.get("title", "")), cell),
                     Paragraph("" if c.get("yoe") is None else str(c["yoe"]), cell),
                     Paragraph(str(c.get("location", "")), cell), Paragraph(str(c.get("headline", "")), cell)])
    name = s["name"] if s else "All candidates"
    els = [Paragraph("CanJob - Candidates", title),
           Paragraph(f"Set: {name} &nbsp;|&nbsp; {d['total']} candidates", sub), Spacer(1, 6),
           _table(data, [32 * mm, 55 * mm, 14 * mm, 50 * mm, 122 * mm], head, cell)]
    return _doc(els, landscape_mode=True)


# ------------------------------- jobs ---------------------------------------
def jobs_csv(set_id: Optional[int]) -> str:
    jobs = db.list_jobs(set_id)
    return _csv(["id", "name", "type", "runs", "created_at"],
                [[j["id"], j["name"], "default" if j["is_default"] else "ad-hoc",
                  j["run_count"], j["created_at"]] for j in jobs])


def jobs_pdf(set_id: Optional[int]) -> bytes:
    jobs = db.list_jobs(set_id)
    cell, head, title, sub = _styles()
    data = [[Paragraph(h, head) for h in ["ID", "Title", "Type", "Runs", "Created"]]]
    for j in jobs:
        data.append([Paragraph(str(j["id"]), cell), Paragraph(j["name"], cell),
                     Paragraph("default" if j["is_default"] else "ad-hoc", cell),
                     Paragraph(str(j["run_count"]), cell), Paragraph(str(j["created_at"] or ""), cell)])
    els = [Paragraph("CanJob - Jobs", title), Paragraph(f"{len(jobs)} jobs", sub), Spacer(1, 6),
           _table(data, [14 * mm, 150 * mm, 24 * mm, 18 * mm, 50 * mm], head, cell)]
    return _doc(els, landscape_mode=False)
