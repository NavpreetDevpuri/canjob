"""Adapter that turns Redrob candidate records into SearchableDocument objects.

One candidate -> one SearchableDocument. The ``text_content`` is a flattened,
search-friendly projection of the profile (headline, summary, titles, skills,
role descriptions); the full record is kept on ``original_json_obj`` so the
deterministic re-ranker and honeypot detector can read every field later.
"""

from __future__ import annotations

from typing import Any, Dict, List

from search_engine.adapter import Adapter
from search_engine.models import SearchableDocument


def candidate_to_text(candidate: Dict[str, Any]) -> str:
    """Flatten the parts of a candidate that are useful for lexical matching."""
    profile = candidate.get("profile", {}) or {}
    parts: List[str] = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]

    for role in candidate.get("career_history", []) or []:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))

    skill_names = [s.get("name", "") for s in candidate.get("skills", []) or []]
    parts.append(" ".join(skill_names))

    for edu in candidate.get("education", []) or []:
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))

    return " ".join(p for p in parts if p)


class CandidateAdapter(Adapter):
    """In-memory adapter: documents are provided up front, no live DB sync."""

    def __init__(self, candidates: List[Dict[str, Any]]):
        super().__init__()
        self._candidates = candidates

    def db_to_searchable_documents(self) -> List[SearchableDocument]:
        docs: List[SearchableDocument] = []
        for c in self._candidates:
            cid = c.get("candidate_id", "")
            profile = c.get("profile", {}) or {}
            docs.append(
                SearchableDocument(
                    parent_doc_id=cid,
                    text_content=candidate_to_text(c),
                    original_json_obj=c,
                    metadata={
                        "candidate_id": cid,
                        "country": profile.get("country", ""),
                        "current_title": profile.get("current_title", ""),
                    },
                )
            )
        return docs

    # The POC indexes once and never mutates, so disable the delta-sync machinery.
    def sync_from_db(self, strategy):  # noqa: D401 - simple override
        pass

    def init_from_db(self, strategy):
        strategy.upsert_documents(self.db_to_searchable_documents())
