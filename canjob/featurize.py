"""Vectorized featurization, JD-faithful scoring, and honeypot detection.

All heavy work is column-wise (pandas/numpy) or sparse (scipy/sklearn) - no
per-row Python loops on the hot path. The scorer is driven by config/jd_facets.json
so the weighting reflects the *real* job description, including its explicit
disqualifiers (CV/speech-primary, services-only firms, keyword stuffers, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import orjson
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_candidates(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(orjson.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _col(base: pd.DataFrame, name: str, default):
    if name in base.columns:
        return base[name]
    return pd.Series([default] * len(base), index=base.index)


def _regex(keywords: List[str]) -> str:
    return "|".join(re.escape(k) for k in keywords)


# --------------------------------------------------------------------------- #
# Feature frame
# --------------------------------------------------------------------------- #
def build_feature_frame(candidates, filters_cfg, facets_cfg) -> pd.DataFrame:
    base = pd.json_normalize(candidates)
    n = len(base)
    df = pd.DataFrame({"candidate_id": base["candidate_id"]})
    idx = pd.Index(df["candidate_id"])

    df["yoe"] = pd.to_numeric(_col(base, "profile.years_of_experience", 0), errors="coerce").fillna(0.0)
    df["title"] = _col(base, "profile.current_title", "").fillna("").astype(str).str.lower()
    df["headline"] = _col(base, "profile.headline", "").fillna("").astype(str)
    df["summary"] = _col(base, "profile.summary", "").fillna("").astype(str)
    df["location"] = (
        _col(base, "profile.location", "").fillna("").astype(str)
        + " " + _col(base, "profile.country", "").fillna("").astype(str)
    ).str.lower()

    num = lambda c, d=0.0: pd.to_numeric(_col(base, c, d), errors="coerce").fillna(d)  # noqa: E731
    df["recruiter_response_rate"] = num("redrob_signals.recruiter_response_rate")
    df["completeness"] = num("redrob_signals.profile_completeness_score") / 100.0
    gh = num("redrob_signals.github_activity_score")
    df["github"] = np.where(gh < 0, 0.0, gh / 100.0)
    df["open_to_work"] = _col(base, "redrob_signals.open_to_work_flag", False).fillna(False).astype(bool)
    df["verified"] = (
        _col(base, "redrob_signals.verified_email", False).fillna(False).astype(int)
        + _col(base, "redrob_signals.verified_phone", False).fillna(False).astype(int)
    )
    df["notice_period_days"] = num("redrob_signals.notice_period_days", 0)
    sal_min = num("redrob_signals.expected_salary_range_inr_lpa.min", 0)
    sal_max = num("redrob_signals.expected_salary_range_inr_lpa.max", 0)
    df["salary_min_gt_max"] = (sal_min > sal_max).to_numpy()

    # additional engagement / reliability signals (Redrob behavioral signals doc)
    icr = num("redrob_signals.interview_completion_rate", -1)
    df["interview_completion_rate"] = icr.where(icr >= 0, np.nan)
    oar = num("redrob_signals.offer_acceptance_rate", -1)
    df["offer_acceptance_rate"] = oar.where(oar >= 0, np.nan)
    df["saved_by_recruiters_30d"] = num("redrob_signals.saved_by_recruiters_30d", 0)
    df["profile_views_30d"] = num("redrob_signals.profile_views_received_30d", 0)
    df["applications_30d"] = num("redrob_signals.applications_submitted_30d", 0)

    # activity recency (months before the most recent activity in the dataset)
    la = pd.to_datetime(_col(base, "redrob_signals.last_active_date", None), errors="coerce")
    su = pd.to_datetime(_col(base, "redrob_signals.signup_date", None), errors="coerce")
    ref = la.max()
    df["recency_months"] = ((ref - la).dt.days / 30.44).fillna(99.0)
    df["last_active_before_signup"] = (la < su).fillna(False).to_numpy()

    # ---- skills ----
    core = set(s.lower() for s in filters_cfg.get("ai_core_skills", []))
    sk = pd.json_normalize(candidates, record_path="skills", meta=["candidate_id"], errors="ignore")
    if len(sk):
        sk["name_l"] = sk["name"].astype(str).str.lower()
        sk["is_core"] = sk["name_l"].isin(core)
        sk["dur"] = pd.to_numeric(sk.get("duration_months"), errors="coerce").fillna(0)
        sk["expert_zero"] = sk["proficiency"].isin(["advanced", "expert"]) & (sk["dur"] == 0)
        yoe_map = df.set_index("candidate_id")["yoe"]
        sk["cand_yoe"] = sk["candidate_id"].map(yoe_map).fillna(0)
        sk["dur_gt_career"] = sk["dur"] > (sk["cand_yoe"] * 12 + 12)
        g = sk.groupby("candidate_id")
        df["ai_core_skill_count"] = g["is_core"].sum().reindex(idx).fillna(0).to_numpy()
        df["expert_zero_count"] = g["expert_zero"].sum().reindex(idx).fillna(0).to_numpy()
        df["skill_dur_gt_career"] = g["dur_gt_career"].any().reindex(idx).fillna(False).to_numpy()
        df["skill_names"] = g["name_l"].apply(" ".join).reindex(idx).fillna("").to_numpy()
    else:
        df["ai_core_skill_count"] = 0
        df["expert_zero_count"] = 0
        df["skill_dur_gt_career"] = False
        df["skill_names"] = ""

    # ---- career history ----
    services_re = _regex(facets_cfg.get("services_companies", []))
    ch = pd.json_normalize(candidates, record_path="career_history", meta=["candidate_id"], errors="ignore")
    if len(ch):
        ch["dur"] = pd.to_numeric(ch.get("duration_months"), errors="coerce").fillna(0)
        s_dt = pd.to_datetime(ch.get("start_date"), errors="coerce")
        e_dt = pd.to_datetime(ch.get("end_date"), errors="coerce")
        computed = (e_dt - s_dt).dt.days / 30.44
        ch["date_inv"] = (e_dt.notna()) & (s_dt > e_dt)
        ch["dur_mismatch"] = e_dt.notna() & ((ch["dur"] - computed).abs() > 9)
        iscur = ch.get("is_current").fillna(False).astype(bool) if "is_current" in ch else pd.Series(False, index=ch.index)
        ch["cur_inconsistent"] = (iscur & e_dt.notna()) | (~iscur & e_dt.isna())
        ch["company_l"] = ch.get("company", "").astype(str).str.lower()
        ch["is_services"] = ch["company_l"].str.contains(services_re, regex=True, na=False) if services_re else False
        ch["is_product"] = (~ch["is_services"]) & (ch["company_l"].str.len() > 0)
        ch["title_l"] = ch.get("title", "").astype(str).str.lower()
        ch["desc"] = ch.get("description", "").astype(str)
        g = ch.groupby("candidate_id")
        df["num_roles"] = g.size().reindex(idx).fillna(0).to_numpy()
        df["total_career_months"] = g["dur"].sum().reindex(idx).fillna(0).to_numpy()
        df["max_role_months"] = g["dur"].max().reindex(idx).fillna(0).to_numpy()
        df["career_date_inversion"] = g["date_inv"].any().reindex(idx).fillna(False).to_numpy()
        df["date_duration_mismatch"] = g["dur_mismatch"].any().reindex(idx).fillna(False).to_numpy()
        df["is_current_inconsistent"] = g["cur_inconsistent"].any().reindex(idx).fillna(False).to_numpy()
        df["services_all"] = (g["is_services"].mean().reindex(idx).fillna(0) == 1.0).to_numpy()
        df["has_product_role"] = g["is_product"].any().reindex(idx).fillna(False).to_numpy()
        df["career_titles"] = g["title_l"].apply(" ".join).reindex(idx).fillna("").to_numpy()
        df["career_desc"] = g["desc"].apply(" ".join).reindex(idx).fillna("").to_numpy()
    else:
        for c in ["num_roles", "total_career_months", "max_role_months"]:
            df[c] = 0
        for c in ["career_date_inversion", "date_duration_mismatch", "is_current_inconsistent",
                  "services_all", "has_product_role"]:
            df[c] = False
        df["career_titles"] = ""
        df["career_desc"] = ""

    df["avg_tenure_months"] = np.where(df["num_roles"] > 0, df["total_career_months"] / df["num_roles"].clip(lower=1), 0)

    # ---- education ----
    ed = pd.json_normalize(candidates, record_path="education", meta=["candidate_id"], errors="ignore")
    if len(ed):
        ed["sy"] = pd.to_numeric(ed.get("start_year"), errors="coerce")
        ed["ey"] = pd.to_numeric(ed.get("end_year"), errors="coerce")
        ed["inv"] = ed["sy"] > ed["ey"]
        g = ed.groupby("candidate_id")
        df["edu_year_inversion"] = g["inv"].any().reindex(idx).fillna(False).to_numpy()
        df["edu_max_end_year"] = g["ey"].max().reindex(idx).fillna(0).to_numpy()
    else:
        df["edu_year_inversion"] = False
        df["edu_max_end_year"] = 0

    # ---- derived ----
    df["tenure_minus_exp_years"] = (df["total_career_months"] - df["yoe"] * 12.0) / 12.0
    df["max_role_minus_exp_years"] = (df["max_role_months"] - df["yoe"] * 12.0) / 12.0

    df["blob"] = (
        df["summary"] + " " + df["headline"] + " " + df["career_titles"] + " "
        + df["career_desc"] + " " + df["skill_names"]
    ).str.lower()
    df["search_text"] = (
        df["headline"] + " " + df["summary"] + " " + df["title"] + " " + df["career_desc"]
    )
    # Focused, short text for semantic embeddings: title + human-written prose only,
    # capped so CPU encoding stays fast and the signal is about role/skill fit (not
    # a giant keyword concat that washes the cosine toward generic similarity).
    df["emb_text"] = (
        df["title"].fillna("") + ". " + df["headline"].fillna("") + ". " + df["summary"].fillna("")
    ).str.slice(0, 600)
    return df


# --------------------------------------------------------------------------- #
# Honeypot detection (vectorized, per-rule breakdown)
# --------------------------------------------------------------------------- #
def honeypot_breakdown(df: pd.DataFrame, filters_cfg) -> Dict[str, np.ndarray]:
    """High-precision 'logically impossible profile' checks only.

    Deliberately excludes noisy signals (salary min>max, skill-duration>career,
    last_active<signup) which fire on ~7-19% of the pool and are clearly data
    noise, not the ~80 hidden honeypots. We prefer precision: a false honeypot
    would wrongly demote/exclude a genuine top candidate.
    """
    t = filters_cfg.get("honeypot_thresholds", {})
    return {
        "expert_zero_skills": (df["expert_zero_count"] >= t.get("expert_zero_min", 3)).to_numpy(),
        "career_date_inversion": df["career_date_inversion"].to_numpy(),
        "edu_year_inversion": df["edu_year_inversion"].to_numpy(),
        "future_graduation": (df["edu_max_end_year"] > t.get("future_grad_year", 2028)).to_numpy(),
        "tenure_exceeds_experience": (df["tenure_minus_exp_years"] >= t.get("tenure_gap_years", 5)).to_numpy(),
        "single_role_exceeds_experience": (df["max_role_minus_exp_years"] >= t.get("role_gap_years", 4)).to_numpy(),
        "date_duration_mismatch": df["date_duration_mismatch"].to_numpy(),
        "is_current_inconsistent": df["is_current_inconsistent"].to_numpy(),
    }


def honeypot_mask(df: pd.DataFrame, filters_cfg) -> np.ndarray:
    bd = honeypot_breakdown(df, filters_cfg)
    out = np.zeros(len(df), dtype=bool)
    for v in bd.values():
        out |= v
    return out


def hard_filter_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["headline"] == "") & (df["summary"] == "")).to_numpy()


# --------------------------------------------------------------------------- #
# JD-faithful scoring
# --------------------------------------------------------------------------- #
def _facet_counts(df: pd.DataFrame, facets: List[dict]) -> Dict[str, np.ndarray]:
    out = {}
    blob = df["blob"]
    for fac in facets:
        rx = _regex(fac["keywords"])
        out[fac["id"]] = blob.str.count(rx).to_numpy().astype(float)
    return out


def _experience_score(yoe: np.ndarray, e: dict) -> np.ndarray:
    s = np.full(len(yoe), 0.1)
    s = np.where((yoe >= e["ok_min"]) & (yoe <= e["ok_max"]), 0.85, s)
    s = np.where((yoe >= e["ideal_min"]) & (yoe <= e["ideal_max"]), 1.0, s)
    below = (yoe < e["ok_min"]) & (yoe >= e["hard_min"])
    s = np.where(below, 0.3 + 0.5 * (yoe - e["hard_min"]) / max(e["ok_min"] - e["hard_min"], 1), s)
    above = (yoe > e["ok_max"]) & (yoe <= e["hard_max"])
    s = np.where(above, np.clip(0.6 - 0.4 * (yoe - e["ok_max"]) / max(e["hard_max"] - e["ok_max"], 1), 0.2, 0.6), s)
    return s


def score_jd_aware(df: pd.DataFrame, filters_cfg, facets_cfg) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    pos = facets_cfg["positive_facets"]
    neg = facets_cfg["negative_facets"]
    w = facets_cfg["weights"]
    pp = facets_cfg["penalty_params"]

    pos_counts = _facet_counts(df, pos)
    neg_counts = _facet_counts(df, neg)

    # positive facet coverage, weighted + normalized
    P = np.zeros(len(df))
    wsum = 0.0
    facet_scores = {}
    for fac in pos:
        cap = fac.get("cap", 2)
        fs = np.clip(pos_counts[fac["id"]] / cap, 0, 1)
        facet_scores[fac["id"]] = fs
        P += fac["weight"] * fs
        wsum += fac["weight"]
    P = P / wsum

    # experience
    E = _experience_score(df["yoe"].to_numpy(), facets_cfg["experience"])

    # title
    strong = df["title"].apply(lambda t: any(s in t for s in facets_cfg["strong_titles"])).to_numpy()
    offdom = df["title"].apply(lambda t: any(s in t for s in facets_cfg["offdomain_titles"])).to_numpy()
    T = np.where(offdom & ~strong, 0.0, np.where(strong, 1.0, 0.4))

    # product vs services
    Prod = np.where(df["has_product_role"].to_numpy(), 1.0, np.where(df["services_all"].to_numpy(), 0.4, 0.6))

    # location tiers
    loc = df["location"]
    pl = facets_cfg["preferred_locations"]
    best = loc.str.contains(_regex(pl["best"]), regex=True, na=False).to_numpy()
    good = loc.str.contains(_regex(pl["good"]), regex=True, na=False).to_numpy()
    okl = loc.str.contains(_regex(pl["ok"]), regex=True, na=False).to_numpy()
    L = np.where(best, 1.0, np.where(good, 0.8, np.where(okl, 0.7, 0.4)))

    base = w["facets"] * P + w["experience"] * E + w["title"] * T + w["product"] * Prod + w["location"] * L

    # ---- multiplicative penalties ----
    penalty = np.ones(len(df))
    flags: Dict[str, np.ndarray] = {}

    # keyword stuffer: off-domain title + many AI skills (the "Marketing Manager w/ all AI skills" trap)
    stuffer = offdom & ~strong & (df["ai_core_skill_count"].to_numpy() >= 5)
    penalty = np.where(stuffer, penalty * pp["keyword_stuffer"], penalty)
    flags["keyword_stuffer"] = stuffer

    # CV/speech/robotics primary without NLP/IR rescue
    for fac in neg:
        has_neg = neg_counts[fac["id"]] > 0
        rescue = np.zeros(len(df), dtype=bool)
        for r in fac.get("rescue_facets", []):
            rescue |= facet_scores.get(r, np.zeros(len(df))) > 0
        hit = has_neg & ~rescue
        penalty = np.where(hit, penalty * fac["penalty"], penalty)
        flags[fac["id"]] = hit

    # title chaser
    chaser = (df["num_roles"].to_numpy() >= pp["title_chaser_min_roles"]) & (
        df["avg_tenure_months"].to_numpy() < pp["title_chaser_max_avg_tenure_months"]
    )
    penalty = np.where(chaser, penalty * pp["title_chaser"], penalty)
    flags["title_chaser"] = chaser

    # ---- availability / engagement modifier (Redrob behavioral signals) ----
    # The JD is explicit: "a perfect-on-paper candidate who hasn't logged in for 6
    # months and has a 5% recruiter response rate is, for hiring purposes, not
    # actually available. Down-weight them appropriately." We combine the signals
    # that proxy for reachability (response, recency, open-to-work), reliability
    # (interview completion, offer acceptance), demand (recruiter saves) and
    # profile trust (completeness, github, verification) into one bounded multiplier.
    rr = df["recruiter_response_rate"].to_numpy()
    comp = df["completeness"].to_numpy()
    stale = df["recency_months"].to_numpy() > pp["stale_months"]
    very_stale = df["recency_months"].to_numpy() > (pp["stale_months"] * 2)
    notice = df["notice_period_days"].to_numpy()
    # reliability: NaN (no history) is treated as neutral, not penalized
    icr = df["interview_completion_rate"].to_numpy()
    oar = df["offer_acceptance_rate"].to_numpy()
    icr_adj = np.where(np.isnan(icr), 0.0, icr - 0.7)
    oar_adj = np.where(np.isnan(oar), 0.0, oar - 0.5)
    # recruiter demand: saved-by-recruiters in last 30d, gently rewarded (log, capped)
    saved = df["saved_by_recruiters_30d"].to_numpy()
    demand = np.clip(np.log1p(saved) / np.log1p(20.0), 0, 1)
    modifier = (
        1.0
        + 0.10 * (rr - 0.4)
        + 0.06 * (comp - 0.7)
        + 0.04 * df["github"].to_numpy()
        + 0.03 * (df["open_to_work"].to_numpy().astype(float) - 0.5)
        + 0.04 * icr_adj
        + 0.03 * oar_adj
        + 0.03 * (demand - 0.3)
        + 0.02 * (df["verified"].to_numpy() / 2.0 - 0.5)
        - 0.08 * stale.astype(float)
        - 0.08 * very_stale.astype(float)
        - 0.05 * (notice > 60).astype(float)
    )
    modifier = np.clip(modifier, 0.7, 1.25)
    flags["stale"] = stale

    final = np.clip(base * penalty * modifier, 0, 1)

    info = {"facet_scores": facet_scores, "flags": flags, "P": P, "E": E, "T": T, "Prod": Prod, "L": L}
    return final, info


# --------------------------------------------------------------------------- #
# Semantic recall + ensemble
# --------------------------------------------------------------------------- #
def tfidf_scores(df: pd.DataFrame, query: str, max_features: int = 40000) -> np.ndarray:
    return tfidf_multi(df, {"q": query}, max_features)["q"]


def tfidf_multi(df: pd.DataFrame, queries: Dict[str, str], max_features: int = 40000) -> Dict[str, np.ndarray]:
    """Fit TF-IDF once, score several JD-facet queries against all candidates.

    Cheap multi-query recall: one sparse fit + one sparse matmul per query, all
    BLAS/multi-core. Lets us ensemble several JD 'blocks' (retrieval, ranking/eval,
    product/LLM) instead of one blurry mega-query.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    doc_matrix = vec.fit_transform(df["search_text"].tolist())
    qmat = vec.transform(list(queries.values()))
    sims = linear_kernel(qmat, doc_matrix)  # (n_queries, n_docs)
    return {name: sims[i] for i, name in enumerate(queries.keys())}


def rrf_merge(score_arrays: List[np.ndarray], weights: List[float] | None = None, k: int = 60) -> np.ndarray:
    weights = weights or [1.0] * len(score_arrays)
    fused = np.zeros(len(score_arrays[0]))
    for scores, wt in zip(score_arrays, weights):
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(len(scores))
        fused += wt / (k + ranks + 1.0)
    return fused
