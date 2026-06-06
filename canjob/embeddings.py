"""Local, offline sentence embeddings (no network, no sentence-transformers dep).

Uses a cached HuggingFace MiniLM via `transformers` + mean pooling. Candidate
embeddings are expensive (~1-3 min for 100k on CPU) so they are computed once and
cached to disk; the ranking step then just loads them + encodes the 1 JD query.
This mirrors the hackathon rule: precompute may exceed the 5-min budget, the
ranking step must not.
"""

from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tok = None
_mdl = None


def _lazy_model():
    global _tok, _mdl
    if _mdl is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(os.cpu_count() or 4)
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _mdl = AutoModel.from_pretrained(MODEL_NAME)
        _mdl.eval()
    return _tok, _mdl


def embed_texts(texts: List[str], batch_size: int = 256, max_length: int = 192) -> np.ndarray:
    """Encode texts to L2-normalized mean-pooled vectors.

    Speedup with zero quality loss: sort by length first so each padded batch holds
    similar-length texts (transformer cost is O(batch * seqlen^2), so padding short
    texts up to a long one in the same batch is pure waste). We encode in sorted order,
    then scatter results back to the original positions. Same model, same max_length,
    same pooling => identical embeddings, just less wasted compute.
    """
    import torch

    tok, mdl = _lazy_model()
    n = len(texts)
    order = sorted(range(n), key=lambda i: len(texts[i]))
    out_vecs: List[np.ndarray] = [None] * n  # type: ignore[list-item]
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = [texts[i] for i in idx]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            hidden = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            mean = torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy().astype(np.float32)
            for j, i in enumerate(idx):
                out_vecs[i] = mean[j]
    return np.vstack(out_vecs)


def candidate_embeddings(texts: List[str], cache_dir: str) -> np.ndarray:
    """Compute (or load cached) L2-normalized embeddings for candidate texts."""
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.md5(("\u0001".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"emb_{len(texts)}_{digest}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    emb = embed_texts(texts)
    np.save(cache, emb)
    return emb


def query_score(cand_emb: np.ndarray, query_text: str) -> np.ndarray:
    q = embed_texts([query_text])[0]
    return cand_emb @ q  # cosine (both L2-normalized)
