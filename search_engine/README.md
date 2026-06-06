# search_engine

A small, self-contained, **offline** local search library used by this project as a
reusable reference layer. Candidates are turned into `SearchableDocument`s (see
`canjob/candidate_adapter.py`) and indexed by one or more pluggable strategies.

It has no external services and no API keys. Configuration lives in plain pydantic
classes in `configs.py`; the defaults are sufficient, so no JSON config file is needed.

> The production ranker (`rank.py`) uses a direct vectorized scikit-learn path for
> speed, and the semantic signal comes from precomputed MiniLM scores. This module is
> the reusable, strategy-based search layer that the candidate adapter builds on.

## Strategies

All strategies are local and offline:

| name | class | what it does |
|---|---|---|
| `substring` | `SubstringSearchStrategy` | exact/substring matching |
| `fuzzy` | `RapidFuzzSearchStrategy` | typo-tolerant fuzzy matching (RapidFuzz) |
| `keyword` | `WhooshSearchStrategy` | BM25 / inverted index (Whoosh) |
| `tfidf` | `TFIDFSearchStrategy` | TF-IDF cosine similarity (scikit-learn) |
| `hybrid` | `HybridSearchStrategy` | weighted blend of strategies |
| `comprehensive` | `ComprehensiveSearchStrategy` | Reciprocal Rank Fusion over the above |

The original framework also shipped a Qdrant + Gemini vector strategy. It is
intentionally **not** bundled here because this project is fully offline; calling it
raises a clear error. Use the lexical strategies, or the precomputed MiniLM scores in
the ranker.

## Usage

```python
from search_engine.configs import TFIDFConfig
from search_engine.models import SearchableDocument
from search_engine.strategies import TFIDFSearchStrategy

docs = [
    SearchableDocument(parent_doc_id="CAND_1", text_content="senior ml engineer, retrieval, recsys"),
    SearchableDocument(parent_doc_id="CAND_2", text_content="frontend react developer"),
]

strategy = TFIDFSearchStrategy(TFIDFConfig(), service_adapter=None)
strategy.upsert_documents(docs)
hits = strategy.search("retrieval and ranking", limit=5)
```

A `SearchableDocument` keeps the original record on `original_json_obj`, so after
retrieval you can recover the full candidate profile for downstream re-ranking.

To plug your own data source, subclass `Adapter` (`adapter.py`) and implement
`db_to_searchable_documents()`; `init_from_db` / `sync_from_db` then keep the index in
sync with your data.

## Files

```
configs.py      pydantic config classes (defaults baked in)
models.py       SearchableDocument and related models
adapter.py      Adapter base class (data source -> documents)
strategies.py   the search strategies listed above
```
