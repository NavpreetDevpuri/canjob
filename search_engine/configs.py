import os
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
import logging

# Default English stop words
DEFAULT_STOP_WORDS = [
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how"
]


class BaseSearchConfig(BaseModel):
    """Base configuration shared by all search strategies."""
    
    default_limit: int = Field(
        default=100,
        description="Maximum number of results to return. Higher values return more results but may include less relevant matches."
    )
    log_level: int = Field(
        default=logging.ERROR,
        description="Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR). Lower values produce more verbose logs."
    )


class WhooshConfig(BaseSearchConfig):
    """
    Configuration for Whoosh keyword search strategy.
    
    Whoosh is a fast, pure-Python full-text indexing and search library.
    It excels at exact keyword matching with support for stemming and stop words.
    
    USE WHEN: You need fast, exact keyword matching with linguistic features.
    
    Example:
        config = WhooshConfig(analyzer_type="stem", use_stop_words=True)
        # Query "running shoes" matches "run", "ran", "runner" via stemming
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STORAGE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    use_ram_storage: bool = Field(
        default=True,
        description="Store index in RAM (fast, lost on restart) vs disk (persistent, slower). "
                    "Use RAM for temporary/session data, disk for persistent indexes."
    )
    index_dir: Optional[str] = Field(
        default=None,
        description="Directory path for disk-based storage. Only used when use_ram_storage=False. "
                    "If None, creates temp directory. Example: '/var/data/search_index'"
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYZER CONFIGURATION - How text is tokenized and processed
    # ═══════════════════════════════════════════════════════════════════════════
    analyzer_type: Literal["standard", "simple", "regex", "stem", "keyword", "ngram"] = Field(
        default="stem",
        description="Text analyzer type. EFFECTS: "
                    "• 'standard': Splits on whitespace/punctuation, lowercases. Good baseline. "
                    "• 'simple': Basic whitespace splitting only. Fast but less accurate. "
                    "• 'regex': Custom regex-based tokenization. For special formats. "
                    "• 'stem': Reduces words to root form ('running'→'run'). BEST FOR RECALL. "
                    "• 'keyword': No tokenization, treats entire field as one token. For IDs/codes. "
                    "• 'ngram': Character n-grams for partial matching. Slower but catches substrings."
    )
    stemming_language: Optional[str] = Field(
        default="english",
        description="Language for stemming (only used with analyzer_type='stem'). "
                    "Stems words to root form: 'running','ran','runs' all become 'run'. "
                    "Options: 'english', 'spanish', 'french', 'german', 'portuguese', etc."
    )
    use_stop_words: bool = Field(
        default=True,
        description="Filter out common words (the, is, at, which, on). "
                    "True = reduces noise, faster search. False = matches common words exactly."
    )
    stop_words: Optional[List[str]] = Field(
        default=None,
        description="Custom stop words list. If None, uses DEFAULT_STOP_WORDS. "
                    "Example: ['the', 'a', 'an', 'is', 'are'] to customize filtering."
    )
    min_word_length: int = Field(
        default=2,
        description="Minimum token length to index. Tokens shorter than this are ignored. "
                    "2 = ignores 'a', 'I'. Higher values skip more short words."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # N-GRAM CONFIGURATION (for analyzer_type='ngram')
    # ═══════════════════════════════════════════════════════════════════════════
    ngram_min: int = Field(
        default=2,
        description="Minimum n-gram size for ngram analyzer. 'apple' with min=2: 'ap','pp','pl','le'. "
                    "Smaller = more matches but slower and more storage."
    )
    ngram_max: int = Field(
        default=4,
        description="Maximum n-gram size for ngram analyzer. 'apple' with max=4: 'appl','pple'. "
                    "Larger = matches longer substrings but increases index size."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HYBRID N-GRAM TYPO TOLERANCE - Dual-field indexing for catching typos
    # ═══════════════════════════════════════════════════════════════════════════
    enable_ngram_typo_matching: bool = Field(
        default=False,
        description="Enable hybrid n-gram typo matching. Indexes content TWICE: "
                    "1) Exact field (stemmed) for precise matches "
                    "2) N-gram field (trigrams) for typo tolerance. "
                    "Example: 'chocolat' matches 'chocolate' via shared trigrams. "
                    "TRADEOFF: 2x index size, slightly slower indexing, better typo handling."
    )
    ngram_typo_size: int = Field(
        default=3,
        description="N-gram size for typo matching (only when enable_ngram_typo_matching=True). "
                    "3 (trigrams) works best: 'chocolate'→['cho','hoc','oco','col','ola','lat','ate']. "
                    "Typo 'chocolat' shares 6 trigrams, enabling the match."
    )
    exact_field_boost: float = Field(
        default=2.0,
        description="Boost multiplier for exact match field in hybrid search. "
                    "Higher = prioritize exact matches over n-gram matches. "
                    "2.0 means exact matches score 2x higher than n-gram matches."
    )
    ngram_field_boost: float = Field(
        default=1.0,
        description="Boost multiplier for n-gram field in hybrid search. "
                    "Lower than exact_field_boost ensures typo matches rank below exact matches."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY PARSING
    # ═══════════════════════════════════════════════════════════════════════════
    default_field: str = Field(
        default="text_content",
        description="Default field to search when no field is specified in query. "
                    "Queries without field prefix search this field."
    )
    query_default_operator: Literal["AND", "OR"] = Field(
        default="OR",
        description="Default operator for multi-word queries. "
                    "• 'AND': 'Manila trip' requires BOTH words to match. More precise, fewer results. "
                    "• 'OR': 'Manila trip' matches if EITHER word is present. Better recall, more results. "
                    "RECOMMENDATION: Use 'OR' for user-facing search to maximize matches."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORING/RANKING - How search results are ranked
    # ═══════════════════════════════════════════════════════════════════════════
    scoring_type: Literal["bm25f", "tfidf", "frequency"] = Field(
        default="bm25f",
        description="Scoring algorithm for ranking results. "
                    "• 'bm25f': Best Matching 25 (recommended). Balances term frequency, doc length, IDF. "
                    "• 'tfidf': Term Frequency-Inverse Document Frequency. Classic IR scoring. "
                    "• 'frequency': Simple term frequency count. Fast but less accurate."
    )
    bm25_b: float = Field(
        default=0.75,
        description="BM25 length normalization parameter (0-1). Only used with scoring_type='bm25f'. "
                    "0 = no length normalization (long docs not penalized). "
                    "1 = full normalization (strongly penalizes long documents). "
                    "0.75 is standard default that works well for most cases."
    )
    bm25_k1: float = Field(
        default=1.2,
        description="BM25 term frequency saturation parameter. Only used with scoring_type='bm25f'. "
                    "Controls how quickly term frequency reaches saturation. "
                    "Higher = more weight to repeated terms. 1.2 is standard default."
    )


class QdrantConfig(BaseSearchConfig):
    """
    Configuration for Qdrant vector search with Gemini embeddings.
    
    Uses Google's Gemini API to generate embeddings, then searches using
    cosine similarity in Qdrant's in-memory vector database.
    
    USE WHEN: You need TRUE semantic search (synonyms, related concepts).
    Example: "athletic footwear" matches "running shoes" via meaning, not words.
    
    REQUIRES: GEMINI_API_KEY environment variable.
    
    Example:
        config = QdrantConfig(score_threshold=0.8)
        # Query "footwear" matches docs about "shoes", "sneakers", "boots"
    """
    
    score_threshold: float = Field(
        default=0.90,
        description="Minimum cosine similarity score (0-1) for a match. "
                    "0.9 = very strict, only near-identical meanings. "
                    "0.7 = moderate, allows related concepts. "
                    "0.5 = loose, may include tangentially related results. "
                    "EFFECT: Lower = more results but potentially less relevant."
    )
    model_name: str = Field(
        default="models/gemini-embedding-2-preview",
        description="Gemini embedding model name. 'text-embedding-004' is the latest/best. "
                    "Different models have different embedding dimensions and quality."
    )
    embedding_task_type: str = Field(
        default="RETRIEVAL_DOCUMENT",
        description="Embedding task type for Gemini API. "
                    "• 'RETRIEVAL_DOCUMENT': Optimized for document retrieval (recommended). "
                    "• 'RETRIEVAL_QUERY': Optimized for queries. "
                    "• 'SEMANTIC_SIMILARITY': For comparing text similarity."
    )
    embedding_size: int = Field(
        default=768,
        description="Dimension of embedding vectors. Must match the model's output size. "
                    "text-embedding-004 outputs 768-dimensional vectors."
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        description="Gemini API key. Reads from GEMINI_API_KEY or GOOGLE_API_KEY env var. "
                    "Required for generating embeddings. Get from Google AI Studio."
    )
    cache_file: Optional[str] = Field(
        default="default_qdrant_cache.tmp",
        description="File path for embedding cache. Caches embeddings to avoid repeated API calls. "
                    "Set to None to disable caching. Reduces API costs and latency."
    )
    max_cache_size: int = Field(
        default=10000,
        description="Maximum number of embeddings to cache. LRU eviction when exceeded. "
                    "Higher = more cache hits but more memory usage."
    )


class RapidFuzzConfig(BaseSearchConfig):
    """
    Configuration for RapidFuzz fuzzy string matching.
    
    Uses edit distance and token-based algorithms to find similar strings.
    Excellent for catching typos, misspellings, and OCR errors.
    
    USE WHEN: You need typo tolerance without semantic understanding.
    Example: "Pyhton" matches "Python", "recieve" matches "receive"
    
    Example:
        config = RapidFuzzConfig(scorer="WRatio", score_cutoff=70)
        # Query "Javascrpt" matches "JavaScript" (missing 'i')
    """
    
    scorer: str = Field(
        default="WRatio",
        description="Fuzzy matching algorithm from rapidfuzz library. "
                    "• 'WRatio': Weighted ratio - best all-around, handles word order. "
                    "• 'ratio': Simple Levenshtein ratio. Fast but sensitive to order. "
                    "• 'partial_ratio': Matches substrings. Good for 'contains' matching. "
                    "• 'token_sort_ratio': Ignores word order. 'hello world' = 'world hello'. "
                    "• 'token_set_ratio': Ignores duplicates and order. Most lenient."
    )
    score_cutoff: int = Field(
        default=70,
        description="Minimum fuzzy match score (0-100) to include a result. "
                    "100 = exact match only. "
                    "80 = strict, 1-2 character differences. "
                    "70 = balanced, handles most typos (RECOMMENDED). "
                    "60 = loose, may include false positives. "
                    "EFFECT: Lower = more results but more noise."
    )


class SubstringConfig(BaseSearchConfig):
    """
    Configuration for simple substring matching.
    
    The most basic search: checks if query is contained anywhere in text.
    Very fast but no linguistic intelligence.
    
    USE WHEN: You need exact substring matching, like searching for IDs or codes.
    Example: Query "abc" matches "xyzabcdef"
    
    Example:
        config = SubstringConfig(case_sensitive=False)
        # Query "python" matches "Learn Python Programming"
    """
    
    case_sensitive: bool = Field(
        default=False,
        description="Whether substring matching is case-sensitive. "
                    "False = 'Python' matches 'python', 'PYTHON'. "
                    "True = exact case required."
    )


class HybridConfig(BaseSearchConfig):
    """
    Configuration for hybrid search combining Qdrant semantic + RapidFuzz fuzzy.
    
    Merges results from both strategies for combined semantic + typo handling.
    
    USE WHEN: You need both semantic understanding AND typo tolerance.
    
    Example:
        config = HybridConfig()
        # Query "atheltic footware" matches "running shoes" 
        # (semantic: footwear→shoes, fuzzy: atheltic→athletic)
    """
    
    qdrant_config: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Configuration for the Qdrant semantic search component. "
                    "Handles synonym/concept matching via embeddings."
    )
    rapidfuzz_config: RapidFuzzConfig = Field(
        default_factory=RapidFuzzConfig,
        description="Configuration for the RapidFuzz fuzzy search component. "
                    "Handles typos and misspellings."
    )


class TFIDFConfig(BaseSearchConfig):
    """
    Configuration for TF-IDF (Term Frequency-Inverse Document Frequency) search.
    
    A lightweight, local alternative to embedding-based semantic search.
    Uses statistical word importance rather than AI embeddings.
    
    IMPORTANT: TF-IDF requires WORD OVERLAP to match!
    "athletic footwear" will NOT match "running shoes" (no shared words).
    For true semantic matching, use QdrantConfig instead.
    
    USE WHEN: You want fast, local search without API calls, and queries
    share vocabulary with documents.
    
    Example:
        config = TFIDFConfig(score_threshold=0.05)
        # Query "programming language" matches "Python programming language basics"
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TF-IDF VECTORIZER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    max_features: int = Field(
        default=10000,
        description="Maximum vocabulary size (number of unique terms to keep). "
                    "Higher = more terms indexed, better recall, more memory. "
                    "Lower = faster, less memory, may miss rare terms."
    )
    ngram_range_min: int = Field(
        default=1,
        description="Minimum n-gram size for TF-IDF features. "
                    "1 = single words ('python', 'code'). "
                    "2 = bigrams only ('python code'). "
                    "1 recommended for good single-word matching."
    )
    ngram_range_max: int = Field(
        default=2,
        description="Maximum n-gram size for TF-IDF features. "
                    "1 = single words only. "
                    "2 = includes bigrams ('machine learning'). "
                    "3 = includes trigrams (more memory, diminishing returns). "
                    "2 recommended for phrase matching without explosion."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    use_stop_words: bool = Field(
        default=True,
        description="Remove common English stop words before vectorizing. "
                    "True = ignores 'the', 'is', 'at', focuses on content words. "
                    "False = includes all words."
    )
    stop_words: Optional[List[str]] = Field(
        default=None,
        description="Custom stop words list. If None, uses sklearn's English stop words. "
                    "Provide custom list to override, e.g., ['the', 'a', 'an']."
    )
    lowercase: bool = Field(
        default=True,
        description="Convert all text to lowercase before vectorizing. "
                    "True = 'Python' and 'python' are the same. "
                    "False = case-sensitive matching."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIMILARITY SCORING
    # ═══════════════════════════════════════════════════════════════════════════
    score_threshold: float = Field(
        default=0.05,
        description="Minimum cosine similarity score (0-1) between query and document TF-IDF vectors. "
                    "0.05 = low threshold, returns documents with minimal word overlap. "
                    "0.1 = moderate, requires more shared vocabulary. "
                    "0.2+ = strict, requires significant overlap. "
                    "EFFECT: Lower = more results, higher = more relevant but fewer."
    )
    sublinear_tf: bool = Field(
        default=True,
        description="Use sublinear term frequency scaling: log(1 + tf) instead of raw tf. "
                    "True = diminishing returns for repeated terms (10 occurrences ≠ 10x importance). "
                    "False = linear scaling (rare, usually worse results)."
    )


class ComprehensiveConfig(BaseSearchConfig):
    """
    Configuration for Comprehensive Search combining 4 strategies with RRF fusion.
    
    Combines the best of all worlds:
    1. KEYWORD (Whoosh): Fast exact matching with stemming
    2. TF-IDF: Lightweight statistical similarity (no API needed)
    3. QDRANT: True semantic search via Gemini embeddings (API needed)
    4. FUZZY (RapidFuzz): Typo and misspelling tolerance
    
    Results are merged using Reciprocal Rank Fusion (RRF) with configurable weights.
    
    USE WHEN: You want maximum recall and can afford the computational cost.
    
    Example:
        # Full power - all 4 strategies
        config = ComprehensiveConfig()
        
        # Fast mode - no API calls
        config = ComprehensiveConfig(enable_qdrant_semantic=False)
        
        # Keyword + Fuzzy only (fastest)
        config = ComprehensiveConfig(enable_tfidf=False, enable_qdrant_semantic=False)
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUB-STRATEGY CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    whoosh_config: WhooshConfig = Field(
        default_factory=WhooshConfig,
        description="Configuration for keyword search sub-strategy. "
                    "Handles exact keyword matching with stemming and stop words."
    )
    tfidf_config: TFIDFConfig = Field(
        default_factory=TFIDFConfig,
        description="Configuration for TF-IDF sub-strategy. "
                    "Provides lightweight statistical similarity without API calls."
    )
    qdrant_config: QdrantConfig = Field(
        default_factory=lambda: QdrantConfig(score_threshold=0.8),
        description="Configuration for Qdrant semantic search sub-strategy. "
                    "Provides true semantic understanding via Gemini embeddings. "
                    "Default score_threshold=0.8 is stricter than standalone Qdrant."
    )
    rapidfuzz_config: RapidFuzzConfig = Field(
        default_factory=RapidFuzzConfig,
        description="Configuration for fuzzy matching sub-strategy. "
                    "Handles typos and misspellings via edit distance algorithms."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY ENABLE/DISABLE FLAGS
    # ═══════════════════════════════════════════════════════════════════════════
    enable_keyword: bool = Field(
        default=True,
        description="Enable Whoosh keyword search strategy. "
                    "Fast, exact matching with stemming. No API needed."
    )
    enable_tfidf: bool = Field(
        default=True,
        description="Enable TF-IDF statistical similarity strategy. "
                    "Lightweight semantic-like matching. No API needed. "
                    "Requires word overlap (not true semantic)."
    )
    enable_qdrant_semantic: bool = Field(
        default=False,
        description="Enable Qdrant semantic search with Gemini embeddings. "
                    "True semantic understanding (synonyms, concepts). "
                    "REQUIRES: GEMINI_API_KEY. Adds latency for API calls."
    )
    qdrant_api_key_missing_severity: Literal["warning", "error"] = Field(
        default="warning",
        description="Behavior when enable_qdrant_semantic=True but no "
                    "GEMINI_API_KEY is available at strategy initialization. "
                    "'warning' (default): emit a Python warning, skip the "
                    "Qdrant sub-strategy for this engine instance, and "
                    "continue with the other enabled sub-strategies "
                    "(keyword/TF-IDF/fuzzy) under RRF. "
                    "'error': raise a ValueError immediately so misconfigured "
                    "deployments fail fast. "
                    "Ignored when enable_qdrant_semantic=False."
    )
    enable_fuzzy: bool = Field(
        default=True,
        description="Enable RapidFuzz fuzzy matching strategy. "
                    "Catches typos and misspellings. No API needed."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RRF (RECIPROCAL RANK FUSION) WEIGHTS
    # Higher weight = more influence on final ranking
    # ═══════════════════════════════════════════════════════════════════════════
    keyword_weight: float = Field(
        default=1.0,
        description="Weight for keyword search results in RRF fusion. "
                    "1.0 = baseline weight. Higher = keyword matches rank higher."
    )
    tfidf_weight: float = Field(
        default=0.8,
        description="Weight for TF-IDF results in RRF fusion. "
                    "0.8 = slightly lower than keyword (not true semantic)."
    )
    qdrant_semantic_weight: float = Field(
        default=1.5,
        description="Weight for Qdrant semantic results in RRF fusion. "
                    "1.5 = highest weight because semantic matches are most valuable. "
                    "Ensures 'footwear' matching 'shoes' ranks highly."
    )
    fuzzy_weight: float = Field(
        default=0.8,
        description="Weight for fuzzy matching results in RRF fusion. "
                    "0.8 = lower weight because fuzzy matches may be false positives."
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESULT FILTERING
    # ═══════════════════════════════════════════════════════════════════════════
    min_rrf_score: float = Field(
        default=0.012,
        description="Minimum RRF score to include a result. Filters out noise. "
                    "0 = include all results. "
                    "0.012 = allows single-strategy top matches (e.g., fuzzy-only). "
                    "0.02+ = requires multiple strategies to agree. "
                    "EFFECT: Higher = fewer but more confident results."
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant (k in formula: 1/(k+rank)). "
                    "60 = standard value, balanced ranking. "
                    "Lower = top ranks matter more. "
                    "Higher = lower ranks get more weight."
    )


current_dir = os.path.dirname(os.path.abspath(__file__))

# Optional JSON overrides. The defaults baked into the config classes above are
# sufficient for this project, so the file is not required.
_cfg_path = os.path.join(current_dir, "search_engine_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as f:
        SEARCH_ENGINE_CONFIG = json.load(f)
else:
    SEARCH_ENGINE_CONFIG = {}

def get_default_strategy_name(service_name: str) -> str:
    """
    Gets the default search strategy name for a service, falling back to the global default.
    """
    service_config = SEARCH_ENGINE_CONFIG.get("services", {}).get(service_name, {})
    global_config = SEARCH_ENGINE_CONFIG.get("global", {})
    return service_config.get("default_strategy_name", global_config.get("default_strategy_name", "substring"))

def get_custom_engine_definitions(service_name: str) -> List[Dict]:
    global_config = SEARCH_ENGINE_CONFIG.get("global", {})
    return SEARCH_ENGINE_CONFIG.get("services", {}).get(service_name, {}).get("custom_engine_definitions", global_config.get("custom_engine_definitions", []))

def get_strategy_configs(service_name: str) -> Dict:
    global_config = SEARCH_ENGINE_CONFIG.get("global", {})
    return SEARCH_ENGINE_CONFIG.get("services", {}).get(service_name, {}).get("strategy_configs", global_config.get("strategy_configs", {}))


__all__ = [
    "DEFAULT_STOP_WORDS",
    "WhooshConfig",
    "QdrantConfig",
    "RapidFuzzConfig",
    "SubstringConfig",
    "HybridConfig",
    "TFIDFConfig",
    "ComprehensiveConfig",
    "get_default_strategy_name",
    "get_custom_engine_definitions",
    "get_strategy_configs",
]
