from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict
from search_engine.configs import (
    WhooshConfig,
    QdrantConfig,
    RapidFuzzConfig,
    SubstringConfig,
    HybridConfig,
    TFIDFConfig,
    ComprehensiveConfig,
)

import hashlib
import json
import uuid

class SearchableDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = None
    parent_doc_id: str
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    original_json_obj: Any = None
    original_json_obj_hash: str = None

    def __init__(self, **data):
        # Compute original_json_obj_hash if not provided
        if "original_json_obj_hash" not in data or data["original_json_obj_hash"] is None:
            original_json_obj = data.get("original_json_obj")
            data["original_json_obj_hash"] = self._hash_obj(original_json_obj)
        # Compute chunk_id if not provided
        if "chunk_id" not in data or data["chunk_id"] is None:
            parent_id = data.get("parent_doc_id", "")
            text_content = data.get("text_content", "")
            content_type = None
            if "metadata" in data and isinstance(data["metadata"], dict):
                content_type = data["metadata"].get("content_type")
            chunk_id_source = f"{parent_id}:{content_type}:{text_content}"
            data["chunk_id"] = self._hash_text(chunk_id_source)
        super().__init__(**data)

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

class EngineDefinition(BaseModel):
    id: str
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StrategyConfigsModel(BaseModel):
    keyword: Optional[WhooshConfig] = Field(default_factory=WhooshConfig)
    semantic: Optional[QdrantConfig] = Field(default_factory=QdrantConfig)
    fuzzy: Optional[RapidFuzzConfig] = Field(default_factory=RapidFuzzConfig)
    hybrid: Optional[HybridConfig] = Field(default_factory=HybridConfig)
    substring: Optional[SubstringConfig] = Field(default_factory=SubstringConfig)
    tfidf: Optional[TFIDFConfig] = Field(default_factory=TFIDFConfig)
    comprehensive: Optional[ComprehensiveConfig] = Field(default_factory=ComprehensiveConfig)

class SearchEngineServiceConfig(BaseModel):
    strategy_name: str
    custom_engine_definitions: list[EngineDefinition] = Field(default_factory=list)
    configs: StrategyConfigsModel = Field(default_factory=dict)

class SearchEngineConfig(BaseModel):
    global_config: Optional[SearchEngineServiceConfig] = Field(default_factory=SearchEngineServiceConfig, alias="global")
    services: Optional[Dict[str, SearchEngineServiceConfig]] = Field(default_factory=dict)
