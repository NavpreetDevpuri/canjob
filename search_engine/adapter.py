from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Dict, TYPE_CHECKING

from search_engine.models import SearchableDocument

if TYPE_CHECKING:
    from search_engine.strategies import SearchStrategy


class Adapter(ABC):
    """
    Abstract base class for service-specific data adapters. It provides methods
    for converting a service's database into searchable documents and keeping
    the search index synchronized with the database state.
    """
    def __init__(self):
        self._strategy_to_last_searchable_documents: Dict[str, Dict[str, SearchableDocument]] = {}

    @abstractmethod
    def db_to_searchable_documents(self) -> List[SearchableDocument]:
        """
        Converts the service's database into a list of searchable documents.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def reset_from_db(self, strategy: SearchStrategy):
        """
        Clears the search index for the given strategy and re-initializes it
        from the database.
        """
        strategy.clear_index()
        self.init_from_db(strategy=strategy)

    def init_from_db(self, strategy: SearchStrategy):
        """
        Initializes the search index for the given strategy with all documents
        from the database.
        """
        added_searchable_documents = self.db_to_searchable_documents()
        strategy.upsert_documents(added_searchable_documents)
        # Set the last known state
        self._strategy_to_last_searchable_documents[strategy.name] = {chunk.chunk_id: chunk for chunk in added_searchable_documents}

    def sync_from_db(self, strategy: SearchStrategy):
        """
        Calculates the changes (added, updated, deleted) between the current
        database state and the last known state, and applies these changes
        to the search index for the given strategy.
        """
        changes = self.get_data_changes(strategy_name=strategy.name)
        if changes:
            if changes["added"]:
                strategy.upsert_documents(changes["added"])
            if changes["updated"]:
                strategy.upsert_documents(changes["updated"])
            if changes["deleted"]:
                strategy.delete_documents(changes["deleted"])

    def get_data_changes(self, strategy_name: str) -> dict[str, List[Any]]:
        """
        Compares the current set of searchable documents from the database with
        the last known state for a given strategy to identify additions,
        updates, and deletions.
        """
        # Regenerate all chunks from current DB
        current_searchable_documents = self.db_to_searchable_documents()
        current_searchable_documents_map = {chunk.chunk_id: chunk for chunk in current_searchable_documents}
        last_searchable_documents_map = self._strategy_to_last_searchable_documents.get(strategy_name, {})

        added = []
        updated = []
        deleted = []

        # Find added and updated
        for chunk_id, chunk in current_searchable_documents_map.items():
            if chunk_id not in last_searchable_documents_map:
                added.append(chunk)
            else:
                # Compare hash of original_json_obj
                if getattr(chunk, "original_json_obj_hash", None) != getattr(last_searchable_documents_map[chunk_id], "original_json_obj_hash", None):
                    updated.append(chunk)

        # Find deleted
        for chunk_id, chunk in last_searchable_documents_map.items():
            if chunk_id not in current_searchable_documents_map:
                deleted.append(chunk)

        # Update the last known state
        self._strategy_to_last_searchable_documents[strategy_name] = current_searchable_documents_map.copy()

        return {
            "added": added,
            "updated": updated,
            "deleted": deleted,
        }