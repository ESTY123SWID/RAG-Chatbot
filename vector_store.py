"""גישה ל-ChromaDB המקומי (Persistent) ועטיפה שלו כ-ChromaVectorStore של LlamaIndex."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_DIR, COLLECTION_NAME


@dataclass
class StoredDocument:
    file_name: str
    file_hash: str
    chunks: int


class VectorStoreManager:
    """מנהל את האוסף ב-ChromaDB: פתיחה, בדיקת כפילויות ורשימת מסמכים."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # מרחק Cosine — מתאים ל-Embeddings של OpenAI
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

    # ---- מידע על המאגר ----
    def chunk_count(self) -> int:
        return self.collection.count()

    def is_empty(self) -> bool:
        return self.chunk_count() == 0

    def list_documents(self) -> list[StoredDocument]:
        """רשימת המסמכים הייחודיים במאגר (לפי Metadata של המקטעים)."""
        if self.is_empty():
            return []
        result = self.collection.get(include=["metadatas"])
        docs: dict[str, StoredDocument] = {}
        for meta in result["metadatas"] or []:
            if not meta:
                continue
            key = meta.get("file_hash", meta.get("file_name", "?"))
            if key not in docs:
                docs[key] = StoredDocument(
                    file_name=meta.get("file_name", "ללא שם"), file_hash=key, chunks=0
                )
            docs[key].chunks += 1
        return sorted(docs.values(), key=lambda d: d.file_name)

    # ---- כפילויות ועדכונים ----
    def has_file_hash(self, file_hash: str) -> bool:
        found = self.collection.get(where={"file_hash": file_hash}, limit=1)
        return bool(found["ids"])

    def has_file_name(self, file_name: str) -> bool:
        found = self.collection.get(where={"file_name": file_name}, limit=1)
        return bool(found["ids"])

    def delete_file_name(self, file_name: str) -> None:
        """מחיקת גרסה קודמת של קובץ באותו שם (משמש רק כאשר מועלית גרסה מעודכנת)."""
        self.collection.delete(where={"file_name": file_name})
