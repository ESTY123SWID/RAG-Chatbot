"""תהליך Ingestion: קובץ ← טקסט + Metadata ← Chunks ← Embeddings ← ChromaDB.

הרצה ישירה מהטרמינל טוענת את כל הקבצים מתיקיית data שעדיין לא נמצאים במאגר:
    python ingestion.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_MODEL,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
    SUPPORTED_EXTENSIONS,
    get_api_key,
    make_http_client,
)
from document_loader import DocumentLoadError, compute_file_hash, load_document
from errors import MISSING_KEY_MESSAGE, friendly_openai_error
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# שדות Metadata טכניים שלא נכנסים לטקסט שעובר Embedding או נשלח למודל
_HIDDEN_METADATA = ["file_hash", "file_type"]


@dataclass
class IngestResult:
    file_name: str
    status: str  # "added" | "updated" | "duplicate" | "error"
    message: str
    chunks: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("added", "updated", "duplicate")


def make_embed_model(api_key: str) -> OpenAIEmbedding:
    return OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
        http_client=make_http_client(),
    )


def make_splitter() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


class Ingestor:
    def __init__(self, store: VectorStoreManager, embed_model: BaseEmbedding, data_dir: Path = DATA_DIR):
        self.store = store
        self.data_dir = Path(data_dir)
        self.splitter = make_splitter()
        self.index = VectorStoreIndex.from_vector_store(store.vector_store, embed_model=embed_model)

    # ------------------------------------------------------------------
    def ingest_file(self, path: Path, display_name: str | None = None) -> IngestResult:
        """טעינת קובץ אחד מהדיסק למאגר."""
        path = Path(path)
        name = display_name or path.name
        try:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise DocumentLoadError(
                    f"הקובץ '{name}' אינו נתמך. ניתן להעלות רק קובצי PDF, DOCX או TXT."
                )
            if path.exists() and path.stat().st_size > 0:
                file_hash = compute_file_hash(path)
                if self.store.has_file_hash(file_hash):
                    return IngestResult(name, "duplicate", f"הקובץ '{name}' כבר נמצא במאגר — לא נוסף שוב.")
            documents = load_document(path, display_name=name)
        except DocumentLoadError as exc:
            return IngestResult(name, "error", str(exc))

        file_hash = documents[0].metadata["file_hash"]
        is_update = self.store.has_file_name(name)

        for doc in documents:
            doc.excluded_embed_metadata_keys = list(_HIDDEN_METADATA)
            doc.excluded_llm_metadata_keys = list(_HIDDEN_METADATA)

        nodes = self.splitter.get_nodes_from_documents(documents)
        # מזהים עקביים: אותו תוכן ← אותם מזהים
        for i, node in enumerate(nodes):
            node.id_ = f"{file_hash[:16]}-{i:04d}"

        try:
            self.index.insert_nodes(nodes)  # יוצר Embeddings ושומר ב-ChromaDB
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", name, type(exc).__name__)
            return IngestResult(name, "error", f"עיבוד הקובץ '{name}' נכשל: {friendly_openai_error(exc)}")

        if is_update:
            # הגרסה החדשה נשמרה בהצלחה — מוחקים רק את המקטעים של הגרסה הקודמת
            self.store.collection.delete(
                where={"$and": [{"file_name": name}, {"file_hash": {"$ne": file_hash}}]}
            )
            return IngestResult(name, "updated", f"הקובץ '{name}' עודכן לגרסה החדשה.", len(nodes))
        return IngestResult(name, "added", f"הקובץ '{name}' נוסף בהצלחה למאגר.", len(nodes))

    # ------------------------------------------------------------------
    def ingest_upload(self, file_name: str, data: bytes) -> IngestResult:
        """שמירת קובץ שהועלה בממשק לתיקיית data והוספתו למאגר."""
        safe_name = Path(file_name).name  # מניעת נתיבים זדוניים בשם הקובץ
        if not data:
            return IngestResult(safe_name, "error", f"הקובץ '{safe_name}' ריק.")
        if self.store.has_file_hash(hashlib.sha256(data).hexdigest()):
            return IngestResult(safe_name, "duplicate", f"הקובץ '{safe_name}' כבר נמצא במאגר — לא נוסף שוב.")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(safe_name).suffix.lower()
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=self.data_dir, prefix=".upload_")
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            result = self.ingest_file(tmp_path, display_name=safe_name)
            if result.status in ("added", "updated"):
                # שומרים את הקובץ המקורי רק אם נקלט בהצלחה
                os.replace(tmp_path, self.data_dir / safe_name)
            return result
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def ingest_directory(self, directory: Path | None = None) -> list[IngestResult]:
        directory = Path(directory or self.data_dir)
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
        return [self.ingest_file(p) for p in files]


def main() -> None:
    api_key = get_api_key()
    if not api_key:
        print(MISSING_KEY_MESSAGE)
        return
    store = VectorStoreManager()
    ingestor = Ingestor(store, make_embed_model(api_key))
    for r in ingestor.ingest_directory():
        extra = f" ({r.chunks} מקטעים)" if r.chunks else ""
        print(f"[{r.status}] {r.message}{extra}")
    print(f"סה\"כ מסמכים במאגר: {len(store.list_documents())}, מקטעים: {store.chunk_count()}")


if __name__ == "__main__":
    main()
