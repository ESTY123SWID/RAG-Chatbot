"""בדיקות Ingestion ו-ChromaDB עם MockEmbedding (ללא OpenAI):
הוספה, כפילות, עדכון גרסה, מזהים עקביים ושמירה לאחר "הפעלה מחדש"."""

from pathlib import Path

from llama_index.core.embeddings import MockEmbedding

from ingestion import Ingestor
from vector_store import VectorStoreManager

TEXT = "סעיף 1: תלמידה שמאחרת נרשמת כמאחרת.\n\n" + ("זהו משפט נוסף בנוהל לדוגמה. " * 300)


def _ingestor(tmp_path: Path) -> Ingestor:
    store = VectorStoreManager(persist_dir=tmp_path / "db", collection_name="test")
    return Ingestor(store, MockEmbedding(embed_dim=8), data_dir=tmp_path / "data")


def test_add_duplicate_and_persistence(tmp_path: Path):
    ing = _ingestor(tmp_path)
    assert ing.store.is_empty()

    first = ing.ingest_upload("נוהל.txt", TEXT.encode("utf-8"))
    assert first.status == "added" and first.chunks > 1  # הטקסט ארוך ונחלק לכמה מקטעים
    assert (tmp_path / "data" / "נוהל.txt").exists()
    count = ing.store.chunk_count()
    assert count == first.chunks

    again = ing.ingest_upload("נוהל.txt", TEXT.encode("utf-8"))
    assert again.status == "duplicate"
    same_content_new_name = ing.ingest_upload("עותק.txt", TEXT.encode("utf-8"))
    assert same_content_new_name.status == "duplicate"
    assert ing.store.chunk_count() == count

    # "הפעלה מחדש": פתיחת חיבור חדש לאותה תיקייה
    reopened = VectorStoreManager(persist_dir=tmp_path / "db", collection_name="test")
    assert reopened.chunk_count() == count
    docs = reopened.list_documents()
    assert [d.file_name for d in docs] == ["נוהל.txt"]


def test_update_replaces_old_version(tmp_path: Path):
    ing = _ingestor(tmp_path)
    ing.ingest_upload("נוהל.txt", "גרסה ראשונה של הנוהל.".encode("utf-8"))
    result = ing.ingest_upload("נוהל.txt", "גרסה שנייה ומעודכנת של הנוהל.".encode("utf-8"))
    assert result.status == "updated"
    docs = ing.store.list_documents()
    assert len(docs) == 1
    stored = ing.store.collection.get(include=["documents"])["documents"]
    assert all("שנייה" in t for t in stored)


def test_consistent_ids(tmp_path: Path):
    ing = _ingestor(tmp_path)
    ing.ingest_upload("a.txt", TEXT.encode("utf-8"))
    ids_first = sorted(ing.store.collection.get()["ids"])

    other = VectorStoreManager(persist_dir=tmp_path / "db2", collection_name="test")
    Ingestor(other, MockEmbedding(embed_dim=8), data_dir=tmp_path / "data2").ingest_upload("a.txt", TEXT.encode("utf-8"))
    assert sorted(other.collection.get()["ids"]) == ids_first


def test_bad_uploads_are_not_saved(tmp_path: Path):
    ing = _ingestor(tmp_path)
    assert ing.ingest_upload("empty.txt", b"").status == "error"
    assert ing.ingest_upload("broken.pdf", b"not a pdf").status == "error"
    assert ing.ingest_upload("x.exe", b"MZ....").status == "error"
    assert ing.store.is_empty()
    data_dir = tmp_path / "data"
    assert not data_dir.exists() or not any(data_dir.iterdir())  # לא נשארו קבצים זמניים


def test_metadata_kept_on_chunks(tmp_path: Path):
    ing = _ingestor(tmp_path)
    ing.ingest_upload("נוהל.txt", TEXT.encode("utf-8"))
    metas = ing.store.collection.get(include=["metadatas"])["metadatas"]
    assert all(m["file_name"] == "נוהל.txt" for m in metas)
    assert all("file_hash" in m for m in metas)
