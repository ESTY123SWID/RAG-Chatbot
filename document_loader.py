"""טעינת מסמכים: זיהוי סוג הקובץ, קריאה באמצעות Reader מתאים של LlamaIndex,
וחילוץ הטקסט וה-Metadata (שם קובץ, מספר עמוד ב-PDF)."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from llama_index.core import Document
from llama_index.readers.file import DocxReader, PDFReader

from config import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS


class DocumentLoadError(Exception):
    """שגיאה בטעינת מסמך. ההודעה מנוסחת בעברית ומיועדת להצגה למשתמשת."""


def compute_file_hash(path: Path) -> str:
    """טביעת אצבע של תוכן הקובץ — משמשת לזיהוי קובץ שכבר נטען ולמזהים עקביים."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_file(path: Path) -> str:
    """בדיקות בסיסיות לפני קריאה. מחזיר את סיומת הקובץ."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"הקובץ '{path.name}' אינו נתמך. ניתן להעלות רק קובצי PDF, DOCX או TXT."
        )
    if not path.exists():
        raise DocumentLoadError(f"הקובץ '{path.name}' לא נמצא.")
    size = path.stat().st_size
    if size == 0:
        raise DocumentLoadError(f"הקובץ '{path.name}' ריק.")
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DocumentLoadError(
            f"הקובץ '{path.name}' גדול מדי (מעל {MAX_FILE_SIZE_MB}MB)."
        )

    # זיהוי קובץ פגום או קובץ שהסיומת שלו אינה תואמת לתוכן
    with path.open("rb") as f:
        head = f.read(5)
    if ext == ".pdf" and not head.startswith(b"%PDF"):
        raise DocumentLoadError(f"הקובץ '{path.name}' אינו קובץ PDF תקין.")
    if ext == ".docx" and not zipfile.is_zipfile(path):
        raise DocumentLoadError(f"הקובץ '{path.name}' אינו קובץ Word (DOCX) תקין.")
    return ext


def _read_txt(path: Path) -> list[Document]:
    """קריאת קובץ טקסט. מנסה UTF-8 ואם לא — קידוד עברית של Windows."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentLoadError(
            f"לא ניתן לקרוא את הקובץ '{path.name}'. ודאי שהוא שמור בקידוד UTF-8."
        )
    return [Document(text=text)]


_HEB_THEN_DIGIT = re.compile(r"([א-ת])(\d)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# נקודה/פסיק שהודבקו לתחילת מילה עברית ('3 .ימים') — סימן פיסוק שזז בגלל היפוך כיווניות
_MISPLACED_PUNCT = re.compile(r"(?<!\S)[.,](?=[א-ת])")


def normalize_pdf_text(text: str) -> str:
    """תיקון קל לטקסט עברי שחולץ מ-PDF:
    - pypdf משמיט לעיתים את הרווח בין מילה עברית למספר ('עד20' במקום 'עד 20').
    - נקודה או פסיק עוברים לתחילת המילה שאחרי מספר ('של 3 .ימים'), כך שנוצר
      סוף משפט מדומה. בעברית סימן כזה אינו מופיע בתחילת מילה, ולכן הוא מוסר.
    - רווחים כפולים מאוחדים."""
    text = _HEB_THEN_DIGIT.sub(r"\1 \2", text)
    text = _MISPLACED_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines())


def _read_pdf(path: Path) -> list[Document]:
    # PDFReader מחזיר מסמך נפרד לכל עמוד, עם page_label ב-Metadata
    docs = PDFReader(return_full_document=False).load_data(file=path)
    for doc in docs:
        doc.set_content(normalize_pdf_text(doc.text))
    return docs


def _read_docx(path: Path) -> list[Document]:
    return DocxReader().load_data(file=path)


_READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}


def load_document(path: Path, display_name: str | None = None) -> list[Document]:
    """קורא קובץ ומחזיר רשימת Documents של LlamaIndex עם Metadata אחיד.

    display_name — שם הקובץ המקורי שיוצג במקורות (ברירת מחדל: שם הקובץ בדיסק).
    זורק DocumentLoadError עם הודעה ברורה בעברית בכל מקרה של כשל.
    """
    path = Path(path)
    file_name = display_name or path.name
    ext = _check_file(path)

    try:
        raw_docs = _READERS[ext](path)
    except DocumentLoadError:
        raise
    except Exception as exc:  # קובץ פגום שהספרייה לא הצליחה לפתוח
        raise DocumentLoadError(
            f"לא ניתן לפתוח את הקובץ '{file_name}'. ייתכן שהוא פגום."
        ) from exc

    file_hash = compute_file_hash(path)
    documents: list[Document] = []
    for doc in raw_docs:
        text = (doc.text or "").replace("\x00", "").strip()
        if not text:
            continue  # עמוד ריק (למשל עמוד סרוק ללא שכבת טקסט)
        metadata = {
            "file_name": file_name,
            "file_type": ext.lstrip("."),
            "file_hash": file_hash,
        }
        page = doc.metadata.get("page_label")
        if page not in (None, ""):
            metadata["page_label"] = str(page)
        documents.append(Document(text=text, metadata=metadata))

    if not documents:
        raise DocumentLoadError(
            f"לא נמצא טקסט שניתן לקרוא בקובץ '{file_name}'. "
            "ייתכן שמדובר במסמך סרוק (תמונה) — המערכת אינה מבצעת זיהוי טקסט מתמונות."
        )
    return documents
