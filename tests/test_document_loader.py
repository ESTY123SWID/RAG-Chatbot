"""בדיקות טעינת מסמכים ומצבי קצה — ללא קריאות ל-OpenAI."""

from pathlib import Path

import pytest

from config import DATA_DIR
from document_loader import DocumentLoadError, load_document, normalize_pdf_text


def test_normalize_pdf_text():
    assert normalize_pdf_text("רשאית להיכנס עד20  דקות") == "רשאית להיכנס עד 20 דקות"
    assert normalize_pdf_text("פרק4: היעדרויות") == "פרק 4: היעדרויות"
    assert normalize_pdf_text("בין 08:00 ל-08:45") == "בין 08:00 ל-08:45"  # ללא שינוי
    assert normalize_pdf_text("מחלה של 3 .ימים רצופים") == "מחלה של 3 ימים רצופים"
    assert normalize_pdf_text("4.5 ,פעילות מטעם") == "4.5 פעילות מטעם"
    assert normalize_pdf_text("סוף משפט. התחלה") == "סוף משפט. התחלה"  # פיסוק תקין נשמר


def test_txt_hebrew_utf8(tmp_path: Path):
    f = tmp_path / "נהלים.txt"
    f.write_text("שלום לכולן.\nזהו נוהל לדוגמה.", encoding="utf-8")
    docs = load_document(f)
    assert len(docs) == 1
    assert "נוהל לדוגמה" in docs[0].text
    assert docs[0].metadata["file_name"] == "נהלים.txt"
    assert "page_label" not in docs[0].metadata  # אין עמודים בקובץ טקסט


def test_txt_windows_hebrew_encoding(tmp_path: Path):
    f = tmp_path / "old.txt"
    f.write_bytes("תקנון ישן".encode("cp1255"))
    assert "תקנון ישן" in load_document(f)[0].text


def test_display_name_overrides_disk_name(tmp_path: Path):
    f = tmp_path / ".upload_abc.txt"
    f.write_text("טקסט", encoding="utf-8")
    assert load_document(f, display_name="מקור.txt")[0].metadata["file_name"] == "מקור.txt"


def test_empty_file(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    with pytest.raises(DocumentLoadError, match="ריק"):
        load_document(f)


def test_whitespace_only_file_has_no_text(tmp_path: Path):
    f = tmp_path / "spaces.txt"
    f.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="לא נמצא טקסט"):
        load_document(f)


def test_unsupported_extension(tmp_path: Path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG....")
    with pytest.raises(DocumentLoadError, match="אינו נתמך"):
        load_document(f)


def test_corrupted_pdf(tmp_path: Path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentLoadError, match="PDF"):
        load_document(f)


def test_truncated_pdf(tmp_path: Path):
    f = tmp_path / "truncated.pdf"
    f.write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog >> garbage")
    with pytest.raises(DocumentLoadError):
        load_document(f)


def test_corrupted_docx(tmp_path: Path):
    f = tmp_path / "broken.docx"
    f.write_bytes(b"not a zip file at all")
    with pytest.raises(DocumentLoadError, match="DOCX"):
        load_document(f)


# ---- בדיקות על מסמכי הדוגמה האמיתיים בתיקיית data ----

SAMPLE_PDF = DATA_DIR / "תקנון_נוכחות_ואיחורים.pdf"
SAMPLE_DOCX = DATA_DIR / "נוהל_הגשת_עבודות_ומבחנים.docx"
SAMPLE_TXT = DATA_DIR / "כללי_התנהגות_ונהלים.txt"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="מסמך הדוגמה לא נוצר")
def test_sample_pdf_pages_and_hebrew_order():
    docs = load_document(SAMPLE_PDF)
    assert [d.metadata["page_label"] for d in docs] == ["1", "2", "3"]
    full = " ".join(d.text for d in docs)
    # סדר מילים לוגי (לא הפוך) — חיוני לחיפוש ולתשובות
    assert "טופס הצדקת היעדרות" in full
    assert "20 דקות" in full


@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="מסמך הדוגמה לא נוצר")
def test_sample_docx():
    docs = load_document(SAMPLE_DOCX)
    assert "מועד ב'" in docs[0].text
    assert docs[0].metadata["file_type"] == "docx"


@pytest.mark.skipif(not SAMPLE_TXT.exists(), reason="מסמך הדוגמה לא נוצר")
def test_sample_txt():
    assert "פינת אבדות" in load_document(SAMPLE_TXT)[0].text
