"""יצירת מסמכי הדוגמה הבדיוניים בתיקיית data (PDF, DOCX, TXT).

הרצה (מתיקיית הפרויקט):
    pip install -r requirements-dev.txt
    python scripts/create_sample_docs.py

ה-PDF נוצר מקובץ DOCX באמצעות Microsoft Word המותקן במחשב (Windows בלבד),
כדי לקבל PDF עם שכבת טקסט עברית תקינה.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_content import DOCUMENTS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FONT = "Arial"


def _rtl_paragraph(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.append(OxmlElement("w:bidi"))


def _rtl_run(run, size: int, bold: bool = False) -> None:
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = FONT
    r_pr = run._r.get_or_add_rPr()
    fonts = r_pr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.append(fonts)
    fonts.set(qn("w:cs"), FONT)
    r_pr.append(OxmlElement("w:rtl"))
    size_cs = OxmlElement("w:szCs")
    size_cs.set(qn("w:val"), str(size * 2))
    r_pr.append(size_cs)
    if bold:
        r_pr.append(OxmlElement("w:bCs"))


def build_docx(blocks, path: Path) -> None:
    doc = Document()
    for kind, text in blocks:
        if kind == "pagebreak":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            continue
        paragraph = doc.add_paragraph()
        _rtl_paragraph(paragraph)
        size, bold = {"title": (17, True), "h": (13, True)}.get(kind, (11, False))
        _rtl_run(paragraph.add_run(text), size, bold)
    doc.save(path)


def build_txt(blocks, path: Path) -> None:
    lines: list[str] = []
    for kind, text in blocks:
        if kind == "pagebreak":
            continue
        if kind == "title":
            lines += [text, "=" * 40, ""]
        elif kind == "h":
            lines += ["", text, "-" * 25]
        else:
            lines += [text, ""]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def docx_to_pdf_with_word(docx_path: Path, pdf_path: Path) -> None:
    """המרה באמצעות Word (COM דרך PowerShell). הנתיבים מועברים במשתני סביבה בגלל תווים בעברית."""
    script = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        "try { $d = $w.Documents.Open($env:SRC_DOCX, $false, $true); "
        "$d.SaveAs2($env:DST_PDF, 17); $d.Close($false) } finally { $w.Quit() }"
    )
    env = {**os.environ, "SRC_DOCX": str(docx_path), "DST_PDF": str(pdf_path)}
    subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env, check=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name, blocks in DOCUMENTS.items():
        target = DATA_DIR / name
        if target.suffix == ".txt":
            build_txt(blocks, target)
        elif target.suffix == ".docx":
            build_docx(blocks, target)
        elif target.suffix == ".pdf":
            with tempfile.TemporaryDirectory() as tmp:
                tmp_docx = Path(tmp) / "source.docx"
                build_docx(blocks, tmp_docx)
                docx_to_pdf_with_word(tmp_docx, target)
        print(f"נוצר: {target.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
