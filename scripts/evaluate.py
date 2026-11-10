"""הרצת שאלות הבדיקה מול המערכת האמיתית (דורש מפתח OpenAI ומאגר טעון).

    python scripts/evaluate.py            # Top-K=5 (ברירת המחדל של המערכת)
    python scripts/evaluate.py --compare  # גם Top-K=2 להשוואה

התוצאות הגולמיות נשמרות ב-docs/evaluation_raw.md. ההערכה (האם התשובה נכונה ומבוססת)
נעשית ידנית על סמך הקובץ הזה ומתועדת ב-README.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import TOP_K, get_api_key  # noqa: E402
from errors import MISSING_KEY_MESSAGE  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402
from vector_store import VectorStoreManager  # noqa: E402

TEST_QUESTIONS = [
    ("1. תשובה ישירה במסמך", "מה קורה אם מגיעים לשיעור יותר מ-10 דקות אחרי שהתחיל?"),
    ("2. ניסוח שונה מהמסמך", "שכחתי את המעיל שלי בבית הספר, איפה אני יכולה למצוא אותו?"),
    ("3. מידע מכמה מקטעים", "הייתי חולה ארבעה ימים ופספסתי מבחן במתמטיקה. מה אני צריכה לעשות?"),
    ("4. שאלה קצרה אך מובנת", "הגשה באיחור?"),
    ("5. אין תשובה במסמכים", "כמה עולה ארוחת צהריים בקפטריה?"),
    ("6. תשובה חלקית", "באיזו שעה מתחילים ונגמרים הלימודים ביום רביעי?"),
    ("7. ניסוח שונה מהמסמך", "המורה לקחה לי את הפלאפון באמצע השיעור, מתי אקבל אותו בחזרה?"),
    ("8. שאלה עמומה מאוד", "מה עושים?"),
]


def run(engine: RAGEngine, top_k: int) -> list[str]:
    out = [f"\n## Top-K = {top_k}\n"]
    for label, q in TEST_QUESTIONS:
        ans = engine.ask(q)
        out.append(f"### {label}\n\n**שאלה:** {q}\n\n**סוג תוצאה:** `{ans.kind}`\n\n**תשובה:**\n\n{ans.text}\n")
        if ans.retrieved:
            out.append("**מקטעים שנשלפו (ציון דמיון):**\n")
            for i, s in enumerate(ans.retrieved, 1):
                page = f", עמוד {s.page_label}" if s.page_label else ""
                score = f"{s.score:.3f}" if s.score is not None else "-"
                snippet = s.text.replace("\n", " ")[:160]
                shown = "מוצג" if s in ans.sources else "לא מוצג"
                out.append(f"{i}. `{score}` — {s.file_name}{page} ({shown}) — {snippet}...")
            out.append("")
        print(f"[k={top_k}] {label}: {ans.kind}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="הרצה נוספת עם Top-K=2")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print(MISSING_KEY_MESSAGE)
        return
    store = VectorStoreManager()
    if store.is_empty():
        print("המאגר ריק. הריצי קודם: python ingestion.py")
        return

    lines = [f"# תוצאות בדיקה גולמיות\n\nהורץ ב-{datetime.now():%Y-%m-%d %H:%M}. "
             f"מסמכים במאגר: {len(store.list_documents())}, מקטעים: {store.chunk_count()}.\n"]
    lines += run(RAGEngine(store, api_key, top_k=TOP_K), TOP_K)
    if args.compare:
        lines += run(RAGEngine(store, api_key, top_k=2), 2)

    out_file = ROOT / "docs" / "evaluation_raw.md"
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"נשמר: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
