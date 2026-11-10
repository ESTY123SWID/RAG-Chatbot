"""בדיקת עשן לממשק: האפליקציה נטענת בלי שגיאות, וכשאין מפתח — מוצגת הודעה ותיבת השאלה מושבתת."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_without_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "התקנון שלנו" in html
    assert "שואלות, מבררות ומקבלות תשובות מתוך מסמכי בית הספר" in html
    assert "מפתח OpenAI לא הוגדר" in html
    assert at.chat_input[0].disabled
    assert at.button[0].disabled  # כפתור ההוספה למאגר מושבת


def test_vague_and_empty_questions_need_no_api():
    """שאלה עמומה/ריקה נענית מקומית לפני כל קריאה ל-OpenAI."""
    from rag_engine import RAGEngine

    engine = RAGEngine.__new__(RAGEngine)  # ללא אתחול מודלים
    assert engine.ask("מה?").kind == "clarify"
    assert engine.ask("  ").kind == "empty"
