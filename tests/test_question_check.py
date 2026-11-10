"""בדיקות זיהוי שאלה ריקה / עמומה מאוד / תקינה — היוריסטיקה מקומית."""

import pytest

from rag_engine import _is_no_info, check_question
from config import NO_INFO_MESSAGE


@pytest.mark.parametrize("q", ["", "   ", "\n"])
def test_empty(q):
    assert check_question(q).status == "empty"


@pytest.mark.parametrize("q", ["?", "מה?", "איך?", "תגידי לי משהו", "שאלה", "עזרה בבקשה", "מה זה?", "היי", "...", "מה עושים?"])
def test_vague(q):
    assert check_question(q).status == "vague"


@pytest.mark.parametrize("q", [
    "איחורים?",                       # קצרה אך מובנת
    "הגשה באיחור",
    "מה הנהלים לגבי איחור לשיעור?",
    "כיצד מגישים בקשה לאישור היעדרות?",
    "איבדתי את הקלמר, מה עושים?",
    "מועד ב'",
])
def test_ok(q):
    assert check_question(q).status == "ok"


def test_no_info_detection():
    assert _is_no_info(NO_INFO_MESSAGE)
    assert _is_no_info(f'"{NO_INFO_MESSAGE}"')
    # תשובה חלקית שמכילה מידע אמיתי אינה נחשבת "אין מידע"
    assert not _is_no_info("איחור מוגדר עד 10 דקות. " + NO_INFO_MESSAGE)


def test_split_citations():
    from rag_engine import split_citations

    assert split_citations("תשובה.\nמקורות: 1, 3") == ("תשובה.", [1, 3])
    assert split_citations("תשובה.\n\n[מקורות: 2]") == ("תשובה.", [2])
    assert split_citations("תשובה ללא ציון מקורות") == ("תשובה ללא ציון מקורות", None)
