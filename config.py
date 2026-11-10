"""הגדרות המערכת — מקום אחד לכל הערכים הקבועים של הפרויקט."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# טעינת משתני סביבה מקובץ .env (אם קיים). המפתח עצמו לעולם לא נכתב בקוד.
load_dotenv(BASE_DIR / ".env")

# ---- תיקיות ----
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "school_documents"

# ---- מודלים (החלטות סופיות) ----
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
OPENAI_TIMEOUT_SECONDS = 60
OPENAI_MAX_RETRIES = 2

# ---- Chunking ----
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ---- Retrieval ----
TOP_K = 5

# ---- סוגי קבצים נתמכים ----
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 20

# ---- הודעות קבועות ----
NO_INFO_MESSAGE = "לא נמצא מידע מספק במסמכים שהועלו כדי לענות על השאלה."


def make_http_client():
    """לקוח HTTP לתקשורת עם OpenAI.

    האימות המלא של תעודות האבטחה נשמר (שרשרת אמון + שם שרת). מבוטל רק הדגל
    VERIFY_X509_STRICT שנוסף כברירת מחדל ב-Python 3.13: רשתות עם סינון תוכן
    (למשל NetFree) משתמשות בתעודת CA שאינה עומדת בבדיקה המחמירה הזו,
    ובלי התיקון כל פנייה ל-OpenAI נכשלת. ההתנהגות זהה לזו של Python 3.12.
    """
    import ssl

    import httpx

    context = ssl.create_default_context()
    context.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)
    return httpx.Client(verify=context, timeout=OPENAI_TIMEOUT_SECONDS)


def get_api_key() -> str | None:
    """מחזיר את מפתח OpenAI ממשתנה הסביבה, או None אם אינו מוגדר."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("your-") or key == "sk-...":
        return None
    return key
