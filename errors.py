"""תרגום שגיאות טכניות (OpenAI / רשת) להודעות ברורות בעברית, בלי Traceback ובלי מפתחות."""

import openai

MISSING_KEY_MESSAGE = (
    "מפתח OpenAI לא הוגדר. יש ליצור קובץ ‎.env‎ בתיקיית הפרויקט "
    "ולהוסיף בו את השורה OPENAI_API_KEY=... ואז להפעיל מחדש את המערכת."
)


def friendly_openai_error(exc: BaseException) -> str:
    """מחזיר הודעה בעברית המתאימה לסוג השגיאה."""
    # LlamaIndex/tenacity עלולים לעטוף את השגיאה המקורית
    cause = exc
    for _ in range(5):
        if isinstance(cause, openai.OpenAIError) or cause.__cause__ is None:
            break
        cause = cause.__cause__

    if isinstance(cause, openai.AuthenticationError):
        return "מפתח OpenAI אינו תקין. יש לבדוק את הערך בקובץ ‎.env‎."
    if isinstance(cause, openai.PermissionDeniedError):
        return "אין הרשאה להשתמש במודל המבוקש עם מפתח ה-OpenAI הנוכחי."
    if isinstance(cause, openai.RateLimitError):
        return "חרגנו ממגבלת השימוש של OpenAI (או שנגמרה המכסה). נסי שוב בעוד מספר דקות."
    if isinstance(cause, (openai.APIConnectionError, openai.APITimeoutError, ConnectionError, TimeoutError)):
        return "לא ניתן להתחבר לשירות OpenAI. בדקי את חיבור האינטרנט ונסי שוב."
    if isinstance(cause, openai.APIStatusError):
        return f"שירות OpenAI החזיר שגיאה (קוד {cause.status_code}). נסי שוב מאוחר יותר."
    if isinstance(cause, openai.OpenAIError):
        return "אירעה שגיאה בתקשורת עם OpenAI. נסי שוב מאוחר יותר."
    return "אירעה שגיאה לא צפויה. נסי שוב, ואם הבעיה חוזרת — פני לצוות הטכני."
