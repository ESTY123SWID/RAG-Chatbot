"""ממשק Streamlit — "התקנון שלנו". הרצה: streamlit run app.py"""

from __future__ import annotations

import html

import streamlit as st

from config import get_api_key
from errors import MISSING_KEY_MESSAGE
from ingestion import IngestResult, Ingestor, make_embed_model
from rag_engine import RAGAnswer, RAGEngine, Source
from vector_store import VectorStoreManager

st.set_page_config(page_title="התקנון שלנו", page_icon="📘", layout="centered")

# ----------------------------------------------------------------------
# עיצוב: RTL, צבעים בהירים (לבן, תכלת, סגול בהיר), כרטיסיות מעוגלות
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700&display=swap');

:root {
  --purple: #7c6fd6;
  --purple-soft: #efeafd;
  --sky: #e6f4fd;
  --sky-strong: #4aa3df;
  --text: #2d2a3e;
  --muted: #6b6880;
  --border: #e4e0f5;
}

html, body, .stApp, [data-testid="stSidebar"], button, input, textarea {
  font-family: 'Heebo', 'Segoe UI', Arial, sans-serif !important;
}
.stApp { background: linear-gradient(180deg, #f7f5ff 0%, #f2f9fe 100%); color: var(--text); }

/* כיווניות מימין לשמאל */
[data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stChatInput"],
[data-testid="stBottom"], .stMarkdown, .stAlert, [data-testid="stExpander"],
[data-testid="stFileUploader"], textarea, input { direction: rtl; text-align: right; }
[data-testid="stChatMessage"] { direction: rtl; text-align: right; }
[data-testid="stChatMessageContent"] { text-align: right; }
.stMarkdown ul, .stMarkdown ol { padding-right: 1.4rem; padding-left: 0; }

/* כותרת */
.hero {
  background: linear-gradient(135deg, #ffffff 0%, var(--purple-soft) 55%, var(--sky) 100%);
  border: 1px solid var(--border); border-radius: 22px;
  padding: 1.6rem 1.5rem; margin-bottom: 1.2rem; text-align: center;
  box-shadow: 0 6px 22px rgba(124, 111, 214, 0.10);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #4b3fa8; }
.hero p { margin: .4rem 0 0; font-size: 1.05rem; color: var(--muted); }

/* כרטיסיות הודעה */
.card {
  border-radius: 16px; padding: .9rem 1.1rem; margin: .5rem 0 1rem;
  border: 1px solid var(--border); background: #fff; line-height: 1.7;
}
.card.info    { background: var(--sky);  border-color: #c7e5f8; }
.card.warn    { background: #fff7e8;     border-color: #f6dfb0; }
.card.error   { background: #fdeef0;     border-color: #f5c6cd; }
.card.success { background: #edf9f1;     border-color: #bfe8cd; }

/* הודעות צ'אט */
[data-testid="stChatMessage"] {
  background: #ffffff; border: 1px solid var(--border);
  border-radius: 18px; padding: .8rem 1rem; margin-bottom: .7rem;
  box-shadow: 0 2px 10px rgba(80, 70, 160, 0.05);
}

/* מקורות */
.sources-title { margin: .8rem 0 .3rem; font-weight: 700; color: #4b3fa8; font-size: .95rem; }
[data-testid="stExpander"] details {
  border-radius: 12px !important; border: 1px solid var(--border) !important; background: #fbfaff;
}
.source-text {
  direction: rtl; text-align: right; white-space: pre-wrap; font-size: .92rem;
  color: #3d3a52; line-height: 1.75; max-height: 320px; overflow-y: auto;
}

/* סרגל צד */
[data-testid="stSidebar"] { background: #ffffff; border-left: 1px solid var(--border); }
.doc-item {
  background: var(--purple-soft); border-radius: 12px; padding: .45rem .7rem;
  margin: .3rem 0; font-size: .92rem; word-break: break-word;
}
.stat {
  background: var(--sky); border-radius: 14px; padding: .7rem; text-align: center;
  font-size: .95rem; margin: .4rem 0 .8rem;
}
.stat b { font-size: 1.5rem; color: var(--sky-strong); display: block; }

.stButton button {
  border-radius: 12px; background: var(--purple); color: #fff; border: none; font-weight: 500;
}
.stButton button:hover { background: #6a5cc9; color: #fff; }

@media (max-width: 640px) {
  .hero h1 { font-size: 1.7rem; }
  .hero { padding: 1.1rem .9rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(text: str, kind: str = "info", icon: str = "") -> None:
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="card {kind}">{icon} {safe}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# משאבים משותפים (נטענים פעם אחת לכל הפעלה)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def get_engine(api_key: str) -> RAGEngine:
    return RAGEngine(get_store(), api_key)


@st.cache_resource(show_spinner=False)
def get_ingestor(api_key: str) -> Ingestor:
    return Ingestor(get_store(), make_embed_model(api_key))


def render_sources(sources: list[Source]) -> None:
    if not sources:
        return
    st.markdown('<div class="sources-title">📚 מקורות המידע</div>', unsafe_allow_html=True)
    for i, src in enumerate(sources, start=1):
        label = f"📄 {src.file_name}"
        if src.page_label:
            label += f" · עמוד {src.page_label}"
        with st.expander(f"{i}. {label}"):
            st.markdown(
                f'<div class="source-text">{html.escape(src.text)}</div>',
                unsafe_allow_html=True,
            )


def render_message(msg: dict) -> None:
    avatar = "🙋‍♀️" if msg["role"] == "user" else "📘"
    with st.chat_message(msg["role"], avatar=avatar):
        kind = msg.get("kind", "answer")
        if msg["role"] == "user" or kind == "answer":
            st.markdown(msg["content"])
        elif kind == "error":
            card(msg["content"], "error", "⚠️")
        else:  # no_info / clarify / empty
            card(msg["content"], "warn", "💡")
        render_sources(msg.get("sources", []))


# ----------------------------------------------------------------------
# מצב התחלתי
# ----------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("upload_results", [])

api_key = get_api_key()
store = get_store()

# ----------------------------------------------------------------------
# סרגל צד — ניהול מסמכים
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 מסמכי בית הספר")

    documents = store.list_documents()
    st.markdown(
        f'<div class="stat"><b>{len(documents)}</b>מסמכים זמינים לחיפוש</div>',
        unsafe_allow_html=True,
    )
    if documents:
        for doc in documents:
            st.markdown(f'<div class="doc-item">📄 {html.escape(doc.file_name)}</div>', unsafe_allow_html=True)
    else:
        st.caption("עדיין לא נטענו מסמכים.")

    st.markdown("#### ➕ הוספת מסמכים")
    uploads = st.file_uploader(
        "בחרי קובצי PDF, Word או טקסט",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        disabled=not api_key,
    )
    if st.button("הוספה למאגר", disabled=not (api_key and uploads), use_container_width=True):
        ingestor = get_ingestor(api_key)
        results: list[IngestResult] = []
        with st.spinner("מעבדת את המסמכים... זה עשוי לקחת רגע"):
            for up in uploads:
                results.append(ingestor.ingest_upload(up.name, up.getvalue()))
        st.session_state.upload_results = results
        st.session_state.uploader_key += 1  # ניקוי רשימת הקבצים שנבחרו
        st.rerun()

    for r in st.session_state.upload_results:
        kind = {"added": "success", "updated": "success", "duplicate": "info"}.get(r.status, "error")
        icon = {"success": "✅", "info": "ℹ️", "error": "❌"}[kind]
        card(r.message, kind, icon)

    st.markdown("---")
    st.caption("המסמכים במאגר זה הם מסמכי דוגמה בדיוניים שנוצרו לצורך תרגול.")

# ----------------------------------------------------------------------
# אזור מרכזי — צ'אט
# ----------------------------------------------------------------------
st.markdown(
    """<div class="hero">
        <h1>📘 התקנון שלנו</h1>
        <p>שואלות, מבררות ומקבלות תשובות מתוך מסמכי בית הספר</p>
    </div>""",
    unsafe_allow_html=True,
)

can_ask = True
if not api_key:
    card(MISSING_KEY_MESSAGE, "error", "🔑")
    can_ask = False
elif store.is_empty():
    card("עדיין אין מסמכים במאגר. כדי להתחיל, העלי מסמך דרך הסרגל שבצד ולחצי על \"הוספה למאגר\".", "info", "📂")
    can_ask = False
elif not st.session_state.messages:
    card("שלום! אפשר לשאול כאן כל שאלה על התקנון והנהלים של בית הספר, "
         "והתשובה תתבסס רק על המסמכים שבמאגר.", "info", "👋")

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("כתבי כאן את השאלה שלך...", disabled=not can_ask)
if question is not None and can_ask:
    user_msg = {"role": "user", "content": question.strip() or "(שאלה ריקה)"}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("מחפשת תשובה במסמכים..."):
            result: RAGAnswer = get_engine(api_key).ask(question)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "kind": result.kind, "sources": result.sources}
    )
    st.rerun()
