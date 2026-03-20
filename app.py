import streamlit as st
from openai import OpenAI
from rag_pipeline import build_index, generate_answer
import posthog
import uuid
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Knowledge Assistant")
st.caption("Enterprise RAG with role-based access control")

# ── PostHog setup ────────────────────────────────────────────────
posthog.project_api_key = st.secrets["POSTHOG_API_KEY"]
posthog.host = "https://us.i.posthog.com"

# ── Session ID — unique per browser session ──────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.subheader("Select Your Role")
    role = st.radio(
        "Role",
        options=["student", "practitioner", "trainer"],
        captions=[
            "Access: Level 1 only",
            "Access: Levels 1-3",
            "Access: All levels"
        ]
    )
    st.divider()
    st.markdown("""
    **Role Access:**
    - 🎓 Student: Level 1 only
    - 🏥 Practitioner: Levels 1-3
    - 👨‍🏫 Trainer: All levels
    """)

# ── Load API key from Streamlit secrets ──────────────────────────
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# ── Build index once per session ─────────────────────────────────
@st.cache_resource
def load_index(api_key):
    with st.spinner("Building knowledge index... (~30 seconds)"):
        collection, embeddings = build_index(api_key)
    return collection, embeddings

collection, embeddings = load_index(api_key)

# ── Chat history — separate per role ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = {
        "student": [],
        "practitioner": [],
        "trainer": []
    }

# ── Message counter per role for session depth ───────────────────
if "message_counts" not in st.session_state:
    st.session_state.message_counts = {
        "student": 0,
        "practitioner": 0,
        "trainer": 0
    }

# ── Display chat history for current role ────────────────────────
for message in st.session_state.messages[role]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# ── Chat input ───────────────────────────────────────────────────
if query := st.chat_input("Ask a question..."):

    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages[role].append({"role": "user", "content": query})
    st.session_state.message_counts[role] += 1
    message_number = st.session_state.message_counts[role]

    # Generate answer
    error_occurred = None
    result = None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = generate_answer(
                    query, role, collection, embeddings, client,
                    chat_history=st.session_state.messages[role]
                )
                st.write(result["answer"])
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write(f"- {source}")
                with st.expander("Query rewritten to"):
                    st.caption(result["rewritten_query"])

            except Exception as e:
                error_occurred = str(e)
                st.error(f"Something went wrong: {error_occurred}")

    # ── PostHog event logging ────────────────────────────────────
    query_was_rewritten = (
        result is not None and
        result["rewritten_query"].strip().lower() != query.strip().lower()
    )

    posthog.capture(
        distinct_id=st.session_state.session_id,
        event="query_asked",
        properties={
            "session_id": st.session_state.session_id,
            "role": role,
            "original_query": query,
            "rewritten_query": result["rewritten_query"] if result else None,
            "query_was_rewritten": query_was_rewritten,
            "sources_used": result["sources"] if result else [],
            "answer_length": len(result["answer"]) if result else 0,
            "message_number_in_session": message_number,
            "error": error_occurred,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Save assistant message
    if result:
        st.session_state.messages[role].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
