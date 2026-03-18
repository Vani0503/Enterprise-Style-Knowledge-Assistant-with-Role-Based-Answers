
import streamlit as st
from openai import OpenAI
from rag_pipeline import build_index, generate_answer

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="EKAA Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 EKAA Knowledge Assistant")
st.caption("Enterprise RAG with role-based access control")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    st.divider()
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

# ── Index builder ────────────────────────────────────────────────
@st.cache_resource
def load_index(api_key):
    with st.spinner("Building knowledge index... (~30 seconds)"):
        collection, embeddings = build_index(api_key)
    return collection, embeddings

# ── Main interface ───────────────────────────────────────────────
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

collection, embeddings = load_index(api_key)
client = OpenAI(api_key=api_key)

st.success(f"Knowledge base ready. Logged in as: **{role}**")

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the theory of mind in hypnotherapy?"
)

if query:
    with st.spinner("Retrieving and generating answer..."):
        result = generate_answer(query, role, collection, embeddings, client)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Sources used"):
        for source in result["sources"]:
            st.write(f"- {source}")

    with st.expander("Retrieved chunks with scores"):
        for i, chunk in enumerate(result["ranked_chunks"]):
            st.markdown(f"**Rank {i+1}: {chunk['metadata']['title']}**")
            st.caption(
                f"Semantic: {chunk['semantic_score']} | "
                f"Authority: {chunk['authority_score']} | "
                f"Freshness: {chunk['freshness_score']} | "
                f"Final: {chunk['final_score']}"
            )
            st.write(chunk["text"][:300] + "...")
            st.divider()
