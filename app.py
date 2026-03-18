import streamlit as st
from openai import OpenAI
from rag_pipeline import build_index, generate_answer

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Knowledge Assistant")
st.caption("Enterprise RAG with role-based access control")

# ── Sidebar — role selection only ────────────────────────────────
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

# ── Chat history ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_role" not in st.session_state:
    st.session_state.current_role = role

# Reset chat if role changes
if st.session_state.current_role != role:
    st.session_state.messages = []
    st.session_state.current_role = role
    st.info(f"Role changed to {role}. Starting fresh conversation.")

# ── Display chat history ─────────────────────────────────────────
for message in st.session_state.messages:
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
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_answer(
                query, role, collection, embeddings, client,
                chat_history=st.session_state.messages
            )
        st.write(result["answer"])
        with st.expander("Sources"):
            for source in result["sources"]:
                st.write(f"- {source}")
        with st.expander("Query rewritten to"):
            st.caption(result["rewritten_query"])

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
