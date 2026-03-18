import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────
ROLE_ACCESS = {
    "student": 1,
    "practitioner": 3,
    "trainer": 5
}

AUTHORITY_SCORES = {
    "official_curriculum": 1.0,
    "internal_guide": 0.6,
    "draft": 0.3
}

FRESHNESS_SCORES = {
    "2022-12-01": 1.0,
    "2020-03-01": 0.7,
}

BOILERPLATE_PATTERNS = [
    "A Basic Course in Integrated Clinical Hypnotherapy",
    "INTEGRATED CLINICAL HYPNOTHERAPY FOUNDATION",
    "Integrated Hypnotic Modalities for Behavioral Resolutions",
    "EKAAL1OT", "EKAAL2OT", "EKAAL3OT",
    "www.ekaa.co.in", "admin@ekaa.co.in"
]

PDF_FILES = [
    {
        "path": "documents/ekaa_level1.pdf",
        "level": 1,
        "roles": ["student", "practitioner", "trainer"],
        "authority": "official_curriculum",
        "department": "training",
        "date": "2020-03-01",
        "title": "EKAA Level 1 Theory Manual"
    },
    {
        "path": "documents/ekaa_level2.pdf",
        "level": 2,
        "roles": ["practitioner", "trainer"],
        "authority": "official_curriculum",
        "department": "training",
        "date": "2020-03-01",
        "title": "EKAA Level 2 Theory Manual"
    },
    {
        "path": "documents/ekaa_level3.pdf",
        "level": 3,
        "roles": ["practitioner", "trainer"],
        "authority": "official_curriculum",
        "department": "clinical",
        "date": "2020-03-01",
        "title": "EKAA Level 3 Theory Manual"
    },
    {
        "path": "documents/ekaa_level5.pdf",
        "level": 5,
        "roles": ["trainer"],
        "authority": "official_curriculum",
        "department": "clinical",
        "date": "2022-12-01",
        "title": "EKAA Level 5 Clinical Manual"
    },
]

# ── Helpers ─────────────────────────────────────────────────────
def is_boilerplate(text):
    if len(text.strip()) < 200:
        return True
    hits = sum(1 for p in BOILERPLATE_PATTERNS if p in text)
    return hits >= 2

# ── Index Builder ───────────────────────────────────────────────
def build_index(openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("ekaa_knowledge_base")

    for doc in PDF_FILES:
        loader = PyPDFLoader(doc["path"])
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        clean = [c for c in chunks if not is_boilerplate(c.page_content)]
        texts = [c.page_content for c in clean]
        all_embeddings = embeddings.embed_documents(texts)

        for i, (chunk, embedding) in enumerate(zip(clean, all_embeddings)):
            chunk_id = f"{doc['title'].replace(' ', '_')}_chunk_{i}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[{
                    "title": doc["title"],
                    "level": doc["level"],
                    "roles": ",".join(doc["roles"]),
                    "authority": doc["authority"],
                    "department": doc["department"],
                    "date": doc["date"],
                }]
            )

    return collection, embeddings

# ── Query Rewriter ──────────────────────────────────────────────
def rewrite_query(query, chat_history, openai_client):
    """
    Rewrites a vague or follow-up query into a self-contained
    question using conversation history.
    Uses a principle-based approach — no word lists needed.
    """
    if not chat_history or len(chat_history) == 0:
        return query

    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_history[-6:]
    ])

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": """You are a query rewriter. Rewrite the user's latest question into a clear, self-contained question that can be understood without any conversation history.

Rules:
- If the question cannot be understood without history, rewrite it to be explicit and specific
- If the question is already self-contained, return it exactly as is
- Return ONLY the rewritten question, nothing else"""
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nLatest question: {query}\n\nRewritten question:"
            }
        ]
    )

    return response.choices[0].message.content.strip()

# ── Ret
