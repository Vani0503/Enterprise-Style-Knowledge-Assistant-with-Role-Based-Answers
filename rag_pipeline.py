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
    Two-step approach:
    Step 1: LLM decides if rewriting is needed (yes/no)
    Step 2: Only if yes, LLM rewrites using history
    """
    if not chat_history or len(chat_history) == 0:
        return query

    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_history[-6:]
    ])

    # Step 1: Should this query be rewritten?
    check_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        messages=[
            {
                "role": "system",
                "content": """You decide if a query needs conversation history to be understood.

Answer only YES or NO.
YES: query refers to something from earlier conversation and cannot be understood standalone
NO: query is a complete standalone question that needs no prior context

Examples of YES: "summarise this", "tell me more", "what about that", "explain it further", "give me the gist", "in simpler terms", "elaborate", "break it down"
Examples of NO: "what is hypnosis", "tell me about suggestibility", "how does self hypnosis work" """
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nQuery: {query}\n\nDoes this query need history to be understood? Answer YES or NO only:"
            }
        ]
    )

    needs_rewrite = check_response.choices[0].message.content.strip().upper().startswith("YES")

    if not needs_rewrite:
        return query

    # Step 2: Rewrite using history
    rewrite_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {
                "role": "system",
                "content": """Rewrite the query by replacing vague references with the specific topic from conversation history.

Examples:
- "summarise this in 30 words" + history about hypnosis → "Summarise hypnosis in 30 words"
- "tell me more" + history about suggestibility → "Tell me more about suggestibility"
- "explain it further" + history about self hypnosis → "Explain self hypnosis further"

Return ONLY the rewritten question, nothing else."""
            },
            {
                "role": "user",
                "content": f"Conversation history:\n{history_text}\n\nQuery: {query}\n\nRewritten query:"
            }
        ]
    )

    return rewrite_response.choices[0].message.content.strip()

# ── Retrieval ───────────────────────────────────────────────────
def retrieve_for_role(query, role, collection, embeddings, n_results=8):
    if role not in ROLE_ACCESS:
        raise ValueError(f"Unknown role: {role}")
    max_level = ROLE_ACCESS[role]
    query_embedding = embeddings.embed_query(query)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"level": {"$lte": max_level}},
        include=["documents", "metadatas", "distances"]
    )

# ── Ranking ─────────────────────────────────────────────────────
def rank_results(results):
    ranked = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        semantic_score = 1 - (distance / 2)
        authority_score = AUTHORITY_SCORES.get(meta["authority"], 0.5)
        freshness_score = FRESHNESS_SCORES.get(meta["date"], 0.5)
        final_score = (
            0.7 * semantic_score +
            0.2 * authority_score +
            0.1 * freshness_score
        )
        ranked.append({
            "text": doc,
            "metadata": meta,
            "semantic_score": round(semantic_score, 3),
            "authority_score": authority_score,
            "freshness_score": freshness_score,
            "final_score": round(final_score, 3)
        })
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked

# ── Answer Generation ───────────────────────────────────────────
def generate_answer(query, role, collection, embeddings, openai_client, chat_history=None):

    # Step 1: Rewrite query using conversation history
    rewritten_query = rewrite_query(query, chat_history, openai_client)

    # Step 2: Retrieve using the REWRITTEN query
    results = retrieve_for_role(rewritten_query, role, collection, embeddings)
    ranked = rank_results(results)

    context = "\n\n---\n\n".join([
        f"Source: {r['metadata']['title']} (Level {r['metadata']['level']})"
        f"\n{r['text']}"
        for r in ranked
    ])

    role_instructions = {
        "student": "You are answering a student enrolled in Level 1 hypnotherapy training. Keep answers foundational and avoid advanced clinical concepts.",
        "practitioner": "You are answering a certified hypnotherapy practitioner. You can reference clinical techniques up to Level 3.",
        "trainer": "You are answering a senior trainer with full curriculum access. You can reference any level including advanced spiritual hypnotherapy."
    }

    system_prompt = f"""You are an EKAA hypnotherapy knowledge assistant.
{role_instructions[role]}
Answer using the context below. Use ALL relevant chunks across any level available to you.
If a foundational concept like hypnosis or suggestibility is asked, always include the core definition even if higher level content is also retrieved.
If context is insufficient, say so clearly.
Always mention which level the information comes from.

CONTEXT:
{context}"""

    # Build messages with history for generation
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    messages.append({"role": "user", "content": rewritten_query})

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "role": role,
        "answer": response.choices[0].message.content,
        "sources": list(set([r["metadata"]["title"] for r in ranked])),
        "ranked_chunks": ranked
    }
