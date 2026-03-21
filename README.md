# Enterprise-Style-Knowledge-Assistant-with-Role-Based-Answers
An enterprise-style knowledge assistant with role-based answers, retrieval using metadata filters, and ranking logic for document freshness.

An enterprise RAG system where the same question gets different answers depending on who is asking. A student, a practitioner, and a trainer asking "what is past life regression?" each receives answers drawn from different documents because each role has different access permissions.
- Built with Python, ChromaDB, LangChain, OpenAI, and Streamlit. Deployed at a permanent URL accessible without any local setup.

# What This Project Demonstrates
Most RAG demos retrieve the same documents for every user and generate the same answer. That works for a personal chatbot. It does not work for any enterprise system where:

A new employee should not see executive compensation documents
A Level 1 student should not access advanced clinical techniques meant for certified practitioners
An intern should not retrieve HR policy documents intended for managers

This project builds the layer that makes RAG enterprise-safe: role-based access control at the retrieval level, source ranking by authority and freshness, and a formal evaluation script that verifies access boundaries are holding.

# Live Demo
URL: https://enterprise-style-knowledge-assistant-with-role-based-answers-j.streamlit.app/
Select a role from the sidebar. Ask any question about hypnotherapy, breathing techniques, and human chakras. Switch roles and ask the same question — you will get a different answer drawn from different sources.

# Analytics
URL: https://us.posthog.com/project/350232/dashboard/1381621

# Project Structure
Enterprise-Style-Knowledge-Assistant-with-Role-Based-Answers/
├── app.py                  ← Streamlit UI: chat interface, role selection, session state
├── rag_pipeline.py         ← RAG logic: index builder, query rewriter, retrieval, ranking, generation
├── requirements.txt        ← Dependencies
└── documents/
    ├── ekaa_level1.pdf     ← Accessible by student, practitioner, trainer
    ├── ekaa_level2.pdf     ← Accessible by practitioner, trainer
    ├── ekaa_level3.pdf     ← Accessible by practitioner, trainer
    └── ekaa_level5.pdf     ← Accessible by trainer only

# Metadata Schema
Every document is tagged with five metadata fields at indexing time. These fields drive all filtering and ranking decisions downstream.
Document, Level, Role,  AccessAuthority, Department, Date.

# Why each field exists:
level: the access control filter. ChromaDB uses {"level": {"$lte": max_level}} to enforce boundaries
Authority: source ranking signal. Official curriculum outranks drafts on the same topic
Department: enables department-scoped queries. Training vs clinical questions retrieve different document sets
Date: freshness ranking signal, newer documents score higher when the authority is equal
Roles: human-readable label for documentation and debugging

# Role access rules:
student → Level 1 only
practitioner → Levels 1, 2, 3
trainer → All levels, including Level 5

# Architecture 
User Query + User Role
        ↓
Per-role conversation history (st.session_state)
        ↓
Query Rewriter
reads: query + last 6 messages
outputs: self-contained rewritten query
e.g. "summarise this" → "summarise the explanation of heart chakra"
        ↓
ChromaDB metadata filter
where={"level": {"$lte": max_level}}
        ↓
Top 8 chunks returned
        ↓
Source Ranking
0.7 × semantic score + 0.2 × authority score + 0.1 × freshness score
        ↓
Top ranked chunks → context
        ↓
Role-aware system prompt + context + conversation history → GPT-4o-mini
        ↓
Role-appropriate answer + sources cited

# Source Ranking: Why It Exists
ChromaDB returns chunks sorted by semantic similarity, which is how closely the chunk text matches the query. This is necessary but not sufficient. Three failure modes occur without source ranking:
##Failure 1: A draft overrides an official policy. If a knowledge base contains both an approved HR policy and an unapproved draft covering the same topic, pure semantic search might surface the draft. Authority ranking ensures official sources always outrank drafts on the same topic.
##Failure 2: Outdated information surfaces over current information. If a company updates its onboarding process every year, both the 2020 and 2024 versions are semantically similar to "what is the onboarding process." Freshness ranking ensures the 2024 version wins.
##Failure 3: Semantic similarity rewards breadth, not accuracy. A long document that mentions a topic once in passing can outscore a short, focused document that directly answers the question. Authority and freshness inject domain judgment into what is otherwise a pure mathematical similarity score.

# Ranking formula
final_score = (
    0.7 * semantic_score +    # relevance is the primary signal
    0.2 * authority_score +   # official sources outrank drafts
    0.1 * freshness_score     # newer sources outrank older ones
)

# Query Rewriting: Why It Exists
Without query rewriting, follow-up questions break retrieval entirely.
Example of the failure: User asks "what is the heart chakra?" and gets a detailed answer. User then asks, "Summarise this in 35 words." The system embeds "summarise this in 35 words" and retrieves chunks about marriage therapy, because that query has no connection to the heart chakra in the vector space.
Why passing history to the LLM doesn't fix this: Conversation history is injected after retrieval. The retrieval step runs on the raw query; it has no idea what "this" refers to.
The fix is principle-based query rewriting:
Before retrieval, an LLM rewrites the query into a self-contained question using the last 6 messages of conversation history. The instruction is: "rewrite so it can be understood without any conversation history."
This handles every vague formulation automatically — "this", "that", "tell me more", "elaborate", "what about that technique" — without maintaining a word list. A principle generalises. A word list breaks on edge cases.
Rewritten query is embedded, not the original. ChromaDB retrieves the right chunks. The LLM generates the right answer.

# Evaluation Script
The evaluation script tests whether access control boundaries are holding. It runs 8 test cases — 5 positive (role should retrieve this source) and 3 negative (role should NOT retrieve this source).
The negative tests are the ones that matter most:
{
    "query": "Explain past life regression therapy",
    "role": "practitioner",
    "should_not_contain_source": "EKAA Level 5 Clinical Manual",
    "description": "Practitioner must NOT access Level 5 content"
}

# Two failure types and their severity:
- Student retrieves Level 5 content🔴 Critical — system is broken
- Practitioner doesn't get Level 2 in top 5🟡 Quality issue — query needs refinement

These are completely different problems. A system that passes all positive tests but fails negative tests appears to work, but is leaking privileged content.
Final eval result: 8/8 passing
All three "SHOULD NOT retrieve" tests pass; no role can access content above their permitted level.

# Tech Stack Decisions
- ChromaDB over FAISS
FAISS has no native metadata filtering. To filter by role with FAISS, you retrieve everything and filter in Python afterwards, which means restricted documents are still retrieved, just discarded after the fact. ChromaDB applies the where filter before the semantic search runs. Restricted documents are never retrieved at all. This is the correct approach for access control.
- embed_documents() over embed_query()
embed_query() makes one API call per chunk. embed_documents() batches the entire document into the minimum number of API calls. With 546 chunks, this is the difference between ~546 API calls and ~4. At production scale (500,000 documents), batching is not optional.
- Index rebuilt at startup, not persisted
ChromaDB runs in-memory. The index is rebuilt from the PDFs in the documents/ folder every time a new Streamlit session starts. This takes ~30 seconds. The tradeoff: no external vector database needed, the repo is fully self-contained, and anyone who clones the repo gets a working system. A pinecone or a persistent ChromaDB instance would eliminate the startup delay in production.
- Per-role conversation history
st.session_state stores a separate message list for each role. Switching from student to trainer and back preserves the student conversation exactly where it was left. This mirrors how production multi-user systems store history keyed by user ID rather than by session.

# What This Assumes Away: The Real Enterprise Problem
This project demonstrates the retrieval and ranking mechanics of enterprise RAG. In a real company, the hardest work happens before any of this code runs.

## Problem 1: Document governance
A company with 40,000 SharePoint documents has no consistent naming, no tagging, no access taxonomy, and no single owner. Before metadata filters can work, someone has to answer:

Who classifies which documents belong to which category?
What happens when a document crosses departments, an engineering spec that also contains budget information?
Who is the source of truth when two documents contradict each other?

Solutions in production: manual tagging by document owners (slow, doesn't scale), rule-based inference from folder structure (fast but brittle), or LLM-assisted classification at ingestion time with a human review pipeline (the emerging approach).

## Problem 2: Identity system integration
Our roles are hardcoded in a dropdown. A real company has roles defined in Active Directory or Okta. The RAG system has to query the identity system at runtime to determine what level the logged-in user has, and that mapping changes dynamically as people get promoted, change teams, or leave.
This means the RAG system does not have access control. It inherits it from wherever the company already manages permissions. Building that integration is a 3-month engineering project before a single document is indexed.

## Problem 3: Evaluation at scale
Our eval script has 8 hand-written test cases covering 3 roles and 4 documents. A production system with 500 document types and 20 roles needs: Hundreds of test cases covering every access boundary, Automated regression testing that runs before every deployment, Monitoring for retrieval quality drift over time as documents are added and updated, Separate evaluation of access control correctness vs retrieval quality; these are different problems with different owners.

# Key Learnings
- The hardest part of enterprise RAG is not the RAG. The retrieval logic, metadata filters, and ranking took hours to build. The metadata schema, deciding which documents belong to which roles, what access taxonomy to use, is months of work in a real company.
- Ranking cannot fix dirty retrieval. Adding a ranking layer reorders what is retrieved. It cannot remove noise that should not have been stored. Boilerplate header chunks that appeared on every PDF page polluted every query. The fix was a content filter at the indexing stage, not a better ranking formula.
- Retrieval does not see conversation history. Passing history to the LLM helps generation, but does nothing for retrieval. "Summarise this" retrieves wrong chunks regardless of how much history the LLM sees. The fix is query rewriting before retrieval, not history injection after retrieval.
- Evaluation test quality matters as much as system quality. A test that asks a vague question and expects a specific source is testing whether semantic search is psychic. Good eval tests use queries specific enough to have a deterministic expected answer.
- Use a principle, not a word list. The query rewriter doesn't maintain a list of vague words. It uses the principle: "Can this be understood without conversation history?" This handles every edge case automatically.

# Built With
Python - Core language 
ChromaDB Vector - database with native metadata filtering
LangChainPDF - loading and text chunking
OpenAI text-embedding-3-small - Document and query embeddings
GPT-4o-mini - Query rewriting and answer generation
StreamlitChat - UI and deployment
GitHub - Version control and source for PDF documents

