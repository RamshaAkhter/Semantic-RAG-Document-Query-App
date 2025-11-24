import streamlit as st
import os
import tempfile
from agent import ingest, query_once  # Import your agent's functions here

st.set_page_config(page_title="Semantic RAG Document Query", layout="centered")
st.title("📄 Semantic RAG Document Query Application")
st.write(
    "Upload a document to ingest its content into the semantic vector store, then ask questions about it."
)

# Initialize session state variables
if "doc_ingested" not in st.session_state:
    st.session_state.doc_ingested = False
if "doc_id" not in st.session_state:
    st.session_state.doc_id = ""
if "doc_title" not in st.session_state:
    st.session_state.doc_title = ""
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Step 1: Upload and ingest document
st.header("Step 1: Insert Document")
uploaded_file = st.file_uploader(
    "Upload your document file (PDF, TXT, DOCX, etc.)", type=["pdf", "txt", "docx"]
)
doc_title_input = st.text_input("Document Title (optional)", placeholder="Optional document title")

if uploaded_file:
    if st.button("Ingest Document"):
        with st.spinner("Ingesting document..."):
            # Save uploaded file to temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Define doc_id and title for ingestion
            doc_id = os.path.basename(tmp_path)
            doc_title = doc_title_input.strip() or uploaded_file.name

            try:
                ingest(tmp_path, doc_id=doc_id, title=doc_title)
                st.success(f"Document '{doc_title}' ingested successfully!")
                st.session_state.doc_ingested = True
                st.session_state.doc_id = doc_id
                st.session_state.doc_title = doc_title
                st.session_state.query_history.clear()
            except Exception as e:
                st.error(f"Error during ingestion:\n{e}")

# Step 2: Query interface
st.header("Step 2: Ask a Query")
query = st.text_input("Enter your question here", placeholder="Type your question...")

if st.button("Submit Query"):
    if not st.session_state.doc_ingested:
        st.warning("Please ingest a document first before querying.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing query..."):
            try:
                answer, sources = query_once(query.strip())
                st.subheader("Answer")
                st.markdown(answer)
                st.session_state.query_history.append({"query": query, "answer": answer, "sources": sources})

                if sources:
                    st.subheader("Sources")
                    for src in sources:
                        title = src["payload"].get("title", "Unknown Document")
                        chunk_id = src["payload"].get("chunk_id", "N/A")
                        score = src.get("rerank_score", src.get("score", 0))
                        st.markdown(f"- **{title}** — {chunk_id} (score: {score:.3f})")
            except Exception as e:
                st.error(f"Error during query:\n{e}")

# Display query history (last 5 queries)
if st.session_state.query_history:
    st.header("Query History")
    for item in reversed(st.session_state.query_history[-5:]):
        with st.expander(f"Q: {item['query']}"):
            st.markdown(f"**Answer:** {item['answer']}")
            if item["sources"]:
                st.markdown("**Sources:**")
                for src in item["sources"]:
                    title = src["payload"].get("title", "Unknown Document")
                    chunk_id = src["payload"].get("chunk_id", "N/A")
                    score = src.get("rerank_score", src.get("score", 0))
                    st.markdown(f"- {title} — {chunk_id} (score: {score:.3f})")
