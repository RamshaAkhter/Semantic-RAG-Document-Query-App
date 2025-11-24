# 📄 Semantic RAG Document Query App

A compact yet mighty Retrieval-Augmented Generation system built with **Streamlit**, **OpenAI**, and **Qdrant**.
Upload a document, watch it get sliced into meaning-aware chunks, and query it with an LLM that responds using strictly grounded context—complete with precise inline citations.

## 🌿 Overview

This app blends semantic understanding, vector search, and modern LLMs into a smooth question-answering pipeline.

| Feature                                        | Description                                                     | Purpose                                            |
| ---------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------- |
| Document ingestion, semantic analysis, and Q&A | Upload PDFs or text files, process them, and ask questions      | Provide accurate, source-grounded answers          |
| Engine                                         | `gpt-4o-mini` + `text-embedding-3-small`                        | Fast, cost-efficient LLM + high-quality embeddings |
| Vector DB                                      | **Qdrant** (cloud or local)                                     | Stores semantic chunks for retrieval               |
| UI                                             | **Streamlit**                                                   | Minimal, responsive chat interface                 |
| Key Tech                                       | Semantic Chunking, Recursive Splitting, Cross-Encoder Reranking | Preserve context and boost retrieval quality       |

## 🧩 How It Works

Ingestion
Users upload a PDF or text file via the Streamlit UI (`app.py`). The text is extracted and passed to the RAG engine.

Semantic Chunking
Inside `agent.py`, the document isn’t chopped arbitrarily. Sentence embeddings glue together related sentences so each chunk represents a meaningful thought rather than a random text slice.

Storage
Each chunk is embedded using OpenAI (or a local model when needed).
Chunks are stored in a Qdrant collection as dense vectors.

Retrieval & Reranking
A user query triggers a similarity search in Qdrant.
A Cross-Encoder can optionally re-score the top candidates for maximum relevance.

Generation
The LLM answers strictly from the retrieved context and includes clear inline citations such as:
`[Document Title — chunk_12]`.

## 🗂️ Project Structure

```
semantic-rag-agent/
├── app.py                  # Streamlit UI
├── agent.py                # Core RAG logic: chunking, embedding, retrieval, LLM
├── data/                   # Optional local docs
├── requirements.txt        # Dependencies
└── .env                    # Environment variables (git-ignored)
```

## ⚙️ Setup

Clone the repository:

```
git clone https://github.com/<your-username>/semantic-rag-agent.git
cd semantic-rag-agent
```

Install dependencies:

```
pip install -r requirements.txt
```

(Requires: streamlit, openai, qdrant-client, sentence-transformers, pdfplumber, python-dotenv.)

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://...        # or local
QDRANT_API_KEY=...            # optional if local

QDRANT_COLLECTION=semantic_agent_v1
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

Run the app:

```
streamlit run app.py
```

Now upload a document and start chatting with it.

## 🧠 Model & Logic Details

Embeddings
Primary: `text-embedding-3-small`
Fallback: `all-MiniLM-L6-v2` (local SentenceTransformers)

Chunking Strategy
Semantic grouping based on cosine similarity (threshold: 0.62).
Recursive splitting ensures upper limits (~450 words) while keeping ~80-word overlaps for coherence.

Reranking
`cross-encoder/ms-marco-MiniLM-L-6-v2` reorders retrieved chunks to maximize answer quality.

Capabilities
• Clean conversational UI with citations and relevance scores
• High-quality PDF extraction via pdfplumber
• Command-line usage:

```
python agent.py ingest --path data/my_doc.pdf
python agent.py query --q "What is the summary?"
```

## 💡 Future Improvements

[ ] Multi-document ingestion
[ ] Persistent chat history (DB-backed)
[ ] Highlight citations directly in the UI
[ ] OCR support for scanned PDFs

## 👨‍💻 Author

**Ramsha Akhter**

🎓 M.S. Data Science | University of Texax Arlington


## 📜 License

Released under the **MIT License**.