#!/usr/bin/env python3
"""
agent_semantic_recursive.py — Single-file RAG agent using semantic + recursive chunking

Features:
- PDF / URL / plain-text extraction
- Semantic chunking (sentence embeddings + similarity)
- Recursive splitting to enforce max chunk size
- OpenAI modern client (v1+) for embeddings & chat with sentence-transformers fallback
- Qdrant upsert & retrieval (handles multiple qdrant-client versions)
- Optional cross-encoder reranking
- CLI: ingest / query / interactive

Default LOCAL_DOC_PATH: /mnt/data/3.pdf
"""
#from __future__ import annotations
import os
import sys
import math
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Configuration (env-driven)
# ----------------------------
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
QDRANT_URL: str = os.getenv("QDRANT_URL", "https://5ea4df03-19ef-4e6c-b0bb-0cf246c0cd2f.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "semantic_agent_v1")

LOCAL_DOC_PATH = os.getenv("LOCAL_DOC_PATH", "data/raw/news.pdf")

EMBEDDING_MODEL_OPENAI = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "450"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "80"))
DENSE_TOP_N = int(os.getenv("DENSE_TOP_N", "50"))
FINAL_K = int(os.getenv("FINAL_K", "5"))

SEMANTIC_SIM_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.62"))
MIN_SENTENCES_PER_CHUNK = int(os.getenv("MIN_SENTENCES_PER_CHUNK", "1"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
QDRANT_UPSERT_BATCH = int(os.getenv("QDRANT_UPSERT_BATCH", "128"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("agent")

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def extract_text_from_pdf(path: str) -> str:
    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError("pdfplumber is required for PDF extraction. Install: pip install pdfplumber") from e
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)

def load_plain_text(path: str) -> str:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def naive_sentence_tokenizer(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sents: List[str] = []
    for ln in lines:
        parts = ln.split(". ")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.endswith("."):
                p = p + ("" if p.endswith(".") else ".")
            sents.append(p)
    return sents

def make_openai_client():
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("openai v1+ package is required. Install: pip install openai")
    return OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

def extract_embedding_from_resp_item(item):
    if isinstance(item, dict):
        return item.get("embedding")
    return getattr(item, "embedding", None)

def embed_sentences_for_chunking(sentences: List[str]) -> List[List[float]]:
    if OPENAI_API_KEY is None or OPENAI_API_KEY.strip() == "":
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for local sentence embedding. Install it.") from e
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
        return [list(vec) for vec in embs]
    else:
        client = make_openai_client()
        embeddings: List[List[float]] = []
        for i in range(0, len(sentences), EMBED_BATCH_SIZE):
            batch = sentences[i : i + EMBED_BATCH_SIZE]
            resp = client.embeddings.create(model=EMBEDDING_MODEL_OPENAI, input=batch)
            for item in resp.data:
                emb = extract_embedding_from_resp_item(item)
                if emb is None:
                    raise RuntimeError("Unexpected embeddings response shape from OpenAI client")
                embeddings.append(emb)
        return embeddings

def _cosine(a: List[float], b: List[float]) -> float:
    num = 0.0
    sa = 0.0
    sb = 0.0
    for x, y in zip(a, b):
        num += x * y
        sa += x * x
        sb += y * y
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return num / (math.sqrt(sa) * math.sqrt(sb))

def semantic_chunk_text(
    text: str,
    max_words_per_chunk: int = CHUNK_WORDS,
    sim_threshold: float = SEMANTIC_SIM_THRESHOLD,
    min_sentences: int = MIN_SENTENCES_PER_CHUNK
) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    sentences = naive_sentence_tokenizer(text)
    if not sentences:
        return []
    sent_word_counts = [len(s.split()) for s in sentences]
    sent_embs = embed_sentences_for_chunking(sentences)
    chunks: List[Dict[str, Any]] = []
    cur_indices: List[int] = []
    cur_words = 0

    def mean_embedding(indices: List[int]) -> List[float]:
        if not indices:
            return [0.0] * len(sent_embs[0])
        dim = len(sent_embs[0])
        mean = [0.0] * dim
        for idx in indices:
            vec = sent_embs[idx]
            for i in range(dim):
                mean[i] += vec[i]
        n = len(indices)
        for i in range(dim):
            mean[i] /= n
        return mean

    for i, sent in enumerate(sentences):
        words = sent_word_counts[i]
        if not cur_indices:
            cur_indices.append(i)
            cur_words = words
            continue
        if cur_words + words > max_words_per_chunk and len(cur_indices) >= min_sentences:
            chunk_text = " ".join(sentences[idx] for idx in cur_indices)
            chunks.append({"text": chunk_text})
            cur_indices = [i]
            cur_words = words
            continue
        cur_mean = mean_embedding(cur_indices)
        sim = _cosine(cur_mean, sent_embs[i])
        if sim >= sim_threshold or len(cur_indices) < min_sentences:
            cur_indices.append(i)
            cur_words += words
        else:
            chunk_text = " ".join(sentences[idx] for idx in cur_indices)
            chunks.append({"text": chunk_text})
            cur_indices = [i]
            cur_words = words
    if cur_indices:
        chunk_text = " ".join(sentences[idx] for idx in cur_indices)
        chunks.append({"text": chunk_text})

    return chunks

def recursive_split_chunks(chunks: List[Dict[str, Any]], max_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        if len(words) <= max_words:
            out.append({"text": text})
            continue
        sents = naive_sentence_tokenizer(text)
        cur: List[str] = []
        cur_words = 0
        for sent in sents:
            w = len(sent.split())
            if cur_words + w <= max_words or not cur:
                cur.append(sent)
                cur_words += w
            else:
                out.append({"text": " ".join(cur)})
                if overlap_words > 0:
                    all_words = " ".join(cur).split()
                    keep = []
                    if len(all_words) > overlap_words:
                        keep = [" ".join(all_words[-overlap_words:])]
                    else:
                        keep = [" ".join(all_words)]
                    cur = keep + [sent]
                    cur_words = sum(len(x.split()) for x in cur)
                else:
                    cur = [sent]
                    cur_words = w
        if cur:
            out.append({"text": " ".join(cur)})
    return out

def embed_texts_openai(texts: List[str], model: str = EMBEDDING_MODEL_OPENAI) -> List[List[float]]:
    client = make_openai_client()
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            emb = extract_embedding_from_resp_item(item)
            if emb is None:
                raise RuntimeError("Unexpected OpenAI embedding response shape")
            embeddings.append(emb)
    return embeddings

def embed_texts_st(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("sentence-transformers is required for local embedding fallback. Install it.")
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return arr.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    if OPENAI_API_KEY:
        return embed_texts_openai(texts)
    logger.info("OPENAI_API_KEY not set — using local sentence-transformers for embeddings.")
    return embed_texts_st(texts)

def make_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except Exception:
        raise RuntimeError("qdrant-client is required. Install: pip install qdrant-client")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)

def ensure_collection(qdrant, dim: int):
    from qdrant_client.http.models import VectorParams, Distance
    try:
        qdrant.get_collection(COLLECTION_NAME)
    except Exception:
        logger.info("Creating Qdrant collection %s with dim=%d", COLLECTION_NAME, dim)
        qdrant.recreate_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def batched_upsert(qdrant, points: List[Dict[str, Any]], batch_size: int = QDRANT_UPSERT_BATCH):
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + batch_size])
        logger.debug("Upserted batch %d..%d", i, i + len(points[i : i + batch_size]))

def upsert_chunks_to_qdrant(doc_id: str, title: str, chunks: List[Dict[str, Any]]):
    qdrant = make_qdrant_client()
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks...", len(texts))
    vectors = embed_texts(texts)
    if not vectors:
        raise RuntimeError("No vectors produced.")
    ensure_collection(qdrant, len(vectors[0]))
    points = []
    for idx, (vec, chunk) in enumerate(zip(vectors, chunks)):
      chunk_id = idx  # Use integer ID for Qdrant point ID
      payload = {"doc_id": doc_id, "title": title, "chunk_id": f"{doc_id}__{idx}", "excerpt": chunk["text"][:400]}
      points.append({"id": chunk_id, "vector": vec, "payload": payload})

    logger.info("Upserting %d vectors to Qdrant collection %s", len(points), COLLECTION_NAME)
    batched_upsert(qdrant, points)
    logger.info("Upsert complete.")

def dense_search(query: str, top_n: int = DENSE_TOP_N) -> List[Dict[str, Any]]:
    qdrant = make_qdrant_client()
    q_vec = embed_texts([query])[0]
    # Try to ensure collection exists (robust against 404 error)
    try:
        qdrant.get_collection(COLLECTION_NAME)
    except Exception:
        # guessing dim from embedding
        dim = len(q_vec)
        ensure_collection(qdrant, dim)
    hits = None
    if hasattr(qdrant, "search"):
        try:
            hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=top_n)
        except TypeError:
            hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_vec, top=top_n)
    elif hasattr(qdrant, "search_points"):
        try:
            resp = qdrant.search_points(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=top_n)
        except TypeError:
            resp = qdrant.search_points(collection_name=COLLECTION_NAME, query_vector=q_vec, top=top_n)
        hits = getattr(resp, "result", resp)
    elif hasattr(qdrant, "points") and hasattr(qdrant.points, "search"):
        resp = qdrant.points.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=top_n)
        hits = getattr(resp, "result", resp)
    else:
        raise RuntimeError("qdrant-client does not expose a known search method (search, search_points, points.search).")

    results: List[Dict[str, Any]] = []
    for h in hits or []:
        try:
            hid = getattr(h, "id", None) or (h.get("id") if isinstance(h, dict) else None)
        except Exception:
            hid = None
        try:
            score = getattr(h, "score", None) or (h.get("score") if isinstance(h, dict) else None)
        except Exception:
            score = None
        payload = None
        try:
            payload = getattr(h, "payload", None) or (h.get("payload") if isinstance(h, dict) else None)
        except Exception:
            payload = None
        if payload is None:
            try:
                payload = h.point.payload
            except Exception:
                pass
        results.append({"id": hid, "score": score, "payload": payload})
    return results

def try_load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None

def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = FINAL_K) -> List[Dict[str, Any]]:
    reranker = try_load_reranker()
    if reranker is None:
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates[:top_k]
    texts = [c["payload"].get("excerpt", "") for c in candidates]
    pairs = [[query, t] for t in texts]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]

SYSTEM_PROMPT = (
    "You are an assistant that MUST answer using ONLY the provided document chunks.\n"
    "If the answer cannot be found in the chunks, reply: \"I don't know.\"\n"
    "Cite chunks inline like: [Title — chunk_id].\n"
)

PROMPT_TEMPLATE = "Context chunks:\n{chunks_text}\n\nQuestion:\n{question}\n\nAnswer concisely. Include inline citations for factual claims.\n"

def assemble_prompt(chunks: List[Dict[str, Any]], question: str) -> str:
    parts = []
    for c in chunks:
        title = c["payload"].get("title", "")
        cid = c["payload"].get("chunk_id", c.get("id"))
        excerpt = c["payload"].get("excerpt", "")[:1200]
        parts.append(f"---\n[{title} — {cid}]\n{excerpt}\n")
    chunks_text = "\n".join(parts)
    return SYSTEM_PROMPT + PROMPT_TEMPLATE.format(chunks_text=chunks_text, question=question)

def extract_chat_text_from_choice(choice_item):
    if isinstance(choice_item, dict):
        msg = choice_item.get("message") or choice_item.get("message", {})
        if isinstance(msg, dict):
            return msg.get("content") or None
        return None
    try:
        m = getattr(choice_item, "message", None)
        if m is not None:
            c = getattr(m, "content", None)
            if isinstance(c, str):
                return c
            try:
                return c.get("text")
            except Exception:
                return None
    except Exception:
        pass
    try:
        return str(choice_item)
    except Exception:
        return ""

def call_openai_chat(prompt: str, model: str = CHAT_MODEL, temperature: float = 0.0, max_tokens: int = 800) -> str:
    client = make_openai_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    choice = None
    try:
        choice = resp.choices[0]
    except Exception:
        choice = resp.get("choices", [None])[0] if isinstance(resp, dict) else None
    if choice is None:
        return str(resp)
    content = extract_chat_text_from_choice(choice)
    return content or str(resp)

def ingest(path: str, doc_id: str, title: str):
    path = os.path.expanduser(path)
    logger.info("Starting ingest: %s (doc_id=%s)", path, doc_id)
    if path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(path)
    else:
        text = load_plain_text(path)
    if not text or not text.strip():
        raise RuntimeError("Document is empty or could not be loaded.")

    logger.info("Performing semantic chunking...")
    sem_chunks = semantic_chunk_text(text, max_words_per_chunk=CHUNK_WORDS, sim_threshold=SEMANTIC_SIM_THRESHOLD, min_sentences=MIN_SENTENCES_PER_CHUNK)
    logger.info("Semantic chunker produced %d chunks", len(sem_chunks))

    logger.info("Applying recursive splitter to enforce max chunk words...")
    final_chunks = recursive_split_chunks(sem_chunks, max_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS)
    logger.info("After recursive splitting: %d chunks", len(final_chunks))

    upsert_chunks_to_qdrant(doc_id, title, final_chunks)
    logger.info("Ingest finished for doc_id=%s", doc_id)

def query_once(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    logger.info("Querying: %s", question)
    dense = dense_search(question, top_n=DENSE_TOP_N)
    if not dense:
        return "No results found in the vector store.", []
    top = rerank(question, dense, top_k=FINAL_K)
    prompt = assemble_prompt(top, question)
    if OPENAI_API_KEY:
        answer = call_openai_chat(prompt)
    else:
        answer = "OPENAI_API_KEY not set: cannot call LLM."
    return answer, top

def interactive_loop():
    logger.info("Entering interactive mode. Type 'exit' to quit.")
    try:
        while True:
            q = input("\nQuestion> ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break
            ans, top = query_once(q)
            safe_print("\n=== ANSWER ===\n")
            safe_print(ans)
            safe_print("\n=== SOURCES ===")
            for c in top:
                safe_print(f"- {c['payload'].get('title')} — {c['payload'].get('chunk_id')} (score {c.get('rerank_score', c.get('score', 0)):.3f})")
    except KeyboardInterrupt:
        logger.info("Interactive terminated by user.")

def build_cli():
    p = argparse.ArgumentParser(prog="agent_semantic_recursive.py", description="RAG agent with semantic + recursive chunking")
    sub = p.add_subparsers(dest="cmd", required=True)
    ig = sub.add_parser("ingest", help="Ingest a document into Qdrant")
    ig.add_argument("--path", required=True, help="Path or URL to document (PDF / url / text)")
    ig.add_argument("--doc-id", default="doc_local", help="Document id to store in Qdrant")
    ig.add_argument("--title", default="Local Document", help="Document title for metadata")
    q = sub.add_parser("query", help="Run a single query")
    q.add_argument("--q", required=True, help="Question text")
    sub.add_parser("interactive", help="Run interactive query loop")
    return p

def main(argv: Optional[List[str]] = None):
    parser = build_cli()
    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        ingest(args.path, args.doc_id, args.title)
    elif args.cmd == "query":
        answer, top = query_once(args.q)
        safe_print(answer)
        if top:
            safe_print("\nSOURCES:")
            for c in top:
                safe_print(f"- {c['payload'].get('title')} — {c['payload'].get('chunk_id')} (score {c.get('rerank_score', c.get('score', 0)):.3f})")
    elif args.cmd == "interactive":
        interactive_loop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
