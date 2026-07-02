from langchain_community.vectorstores import FAISS
from retrieval.faiss_store import load_faiss_from_s3
from config.config import TOP_K, S3_BUCKET


# Global variable to cache loaded index in memory
_vectorstore = None


def get_vectorstore(embeddings):
    """Load FAISS index from S3 — reuse if already loaded in memory"""
    global _vectorstore

    if _vectorstore is None:
        print("Loading FAISS index from S3...")
        _vectorstore = load_faiss_from_s3(embeddings, bucket=S3_BUCKET)
        print("FAISS index loaded into memory")
    else:
        print("Reusing FAISS index from memory")

    return _vectorstore


def retrieve_chunks(query, embeddings, k=TOP_K):
    """Search FAISS index and return top K relevant chunks"""
    try:
        vectorstore = get_vectorstore(embeddings)

        print(f"Searching for: {query}")
        results = vectorstore.similarity_search(query, k=k)

        print(f"Retrieved {len(results)} chunks")
        return results

    except Exception as e:
        print(f"Error during retrieval: {e}")
        raise


def retrieve_with_scores(query, embeddings, k=TOP_K):
    """Search FAISS index and return chunks with similarity scores"""
    try:
        vectorstore = get_vectorstore(embeddings)

        results = vectorstore.similarity_search_with_score(query, k=k)

        print(f"Retrieved {len(results)} chunks with scores")
        for doc, score in results:
            print(f"Score: {score:.4f} — Source: {doc.metadata.get('source_file', 'unknown')}")

        return results

    except Exception as e:
        print(f"Error during retrieval with scores: {e}")
        raise


def format_context(chunks):
    """Format retrieved chunks into a single context string for the LLM"""
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_file", "unknown")
        content = chunk.page_content
        context_parts.append(f"[Chunk {i+1} — Source: {source}]\n{content}")

    return "\n\n".join(context_parts)


def retrieve_and_format(query, embeddings, k=TOP_K):
    """Retrieve chunks and return both raw chunks and formatted context"""
    chunks = retrieve_chunks(query, embeddings, k=k)
    context = format_context(chunks)
    return chunks, context