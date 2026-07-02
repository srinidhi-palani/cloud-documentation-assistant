import time
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("Embedding model initialized: all-MiniLM-L6-v2")
    return embeddings


def test_embedding(embeddings, retries=1, wait=1):
    try:
        vector = embeddings.embed_query("What is Amazon ECS?")
        print(f"Embedding test passed — vector dimensions: {len(vector)}")
        return True
    except Exception as e:
        print(f"Embedding test failed: {e}")
        return False


def embed_query(query, embeddings):
    try:
        vector = embeddings.embed_query(query)
        return vector
    except Exception as e:
        print(f"Error embedding query: {e}")
        raise


def embed_documents(chunks, embeddings):
    try:
        all_vectors = []
        batch_size = 10
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.page_content for chunk in batch]
            vectors = embeddings.embed_documents(texts)
            all_vectors.extend(vectors)
            print(f"Embedded {min(i + batch_size, total)}/{total} chunks...")

        print(f"Embedded {len(all_vectors)} chunks successfully")
        return all_vectors
    except Exception as e:
        print(f"Error embedding documents: {e}")
        raise