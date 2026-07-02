import os
import boto3
from ingestion.chunker import load_and_chunk
from ingestion.embedder import get_embeddings, test_embedding
from retrieval.faiss_store import build_faiss_index, save_faiss_index, upload_faiss_to_s3
from config.config import S3_BUCKET, FAISS_INDEX_PATH, REGION

def run_ingestion():
    """Main ingestion pipeline — chunks, embeds, builds FAISS index, uploads to S3"""
    
    print("=" * 50)
    print("Starting ingestion pipeline...")
    print("=" * 50)

    # Step 1 — initialize embedding model
    print("\nStep 1: Initializing embedding model...")
    embeddings = get_embeddings()
    
    # Step 2 — test embedding model connection
    print("\nStep 2: Testing embedding model...")
    test_passed = test_embedding(embeddings)
    if not test_passed:
        print("Embedding test failed — stopping ingestion.")
        return
    
    # Step 3 — download docs from S3, load and chunk
    print("\nStep 3: Loading and chunking documents...")
    chunks = load_and_chunk(
        local_dir="data/raw_docs",
        download_from_s3=True
    )
    
    if not chunks:
        print("No chunks created — stopping ingestion.")
        return
    
    print(f"Total chunks ready for embedding: {len(chunks)}")
    
    # Step 4 — build FAISS index from chunks
    print("\nStep 4: Building FAISS index...")
    vectorstore = build_faiss_index(chunks, embeddings)
    
    if not vectorstore:
        print("FAISS index build failed — stopping ingestion.")
        return
    
    # Step 5 — save FAISS index locally
    print("\nStep 5: Saving FAISS index locally...")
    save_faiss_index(vectorstore, FAISS_INDEX_PATH)
    
    # Step 6 — upload FAISS index to S3
    print("\nStep 6: Uploading FAISS index to S3...")
    upload_faiss_to_s3(FAISS_INDEX_PATH, S3_BUCKET)
    
    print("\n" + "=" * 50)
    print("Ingestion pipeline completed successfully!")
    print(f"FAISS index uploaded to s3://{S3_BUCKET}/faiss_index/")
    print("=" * 50)

if __name__ == "__main__":
    run_ingestion()