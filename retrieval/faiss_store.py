import os
import boto3
from langchain_community.vectorstores import FAISS
from config.config import S3_BUCKET, REGION, FAISS_INDEX_PATH, S3_PROFILE


def _get_s3_client():
    """Always use the dedicated S3 account profile, never the default
    credential chain — the default may resolve to a different account's
    borrowed key (e.g. one only granted for Bedrock access)."""
    session = boto3.Session(profile_name=S3_PROFILE)
    return session.client("s3", region_name=REGION)


def build_faiss_index(chunks, embeddings):
    """Build FAISS index from document chunks with throttle handling"""
    try:
        import time
        from langchain_community.vectorstores import FAISS

        print(f"Building FAISS index from {len(chunks)} chunks...")

        # Process in small batches to avoid throttling
        batch_size = 5
        vectorstore = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{-(-len(chunks)//batch_size)}...")

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

            time.sleep(1)  # wait 1 second between batches

        print("FAISS index built successfully")
        return vectorstore
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None


def save_faiss_index(vectorstore, index_path=FAISS_INDEX_PATH):
    """Save FAISS index locally"""
    try:
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        print(f"FAISS index saved locally to {index_path}/")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
        raise


def upload_faiss_to_s3(index_path=FAISS_INDEX_PATH, bucket=S3_BUCKET):
    """Upload FAISS index files to S3"""
    try:
        s3 = _get_s3_client()

        faiss_file = os.path.join(index_path, "index.faiss")
        pkl_file = os.path.join(index_path, "index.pkl")

        s3.upload_file(faiss_file, bucket, "faiss_index/index.faiss")
        print(f"Uploaded index.faiss to s3://{bucket}/faiss_index/")

        s3.upload_file(pkl_file, bucket, "faiss_index/index.pkl")
        print(f"Uploaded index.pkl to s3://{bucket}/faiss_index/")

    except Exception as e:
        print(f"Error uploading FAISS index to S3: {e}")
        raise


def download_faiss_from_s3(index_path="/tmp/faiss_index", bucket=S3_BUCKET):
    """Download FAISS index files from S3"""
    try:
        s3 = _get_s3_client()

        os.makedirs(index_path, exist_ok=True)

        s3.download_file(bucket, "faiss_index/index.faiss",
                         os.path.join(index_path, "index.faiss"))
        print(f"Downloaded index.faiss from s3://{bucket}/faiss_index/")

        s3.download_file(bucket, "faiss_index/index.pkl",
                         os.path.join(index_path, "index.pkl"))
        print(f"Downloaded index.pkl from s3://{bucket}/faiss_index/")

    except Exception as e:
        print(f"Error downloading FAISS index from S3: {e}")
        raise


def load_faiss_index(embeddings, index_path="/tmp/faiss_index"):
    """Load FAISS index from local path into memory"""
    try:
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"FAISS index loaded from {index_path}")
        return vectorstore
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise


def load_faiss_from_s3(embeddings, bucket=S3_BUCKET):
    """Download FAISS index from S3 and load into memory"""
    index_path = "/tmp/faiss_index"
    download_faiss_from_s3(index_path=index_path, bucket=bucket)
    return load_faiss_index(embeddings, index_path=index_path)