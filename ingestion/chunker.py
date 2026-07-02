import os
import boto3
from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, S3_BUCKET, REGION


def download_docs_from_s3(local_dir="data/raw_docs", s3_prefix="raw_docs"):
    """Download raw docs and templates from S3 to local directory"""
    s3 = boto3.client("s3", region_name=REGION)

    os.makedirs(local_dir, exist_ok=True)

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)

    if "Contents" not in response:
        print(f"No files found in s3://{S3_BUCKET}/{s3_prefix}")
        return

    for obj in response["Contents"]:
        key = obj["Key"]
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)

        print(f"Downloading {key} → {local_path}")
        s3.download_file(S3_BUCKET, key, local_path)


def load_documents(local_dir="data/raw_docs"):
    """Load all documents from local directory"""
    documents = []

    for filename in os.listdir(local_dir):
        filepath = os.path.join(local_dir, filename)

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".html"):
                loader = BSHTMLLoader(filepath)
            elif filename.endswith((".yaml", ".yml", ".yam", ".json")):
                loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
            elif filename.endswith(".txt") or filename.endswith(".md"):
                loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["file_type"] = filename.split(".")[-1]

            documents.extend(docs)
            print(f"Loaded: {filename} ({len(docs)} pages/sections)")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def load_and_chunk(local_dir="data/raw_docs", download_from_s3=True):
    """Main function — download, load and chunk all documents"""

    # Step 1 — download from S3
    if download_from_s3:
        print("Downloading docs from S3...")
        download_docs_from_s3(
            local_dir=local_dir,
            s3_prefix="raw_docs"
        )
        download_docs_from_s3(
            local_dir="data/sample_templates",
            s3_prefix="sample_templates"
        )

    # Step 2 — load documents
    print("\nLoading documents...")
    documents = load_documents(local_dir)

    # Also load sample templates
    template_dir = "data/sample_templates"
    if os.path.exists(template_dir):
        print("\nLoading sample templates...")
        template_docs = load_documents(template_dir)
        documents.extend(template_docs)

    # Step 3 — chunk documents
    print("\nChunking documents...")
    chunks = chunk_documents(documents)

    return chunks