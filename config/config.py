import os
from dotenv import load_dotenv
load_dotenv()
# AWS settings
REGION = "eu-north-1"
S3_BUCKET = "srinidhi-palani"
S3_PROFILE = "s3-account"
LLM_BACKEND = os.getenv("LLM_BACKEND", "nova")  # "nova" or "claude"
if LLM_BACKEND == "claude":
    BEDROCK_PROFILE = "office-bedrock"
    BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
else:
    BEDROCK_PROFILE = "bedrock-account"
    BEDROCK_MODEL_ID = "amazon.nova-lite-v1:0"
# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AWS_REGION = "us-east-1"
# FAISS settings
FAISS_INDEX_PATH = "faiss_index"
FAISS_S3_PREFIX = "faiss_index"
# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Retrieval settings
TOP_K = 5