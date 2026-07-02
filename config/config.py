import os
from dotenv import load_dotenv

load_dotenv()

# AWS settings
REGION = "eu-north-1"
S3_BUCKET = "srinidhi-palani"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# OpenRouter settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
LLM_MODEL = "openrouter/free"

# FAISS settings
FAISS_INDEX_PATH = "faiss_index"
FAISS_S3_PREFIX = "faiss_index"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K = 5