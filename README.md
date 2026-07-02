# Cloud Documentation Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about AWS infrastructure and CloudFormation templates, with citations back to source documentation. Built as a cost-efficient, fully local proof of concept.

## What it does

- Ingests raw infrastructure documentation and CloudFormation/Terraform templates
- Answers natural-language questions about your infrastructure, with source citations
- Generates and reviews CloudFormation templates based on ingested examples and rules
- Runs entirely locally via a Streamlit UI — no hosted vector database required

## Tech stack

| Component        | Choice                                  |
|-------------------|------------------------------------------|
| Embeddings        | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store       | FAISS (local, in-process)                |
| Orchestration      | LangChain                                |
| LLM                | Gemma (via OpenRouter)                   |
| UI                 | Streamlit                                |
| Target infra        | AWS (`eu-north-1`)                       |

FAISS was chosen over a hosted option like OpenSearch to keep the stack serverless-leaning and cost-efficient for a POC.

## Project structure

```
app/                    Streamlit application entry point
config/                 Central configuration (region, model names, chunking, paths)
data/
  raw_docs/              Source infrastructure documentation
  sample_templates/       Example CloudFormation / Terraform templates
generation/             LLM chains for Q&A and template generation
ingestion/              Chunking, embedding, and ingestion pipeline
lambda/                 AWS Lambda handlers (ingestion, orchestration)
retrieval/              FAISS store and retriever logic
requirements.txt        Python dependencies
```

## Setup

### Prerequisites

- Python 3.11
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
git clone https://github.com/srinidhi-palani/cloud-documentation-assistant.git
cd cloud-documentation-assistant

python -m venv venv311
venv311\Scripts\activate        # Windows
# source venv311/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

Other settings (region, S3 bucket, chunk size, top-k retrieval, etc.) are defined in `config/config.py`.

### Run

```bash
streamlit run app/streamlit_app.py
```

## How it works

1. **Ingestion** — Raw docs and templates in `data/` are chunked (`ingestion/chunker.py`) and embedded (`ingestion/embedder.py`) into a local FAISS index.
2. **Retrieval** — On each query, the most relevant chunks are retrieved from FAISS (`retrieval/retriever.py`) based on similarity to the question.
3. **Generation** — Retrieved context is passed to the LLM via LangChain (`generation/qa_chain.py`) to produce a cited answer, or to `generation/template_chain.py` to generate/review a CloudFormation template.
4. **Interface** — All of the above is exposed through a Streamlit chat UI.

## Status

Proof of concept — runs locally. Lambda handlers in `lambda/` are early scaffolding toward a future serverless deployment on AWS.

## License

Not yet specified.
