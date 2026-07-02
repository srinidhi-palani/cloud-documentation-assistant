import json
import traceback
from ingestion.embedder import get_embeddings
from retrieval.retriever import retrieve_and_format
from generation.qa_chain import answer_question
from generation.template_chain import generate_template, is_template_request


# Global embeddings — loaded once, reused across warm invocations
_embeddings = None


def get_cached_embeddings():
    """Load embeddings once and cache in memory"""
    global _embeddings
    if _embeddings is None:
        print("Initializing embeddings...")
        _embeddings = get_embeddings()
        print("Embeddings initialized")
    return _embeddings


def lambda_handler(event, context):
    """
    Lambda handler for live user queries
    Triggered by Lambda Function URL
    """
    print("Orchestrator Lambda triggered")

    try:
        # Step 1 — parse request body
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        elif isinstance(event.get("body"), dict):
            body = event["body"]
        else:
            body = event

        query = body.get("query", "").strip()

        if not query:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "No query provided",
                    "status": "failed"
                })
            }

        print(f"Received query: {query}")

        # Step 2 — get cached embeddings
        embeddings = get_cached_embeddings()

        # Step 3 — intent check: Q&A or template?
        if is_template_request(query):
            print("Intent: Template generation")
            result = generate_template(query, embeddings)

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "type": "template",
                    "request": result["request"],
                    "template": result["template"],
                    "sources": result["sources"],
                    "status": "success"
                })
            }

        else:
            print("Intent: Q&A")
            result = answer_question(query, embeddings)

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "type": "qa",
                    "question": result["question"],
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "status": "success"
                })
            }

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        print(f"Orchestrator failed: {error_msg}")
        print(f"Traceback: {traceback_str}")

        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": error_msg,
                "status": "failed"
            })
        }