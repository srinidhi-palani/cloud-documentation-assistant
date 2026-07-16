from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval.retriever import retrieve_and_format
from generation.template_chain import _strip_think_block  # shared Qwen <think> stripper
from langchain_aws import ChatBedrock
from config.config import BEDROCK_MODEL_ID, AWS_REGION, BEDROCK_PROFILE
import boto3

def get_llm():
    session = boto3.Session(profile_name=BEDROCK_PROFILE)
    llm = ChatBedrock(
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        client=session.client("bedrock-runtime", region_name=AWS_REGION),
        model_kwargs={
            "temperature": 0.4,
            "max_tokens": 4096,
        }
    )
    print(f"LLM initialized: Bedrock {BEDROCK_MODEL_ID}")
    return llm


def answer_question(question, embeddings, chat_history=[]):
    try:
        print(f"\nQuestion: {question}")

        print("Retrieving relevant chunks...")
        chunks, context = retrieve_and_format(question, embeddings)

        if not chunks:
            return {
                "question": question,
                "answer": "I could not find relevant information in the documentation.",
                "sources": []
            }

        llm = get_llm()

        # System message with context
        system_message = SystemMessage(content=f"""You are a cloud documentation assistant.
First try to answer using the context provided below.
If the answer is not in the context, use your general AWS knowledge to answer
but clearly state: "This answer is from general knowledge, not from your documentation."
Always cite which source you used.
Remember the conversation history and refer to previous messages when relevant.

Context:
{context}""")

        # Build messages list with history
        messages = [system_message]

        # Add previous conversation history
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current question
        messages.append(HumanMessage(content=question))

        print("Generating answer...")
        response = llm.invoke(messages)
        # Strip Qwen's <think> reasoning block so it never leaks into the
        # answer shown to the user in Streamlit.
        answer = _strip_think_block(response.content)

        sources = list(set([
            chunk.metadata.get("source_file", "unknown")
            for chunk in chunks
        ]))

        print(f"Answer generated successfully")
        print(f"Sources used: {sources}")

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": chunks
        }

    except Exception as e:
        print(f"Error generating answer: {e}")
        raise


def test_qa_chain(embeddings):
    test_question = "What resources are in the CloudFormation template?"
    result = answer_question(test_question, embeddings)
    print(f"\nTest Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    return result