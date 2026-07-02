from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from retrieval.retriever import retrieve_and_format
from config.config import OPENROUTER_API_KEY, LLM_MODEL


def get_llm():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=1024
    )
    print(f"LLM initialized: {LLM_MODEL}")
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
        answer = response.content

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