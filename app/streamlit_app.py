import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.embedder import get_embeddings
from generation.qa_chain import answer_question
from generation.template_chain import generate_template, is_template_request

# Page config
st.set_page_config(
    page_title="Cloud Documentation Assistant",
    page_icon="☁️",
    layout="wide"
)

# Title
st.title("☁️ Cloud Documentation Assistant")
st.markdown("Ask questions about AWS services or generate CloudFormation / Terraform templates.")
st.divider()

# Initialize embeddings once using session state
@st.cache_resource
def load_embeddings():
    with st.spinner("Loading embedding model..."):
        embeddings = get_embeddings()
    return embeddings

embeddings = load_embeddings()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant uses RAG (Retrieval Augmented Generation) to:
    - Answer questions about AWS services
    - Generate CloudFormation templates (YAML)
    - Generate Terraform templates (JSON)
    """)
    st.divider()
    st.header("Example Questions")
    st.markdown("""
    - What resources are in the CloudFormation template?
    - What is Amazon ECS?
    - Generate a CloudFormation template for an S3 bucket
    - Generate the same thing in Terraform JSON
    - Explain the parameters in the template
    - Fix the IAM role in the template above
    """)
    st.divider()
    st.header("Stack")
    st.markdown("""
    - **Embeddings:** all-MiniLM-L6-v2
    - **LLM:** Gemma 4 31B (OpenRouter)
    - **Vector DB:** FAISS
    - **Storage:** Amazon S3
    - **IaC formats:** CloudFormation (YAML), Terraform (JSON)
    """)
    st.divider()
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "template":
            st.markdown(message["content"])
            if "template" in message:
                fmt = message.get("format", "cloudformation_yaml")
                lang = "json" if fmt == "terraform_json" else "yaml"
                ext = "tf.json" if fmt == "terraform_json" else "yaml"
                mime = "application/json" if fmt == "terraform_json" else "text/yaml"
                st.code(message["template"], language=lang)
                # Download button for historical templates
                st.download_button(
                    label="⬇️ Download Template",
                    data=message["template"],
                    file_name=f"template.{ext}",
                    mime=mime,
                    key=f"download_{st.session_state.messages.index(message)}"
                )
        else:
            st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"Sources: {', '.join(message['sources'])}")

# Chat input
if query := st.chat_input("Ask a question or request a template..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if is_template_request(query):
                    # Template generation path — passes chat history so the
                    # model remembers the last template (and its format) and
                    # can fix/update it without the user re-pasting it.
                    recent_history = st.session_state.messages[-10:]

                    result = generate_template(query, embeddings, chat_history=recent_history)

                    fmt = result.get("format", "cloudformation_yaml")
                    lang = "json" if fmt == "terraform_json" else "yaml"
                    ext = "tf.json" if fmt == "terraform_json" else "yaml"
                    mime = "application/json" if fmt == "terraform_json" else "text/yaml"
                    label = "Terraform (JSON)" if fmt == "terraform_json" else "CloudFormation"

                    st.markdown(f"Here is your {label} template:")
                    st.code(result["template"], language=lang)

                    # Download button for generated template
                    st.download_button(
                        label="⬇️ Download Template",
                        data=result["template"],
                        file_name=f"template.{ext}",
                        mime=mime,
                        key="download_latest"
                    )

                    if result["sources"]:
                        st.caption(f"Sources: {', '.join(result['sources'])}")

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "template",
                        "content": f"Here is your {label} template:",
                        "template": result["template"],
                        "format": fmt,
                        "sources": result["sources"]
                    })

                else:
                    # Q&A path — pass last 10 messages as chat history
                    recent_history = st.session_state.messages[-10:]

                    result = answer_question(
                        query,
                        embeddings,
                        chat_history=recent_history
                    )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        st.caption(f"Sources: {', '.join(result['sources'])}")

                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "qa",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

            except Exception as e:
                st.error(f"Error: {str(e)}")