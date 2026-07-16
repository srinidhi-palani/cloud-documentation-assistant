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
MAX_IMPROVISE_ROUNDS = 1

def _is_affirmative(text):
    """Loose match for a 'yes' reply to the improvise prompt."""
    text = text.strip().lower().strip(".!?")
    return text in ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead", "please", "yes please")

def _is_negative(text):
    """Loose match for a 'no' reply to the improvise prompt — checked
    explicitly so a decline never falls through to the Q&A path with 'no'
    itself as the question (previously burned an LLM call on a non-question
    and returned a confusing, irrelevant answer)."""
    text = text.strip().lower().strip(".!?")
    return text in ("no", "n", "nope", "nah", "no thanks", "not now", "skip", "cancel")

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
                pending = st.session_state.get("pending_improvise")
                handled_as_improvise = False

                if pending and _is_affirmative(query):
                    handled_as_improvise = True
                    improve_request = (
                        "Apply ONLY the following specific improvements to the template above. "
                        "Do not change, add, or remove anything else in the template beyond what "
                        "each item below requires:\n"
                        + "\n".join(f"- {s}" for s in pending["suggestions"])
                        + "\n\nReturn the complete corrected template."
                    )
                    st.session_state.pending_improvise = None

                    previous_template = None
                    for hist_msg in reversed(st.session_state.messages):
                        if hist_msg.get("role") == "assistant" and hist_msg.get("type") == "template":
                            previous_template = hist_msg.get("template")
                            break

                    recent_history = st.session_state.messages[-10:]
                    result = generate_template(improve_request, embeddings, chat_history=recent_history)

                    fmt = result.get("format", "cloudformation_yaml")
                    lang = "json" if fmt == "terraform_json" else "yaml"
                    ext = "tf.json" if fmt == "terraform_json" else "yaml"
                    mime = "application/json" if fmt == "terraform_json" else "text/yaml"
                    label = "Terraform (JSON)" if fmt == "terraform_json" else "CloudFormation"

                    if result.get("repair_status") in ("repair_failed", "lint_skipped"):
                        st.warning(
                            "⚠️ This template still has unresolved validation issues after the "
                            "automatic repair pass. Review it carefully before deploying — check "
                            "the terminal log for the specific cfn-lint errors or static-check warnings."
                        )

                    if previous_template is not None and result["template"].strip() == previous_template.strip():
                        st.warning(
                            "⚠️ The model did not actually apply the requested improvements — the "
                            "returned template is identical to the previous version. Try replying "
                            "'yes' again, or apply the change manually."
                        )

                    st.markdown(f"Here is your improved {label} template:")
                    st.code(result["template"], language=lang)
                    st.download_button(
                        label="⬇️ Download Template",
                        data=result["template"],
                        file_name=f"template.{ext}",
                        mime=mime,
                        key="download_improved"
                    )
                    if result["sources"]:
                        st.caption(f"Sources: {', '.join(result['sources'])}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "template",
                        "content": f"Here is your improved {label} template:",
                        "template": result["template"],
                        "format": fmt,
                        "sources": result["sources"]
                    })

                    # Offer another round if the improved template still has suggestions
                    new_suggestions = result.get("suggestions", [])
                    prior_round = pending.get("round", 0)
                    if new_suggestions and prior_round < MAX_IMPROVISE_ROUNDS:
                        bullets = "\n".join(f"- {s}" for s in new_suggestions)
                        improvise_msg = f"Do you want to improvise with:\n{bullets}\n\n(yes/no)"
                        st.markdown(improvise_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "improvise_prompt",
                            "content": improvise_msg
                        })
                        st.session_state.pending_improvise = {"suggestions": new_suggestions, "round": prior_round + 1}

                elif pending and _is_negative(query):
                    handled_as_improvise = True
                    st.session_state.pending_improvise = None
                    decline_msg = "No problem — keeping the template as is."
                    st.markdown(decline_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "qa",
                        "content": decline_msg
                    })

                elif pending:
                    # Ambiguous reply — not recognized as yes or no. Clear the
                    # pending offer and fall through to handle the message
                    # normally (it may be a new, unrelated request).
                    st.session_state.pending_improvise = None

                if not handled_as_improvise:
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

                        if result.get("repair_status") in ("repair_failed", "lint_skipped"):
                            st.warning(
                                "⚠️ This template still has unresolved validation issues after the "
                                "automatic repair pass. Review it carefully before deploying — check "
                                "the terminal log for the specific cfn-lint errors or static-check warnings."
                            )

                        st.markdown(f"Here is your {label} template:")
                        st.code(result["template"], language=lang)

                        st.download_button(
                            label="⬇️ Download Template",
                            data=result["template"],
                            file_name=f"template.{ext}",
                            mime=mime,
                            key="download_latest"
                        )

                        if result["sources"]:
                            st.caption(f"Sources: {', '.join(result['sources'])}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "template",
                            "content": f"Here is your {label} template:",
                            "template": result["template"],
                            "format": fmt,
                            "sources": result["sources"]
                        })

                        suggestions = result.get("suggestions", [])
                        if suggestions:
                            bullets = "\n".join(f"- {s}" for s in suggestions)
                            improvise_msg = f"Do you want to improvise with:\n{bullets}\n\n(yes/no)"
                            st.markdown(improvise_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "improvise_prompt",
                                "content": improvise_msg
                            })
                            st.session_state.pending_improvise = {"suggestions": suggestions, "round": 0}

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

                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "qa",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })

            except Exception as e:
                st.error(f"Error: {str(e)}")


# import streamlit as st
# import sys
# import os

# # Add project root to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from ingestion.embedder import get_embeddings
# from generation.qa_chain import answer_question
# from generation.template_chain import generate_template, is_template_request

# # Page config
# st.set_page_config(
#     page_title="Cloud Documentation Assistant",
#     page_icon="☁️",
#     layout="wide"
# )

# # ---------------------------------------------------------------------------
# # ChatGPT-style theme: dark sidebar, light main area, left/right chat bubbles
# # ---------------------------------------------------------------------------
# st.markdown("""
# <style>
#     /* Light main area, dark sidebar — ChatGPT look */
#     .stApp { background-color: #ffffff; }

#     section[data-testid="stSidebar"] {
#         background-color: #171717;
#     }
#     section[data-testid="stSidebar"] * {
#         color: #ececec !important;
#     }
#     section[data-testid="stSidebar"] hr {
#         border-color: #2f2f2f;
#     }

#     #MainMenu, footer, header { visibility: hidden; }

#     h1 {
#         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
#         font-weight: 700;
#         color: #0d0d0d;
#     }

#     [data-testid="stChatMessageContent"] p {
#         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
#         font-size: 16px;
#         color: #0d0d0d;
#     }

#     /* Chat input — white box, black border, roomier, black circular send button */
#     [data-testid="stChatInput"],
#     [data-testid="stChatInput"] > div,
#     [data-testid="stChatInput"] section,
#     [data-testid="stChatInput"] div {
#         border: none !important;
#         box-shadow: none !important;
#         outline: none !important;
#         background-color: transparent !important;
#     }
#     [data-testid="stChatInput"] div:has(> textarea) {
#         position: relative !important;
#     }
#     [data-testid="stChatInput"] textarea {
#         border: 1.5px solid #1a1a1a !important;
#         border-radius: 24px !important;
#         background-color: #ffffff !important;
#         color: #1a1a1a !important;
#         box-shadow: none !important;
#         outline: none !important;
#         padding: 14px 60px 14px 20px !important;
#         font-size: 16px !important;
#         min-height: 54px !important;
#         width: 100% !important;
#         box-sizing: border-box !important;
#     }
#     [data-testid="stChatInput"] textarea::placeholder {
#         color: #8a8a8a !important;
#     }
#     [data-testid="stChatInput"] textarea:focus {
#         border: 1.5px solid #1a1a1a !important;
#         box-shadow: 0 0 0 1px #1a1a1a !important;
#     }
#     [data-testid="stChatInput"] button {
#         position: absolute !important;
#         right: 35px !important;
#         top: 50% !important;
#         transform: translateY(-65%) !important;
#         margin: 0 !important;
#         background-color: #1a1a1a !important;
#         border-radius: 50% !important;
#         border: none !important;
#         width: 36px !important;
#         height: 36px !important;
#         min-width: 36px !important;
#         min-height: 36px !important;
#         z-index: 10 !important;
#         display: flex !important;
#         align-items: center !important;
#         justify-content: center !important;
#     }
#     [data-testid="stChatInput"] button svg {
#         fill: #ffffff !important;
#     }

#     .stButton button, .stDownloadButton button {
#         border-radius: 8px;
#         border: 1px solid #d9d9e3;
#         background-color: #ffffff;
#         color: #0d0d0d;
#     }
#     .stButton button:hover, .stDownloadButton button:hover {
#         background-color: #f7f7f8;
#         border-color: #0d0d0d;
#     }

#     /* Sidebar buttons need their own dark-mode styling, not the light default above */
#     section[data-testid="stSidebar"] .stButton button,
#     section[data-testid="stSidebar"] .stDownloadButton button {
#         background-color: #2f2f2f !important;
#         color: #ececec !important;
#         border: 1px solid #4d4d4d !important;
#     }
#     section[data-testid="stSidebar"] .stButton button:hover,
#     section[data-testid="stSidebar"] .stDownloadButton button:hover {
#         background-color: #3d3d3d !important;
#         border-color: #6e6e6e !important;
#     }

#     pre { border-radius: 8px; }

#     /* --- Chat bubbles, left/right like a real chat --- */
#     [data-testid="stChatMessage"] {
#         display: flex;
#         gap: 0.6rem;
#         padding: 0.4rem 0;
#         max-width: 100%;
#         background-color: transparent;
#         border: none;
#     }

#     [data-testid="stChatMessage"] {
#         flex-wrap: nowrap;
#         align-items: flex-start;
#     }
#     [data-testid="stChatMessage"] img {
#         flex-shrink: 0;
#     }

#     /* Avatar colors — override Streamlit's default red (user) / orange (assistant) */
#     [data-testid="stChatMessageAvatarUser"] {
#         background-color: #d9d9d9  !important;   /* teal-green */
#     }
#     [data-testid="stChatMessageAvatarAssistant"] {
#         background-color: #000000  !important;    /* purple */
#     }

#     /* Assistant → left aligned (default row) */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
#         flex-direction: row;
#         justify-content: flex-start;
#     }
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
#         background-color: #f4f4f4;
#         border-radius: 18px;
#         padding: 10px 16px;
#         max-width: 70% !important;
#         width: fit-content !important;
#         flex-grow: 0 !important;
#         flex-shrink: 1 !important;
#         margin-right: auto !important;
#         margin-left: 0 !important;
#     }

#     /* User → right aligned (reversed row) */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
#         flex-direction: row-reverse;
#         justify-content: flex-end;
#     }
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
#         background-color: #1a1a1a;
#         border-radius: 18px;
#         padding: 10px 16px;
#         max-width: 70% !important;
#         width: fit-content !important;
#         flex-grow: 0 !important;
#         flex-shrink: 1 !important;
#         margin-left: auto !important;
#         margin-right: 0 !important;
#     }

#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
#         color: #ffffff !important;
#     }

#     /* Code blocks inside bubbles shouldn't be squeezed */
#     [data-testid="stChatMessage"] pre {
#         max-width: 100%;
#         overflow-x: auto;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Title
# st.title("☁️ Cloud Documentation Assistant")
# st.markdown("Ask questions about AWS services or generate CloudFormation / Terraform templates.")
# st.divider()

# # Initialize embeddings once using session state
# @st.cache_resource
# def load_embeddings():
#     with st.spinner("Loading embedding model..."):
#         embeddings = get_embeddings()
#     return embeddings

# embeddings = load_embeddings()
# MAX_IMPROVISE_ROUNDS = 1

# def _is_affirmative(text):
#     """Loose match for a 'yes' reply to the improvise prompt."""
#     text = text.strip().lower().strip(".!?")
#     return text in ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead", "please", "yes please")

# # Sidebar
# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This assistant uses RAG (Retrieval Augmented Generation) to:
#     - Answer questions about AWS services
#     - Generate CloudFormation templates (YAML)
#     - Generate Terraform templates (JSON)
#     """)
#     st.divider()
#     st.header("Example Questions")
#     st.markdown("""
#     - What resources are in the CloudFormation template?
#     - What is Amazon ECS?
#     - Generate a CloudFormation template for an S3 bucket
#     - Generate the same thing in Terraform JSON
#     - Explain the parameters in the template
#     - Fix the IAM role in the template above
#     """)
#     st.divider()
#     st.header("Stack")
#     st.markdown("""
#     - **Embeddings:** all-MiniLM-L6-v2
#     - **LLM:** Gemma 4 31B (OpenRouter)
#     - **Vector DB:** FAISS
#     - **Storage:** Amazon S3
#     - **IaC formats:** CloudFormation (YAML), Terraform (JSON)
#     """)
#     st.divider()
#     # Clear conversation button
#     if st.button("🗑️ Clear Conversation"):
#         st.session_state.messages = []
#         st.rerun()

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         if message.get("type") == "template":
#             st.markdown(message["content"])
#             if "template" in message:
#                 fmt = message.get("format", "cloudformation_yaml")
#                 lang = "json" if fmt == "terraform_json" else "yaml"
#                 ext = "tf.json" if fmt == "terraform_json" else "yaml"
#                 mime = "application/json" if fmt == "terraform_json" else "text/yaml"
#                 st.code(message["template"], language=lang)
#                 # Download button for historical templates
#                 st.download_button(
#                     label="⬇️ Download Template",
#                     data=message["template"],
#                     file_name=f"template.{ext}",
#                     mime=mime,
#                     key=f"download_{st.session_state.messages.index(message)}"
#                 )
#         else:
#             st.markdown(message["content"])
#         if "sources" in message and message["sources"]:
#             st.caption(f"Sources: {', '.join(message['sources'])}")

# # Chat input
# if query := st.chat_input("Ask a question or request a template..."):

#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(query)

#     # Add user message to history
#     st.session_state.messages.append({
#         "role": "user",
#         "content": query
#     })

#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 pending = st.session_state.get("pending_improvise")
#                 handled_as_improvise = False

#                 if pending and _is_affirmative(query):
#                     handled_as_improvise = True
#                     improve_request = (
#                         "Apply the following improvements to the template above and "
#                         "regenerate the FULL corrected template:\n"
#                         + "\n".join(f"- {s}" for s in pending["suggestions"])
#                     )
#                     st.session_state.pending_improvise = None

#                     recent_history = st.session_state.messages[-10:]
#                     result = generate_template(improve_request, embeddings, chat_history=recent_history)

#                     fmt = result.get("format", "cloudformation_yaml")
#                     lang = "json" if fmt == "terraform_json" else "yaml"
#                     ext = "tf.json" if fmt == "terraform_json" else "yaml"
#                     mime = "application/json" if fmt == "terraform_json" else "text/yaml"
#                     label = "Terraform (JSON)" if fmt == "terraform_json" else "CloudFormation"

#                     st.markdown(f"Here is your improved {label} template:")
#                     st.code(result["template"], language=lang)
#                     st.download_button(
#                         label="⬇️ Download Template",
#                         data=result["template"],
#                         file_name=f"template.{ext}",
#                         mime=mime,
#                         key="download_improved"
#                     )
#                     if result["sources"]:
#                         st.caption(f"Sources: {', '.join(result['sources'])}")

#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "type": "template",
#                         "content": f"Here is your improved {label} template:",
#                         "template": result["template"],
#                         "format": fmt,
#                         "sources": result["sources"]
#                     })

#                     # Offer another round if the improved template still has suggestions
#                     new_suggestions = result.get("suggestions", [])
#                     prior_round = pending.get("round", 0)
#                     if new_suggestions and prior_round < MAX_IMPROVISE_ROUNDS:
#                         bullets = "\n".join(f"- {s}" for s in new_suggestions)
#                         improvise_msg = f"Do you want to improvise with:\n{bullets}\n\n(yes/no)"
#                         st.markdown(improvise_msg)
#                         st.session_state.messages.append({
#                             "role": "assistant",
#                             "type": "improvise_prompt",
#                             "content": improvise_msg
#                         })
#                         st.session_state.pending_improvise = {"suggestions": new_suggestions, "round": prior_round + 1}

#                 elif pending:
#                     # User declined or said something unrelated — clear the
#                     # pending offer and fall through to handle their message normally.
#                     st.session_state.pending_improvise = None

#                 if not handled_as_improvise:
#                     if is_template_request(query):
#                         # Template generation path — passes chat history so the
#                         # model remembers the last template (and its format) and
#                         # can fix/update it without the user re-pasting it.
#                         recent_history = st.session_state.messages[-10:]

#                         result = generate_template(query, embeddings, chat_history=recent_history)

#                         fmt = result.get("format", "cloudformation_yaml")
#                         lang = "json" if fmt == "terraform_json" else "yaml"
#                         ext = "tf.json" if fmt == "terraform_json" else "yaml"
#                         mime = "application/json" if fmt == "terraform_json" else "text/yaml"
#                         label = "Terraform (JSON)" if fmt == "terraform_json" else "CloudFormation"

#                         st.markdown(f"Here is your {label} template:")
#                         st.code(result["template"], language=lang)

#                         st.download_button(
#                             label="⬇️ Download Template",
#                             data=result["template"],
#                             file_name=f"template.{ext}",
#                             mime=mime,
#                             key="download_latest"
#                         )

#                         if result["sources"]:
#                             st.caption(f"Sources: {', '.join(result['sources'])}")

#                         st.session_state.messages.append({
#                             "role": "assistant",
#                             "type": "template",
#                             "content": f"Here is your {label} template:",
#                             "template": result["template"],
#                             "format": fmt,
#                             "sources": result["sources"]
#                         })

#                         suggestions = result.get("suggestions", [])
#                         if suggestions:
#                             bullets = "\n".join(f"- {s}" for s in suggestions)
#                             improvise_msg = f"Do you want to improvise with:\n{bullets}\n\n(yes/no)"
#                             st.markdown(improvise_msg)
#                             st.session_state.messages.append({
#                                 "role": "assistant",
#                                 "type": "improvise_prompt",
#                                 "content": improvise_msg
#                             })
#                             st.session_state.pending_improvise = {"suggestions": suggestions, "round": 0}

#                     else:
#                         # Q&A path — pass last 10 messages as chat history
#                         recent_history = st.session_state.messages[-10:]

#                         result = answer_question(
#                             query,
#                             embeddings,
#                             chat_history=recent_history
#                         )

#                         st.markdown(result["answer"])

#                         if result["sources"]:
#                             st.caption(f"Sources: {', '.join(result['sources'])}")

#                         st.session_state.messages.append({
#                             "role": "assistant",
#                             "type": "qa",
#                             "content": result["answer"],
#                             "sources": result["sources"]
#                         })

#             except Exception as e:
#                 st.error(f"Error: {str(e)}")