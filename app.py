import os
import uuid
import atexit
import streamlit as st
from dotenv import load_dotenv

import vector_store
import agent
import config



# --- Page Setup ---
st.set_page_config(page_title="Support Agent", page_icon="🤖", layout="centered")
st.title("🤖 Customer Support Agent")
st.markdown(
    "Upload your company's documents, and I will answer questions as a customer support agent."
)


# --- Session Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = None


# --- Utility Functions ---
def get_api_key():
    """Get the Google API key from environment or Streamlit secrets."""
    load_dotenv()
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY", None)
        if hasattr(st, "secrets")
        else None
    )
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found. Please set it in .env or Streamlit secrets.")
        st.stop()
    return api_key


def setup_retriever(source_type: str, api_key: str):
    """Initialize retriever for either default or user-uploaded documents."""
    collection_name = None

    if source_type == "Default Demo":
        collection_name = config.DEFAULT_COLLECTION

    elif source_type == "My Uploaded Files":
        collection_name = f"{config.USER_COLLECTION_PREFIX}{st.session_state.session_id}"
        uploaded_files = st.file_uploader(
            "Upload your .txt files", type=["txt"], accept_multiple_files=True
        )

        if uploaded_files and st.button("📁 Index Documents"):
            with st.spinner("Indexing uploaded files..."):
                try:
                    vector_store.load_user_docs(uploaded_files, api_key, st.session_state.session_id)
                    st.success("✅ Files indexed successfully!")
                except Exception as e:
                    st.error(f"Error indexing files: {e}")
                    st.stop()

    # Try to set up retriever
    if collection_name:
        try:
            st.session_state.retriever = vector_store.get_retriever(collection_name, api_key)
        except ValueError as e:
            if "My Uploaded Files" in source_type:
                st.warning("⚠️ Please upload and index your documents to activate the chat.")
            else:
                st.error(f"Retriever setup error: {e}")
            st.session_state.retriever = None


# --- Main Chat App ---
def main():
    api_key = get_api_key()

    if "stale_cleanup_done" not in st.session_state:
        vector_store.cleanup_stale_collections()
        st.session_state.stale_cleanup_done = True

    # Initialize default demo documents once
    try:
        vector_store.initialize_default_docs(api_key)
    except Exception as e:
        st.error(f"Error initializing default documents: {e}")
        st.stop()

    with st.sidebar:
        st.header("📚 Knowledge Source")
        source_type = st.radio(
            "Select documents to use:",
            ["Default Demo", "My Uploaded Files"],
            index=0,
        )

        st.header("⚙️ File Indexing")
        setup_retriever(source_type, api_key)

        st.header("💬 Chat Controls")
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.agent_graph = None
            st.success("Chat history cleared!")

        st.markdown("---")
        st.info(f"**Session ID:** `{st.session_state.session_id}`")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.retriever:
            st.error("⚠️ Please index or select a document source first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create agent graph if needed
        if not st.session_state.agent_graph:
            st.session_state.agent_graph = agent.create_agent_graph(api_key, st.session_state.retriever)

        # --- Streaming AI response ---
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            try:
                chat_history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
                graph = st.session_state.agent_graph

                for chunk in graph.stream(
                    {"question": prompt, "chat_history": chat_history}
                ):
                    content = ""
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk["content"]

                    if content:
                        full_response += content
                        response_container.markdown(full_response + "▌")

                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error during response generation: {e}")


# --- Run App ---
if __name__ == "__main__":
    main()
