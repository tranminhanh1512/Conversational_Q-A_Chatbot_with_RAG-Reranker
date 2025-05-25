import streamlit as st
from src.base.llm_model import get_ollama_llm
from src.rag.main import build_rag_chain
from dotenv import load_dotenv
import os

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Validate data directory
data_dir = "./data_source/generative_ai/data_test"
if not os.path.exists(data_dir):
    st.error(f"Data directory {data_dir} does not exist.")
    st.stop()

# Set up Streamlit interface
st.title("Conversational RAG With Pre-processed PDF Documents")
st.write("Chat with the content of pre-existing PDF files stored in the system.")

# Initialize the Ollama model
llm = get_ollama_llm(model_name="llama3.2", temperature=0.5)

# Define data type
data_type = "pdf"

# Build RAG chain using the cached vector store
if "rag_chain" not in st.session_state:
    with st.spinner("Building RAG chain..."):
        try:
            st.session_state.rag_chain = build_rag_chain(llm=llm, data_dir=data_dir, data_type=data_type)
        except Exception as e:
            st.error(f"Failed to build RAG chain: {str(e)}")
            st.stop()

# Streamlit input handling
session_id = st.text_input("Enter your session ID:", value="default_session")
user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Generating response..."):
        try:
            response = st.session_state.rag_chain.invoke(
                {"input": user_input, "session_id": session_id},
                config={"configurable": {"session_id": session_id}}
            )
            st.write(f"Assistant: {response['answer']}")
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display the conversation history
st.subheader("Conversation History")
try:
    session_history = st.session_state.rag_chain.get_session_history(session_id)
    if session_history.messages:
        for message in session_history.messages:
            st.write(f"{message.type.capitalize()}: {message.content}")
    else:
        st.write("No conversation history yet.")
except Exception as e:
    st.error(f"Error retrieving session history: {str(e)}")