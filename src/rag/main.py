import os
from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vector_store import VectorDB
from src.rag.offline_rag import Offline_RAG
import streamlit as st
from sentence_transformers import CrossEncoder

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

def build_rag_chain(llm, data_dir, data_type):
    """
    Build a conversational RAG chain with a cross-encoder reranker.
    
    :param llm: The language model for generating answers.
    :param data_dir: Directory containing the data files (e.g., PDFs).
    :param data_type: Type of files to load (e.g., 'pdf').
    :return: Conversational RAG chain.
    """
    # Validate data directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Check if the vector store is already cached in session state
    if "vector_db" not in st.session_state:
        # Load documents
        try:
            doc_loader = Loader(file_type=data_type).load_dir(data_dir, workers=1)
            if not doc_loader:
                raise ValueError(f"No {data_type} documents found in {data_dir}")
            # Create and cache the retriever
            retriever = VectorDB(documents=doc_loader).get_retriever(search_kwargs={"k": 5})  # Increased k for reranking
            st.session_state.vector_db = retriever
        except Exception as e:
            raise ValueError(f"Failed to load documents or create retriever: {str(e)}")

    # Retrieve the cached retriever
    retriever = st.session_state.vector_db

    # Initialize cross-encoder for reranking
    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    except Exception as e:
        raise ValueError(f"Failed to initialize cross-encoder reranker: {str(e)}")

    # Build the RAG chain
    try:
        rag_chain = Offline_RAG(
            llm=llm,
            retriever=retriever,
            reranker=reranker,
            session_store={},
            max_history_messages=5,
            max_doc_tokens=300
        )
        conversational_rag_chain = rag_chain.get_conversational_rag_chain()
        return conversational_rag_chain
    except Exception as e:
        raise ValueError(f"Failed to build RAG chain: {str(e)}")