# Conversational Q&A Chatbot with RAG and Reranker

## Overview
The Conversational Q&A Chatbot with Retrieval-Augmented Generation (RAG) is a cutting-edge system designed to deliver accurate, context-aware answers to user inquiries by utilizing advanced natural language processing (NLP) methods. The chatbot leverages a RAG model, which combines the power of a pre-trained language model with real-time document retrieval, enabling it to generate high-quality, informative responses based on the context of the retrieved content.

To enhance the model’s performance, the system includes a reranking mechanism, which assesses the relevance of retrieved documents before they are processed by the language model. This ensures that only the most relevant information is selected, improving the quality and accuracy of the responses. By integrating RAG for contextual retrieval and reranking for relevance, the chatbot is capable of handling complex queries and providing users with highly accurate and insightful answers, particularly in contexts where detailed information is critical.

This project aims to create an intelligent assistant that can be applied across a variety of domains—ranging from customer support to educational tools—enabling dynamic conversations, understanding user intent, and delivering responses grounded in relevant document retrieval.


## Project structure
```
Conversational_Q&A_Chatbot_with_RAG&Reranker/
├── app.py
├── data_source/
│   ├── data_acquisition/
│   │   ├── DataAcquisition.py
│   │   └── arxiv_large_language_model.json
│   └── generative_ai/
│       ├── data/
│       ├── data_test/  
│       ├── download.py
│       └── remove_invalid_pdfs.py
├── history_default_session.json
├── requirements.txt
└── src/
    ├── base/
    │   ├── __init__.py
    │   └── llm_model.py
    └── rag/
        ├── file_loader.py
        ├── main.py
        ├── offline_rag.py
        ├── utils.py
        └── vector_store.py
```
- **data_source**: A directory used to store documents for building the vector database system.
- **_data_source/data_acquisition/DataAcquisition.py**: A code file responsible for acquiring and processing data from various sources, preparing it for storage in the vector database.
- **arxiv_large_languague_model.json**: A configuration file that contains metadata related to a large language model (LLM) from Arxiv data.
- **_data_source/generative_ai/download.py**: A code file used to automatically download several research papers in PDF format.
- **_data_source/generative_ai/data**: A directory contains all PDFs file from Arxiv
- **_data_source/generative_ai/data_test**: A directory containing test datasets for evaluating the generative AI model
- **src**: The main source code directory for the project, containing subdirectories for various components.
- **_src/base/llm_model.py**: A code file that defines and initializes the large language model (Ollama 3.2B) used in the project.
- **_src/rag/file_loader.py**: A code file responsible for loading and preprocessing documents or files to be used in the retrieval-augmented generation (RAG) system.
- **_src/rag/main.py**: The main execution script for the RAG system, orchestrating the flow of data and model interactions
- **_src/rag/offline_rag.py**: A code file that implements the offline version of the RAG system, which can perform inference without requiring real-time access to the data.
- **_src/rag/utils.py**: A code file used to define functions for extracting answers from the model's output.
- **_src/rag/vector_store.py**: A code file that implements the storage and management of vectorized representations of documents.
- **app.py**: A Streamlit application that allows users to interact with the chatbot built using the Retrieval-Augmented Generation (RAG) system. It provides an interactive interface for users to ask questions and receive answers from the model, enabling real-time conversation with the chatbot.
- **requirements.txt**: A file listing all the necessary Python dependencies and libraries required to run the project.
- **.env**: Environment variables.

## Key Features 
- Conversational Q&A System
- Retrieval-Augmented Generation (RAG)
- Reranking Mechanism
- Document Acquisition and Processing
- Vector Database for Efficient Retrieval
- Interactive Streamlit Interface

## Technologies Used
- Python
- Streamlit
- Ollama
- Chroma
- RAG

