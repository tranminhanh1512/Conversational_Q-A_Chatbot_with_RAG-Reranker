import re
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> dict:
        return {"answer": self.extract_answer(text)}
    
    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        return text_response.strip()

class Offline_RAG:
    def __init__(self, llm, retriever, session_store=None, max_history_messages=5, max_doc_tokens=300, reranker=None):
        """
        Initialize the History-Aware RAG system with reranking.

        :param llm: The language model used for generating answers.
        :param retriever: The retriever used to fetch relevant context from the knowledge base.
        :param session_store: A custom session store to manage conversation history.
        :param max_history_messages: Maximum number of chat history messages to include.
        :param max_doc_tokens: Maximum number of tokens per retrieved document.
        :param reranker: A reranking model (e.g., CrossEncoder) to reorder retrieved documents (optional).
        """
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker if reranker else CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.session_store = session_store or {}
        self.max_history_messages = max_history_messages
        self.max_doc_tokens = max_doc_tokens

        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do NOT answer the question."
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.str_parser = Str_OutputParser()
        self.cached_documents = None

    def load_documents(self, documents):
        if self.cached_documents is None:
            self.cached_documents = self.truncate_documents(documents)
        return self.cached_documents

    def truncate_documents(self, documents):
        """Truncate each document to max_doc_tokens."""
        if not documents:
            raise ValueError("No documents provided")
        truncated_docs = []
        for doc in documents:
            if not hasattr(doc, 'page_content') or not isinstance(doc.page_content, str):
                raise ValueError(f"Invalid document: {doc}")
            doc_length = len(doc.page_content.split())
            if doc_length > self.max_doc_tokens:
                truncated_text = ' '.join(doc.page_content.split()[:self.max_doc_tokens])
                truncated_doc = Document(page_content=truncated_text, metadata=doc.metadata)
                truncated_docs.append(truncated_doc)
            else:
                truncated_docs.append(doc)
        if not truncated_docs:
            raise ValueError("No valid documents after processing")
        return truncated_docs

    def rerank_documents(self, query: str, documents: list) -> list:
        """
        Rerank retrieved documents using a cross-encoder model.

        :param query: The input query.
        :param documents: List of retrieved documents.
        :return: List of reranked documents.
        """
        if not self.reranker or not documents:
            return documents
        
        # Prepare query-document pairs for scoring
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores from the cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Pair documents with their scores and sort
        scored_docs = [(doc, score) for doc, score in zip(documents, scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked documents
        return [doc for doc, _ in scored_docs]

    def get_conversational_rag_chain(self):
        # Create the history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=self.contextualize_q_prompt
        ).with_config({"run_name": "history_aware_retriever", "search_kwargs": {"k": 5}})

        # Create the question-answering chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the question. "
            "If you don't know the answer, say so. "
            "Keep the answer concise (max 3 sentences) and start with 'Answer:'."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = self.create_stuff_documents_chain(qa_prompt)

        # Create the retrieval chain with reranking
        retrieval_chain = self.create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Return the conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return conversational_rag_chain

    def create_stuff_documents_chain(self, qa_prompt):
        """Create a chain for answering questions."""
        return qa_prompt | self.llm | self.str_parser

    def create_retrieval_chain(self, history_aware_retriever, question_answer_chain):
        """Create a retrieval chain with reranking."""
        def combine_input_and_docs(x):
            # x is a dict with 'input' and 'chat_history' from the chain input
            # Retrieve documents using history_aware_retriever
            docs = history_aware_retriever.invoke(x)
            # Pass the input query and retrieved documents to rerank_documents
            reranked_docs = self.rerank_documents(x["input"], docs)
            return {
                "context": self.truncate_documents(reranked_docs),
                "input": x["input"],
                "chat_history": x.get("chat_history", [])
            }

        return {
            "context": RunnableLambda(combine_input_and_docs),
            "input": RunnableLambda(lambda x: self._validate_input(x, "input")),
            "chat_history": RunnableLambda(lambda x: self.get_session_history(self._validate_input(x, "session_id")).messages[-self.max_history_messages:])
        } | question_answer_chain

    def _validate_input(self, x, key):
        """Validate that x is a dictionary and contains the key."""
        if not isinstance(x, dict):
            raise ValueError(f"Expected a dictionary for {key}, got {type(x)}: {x}")
        value = x.get(key, "")
        if not value:
            raise ValueError(f"{key} cannot be empty")
        return value

    def get_session_history(self, session: str) -> BaseChatMessageHistory:
        """Retrieve session history based on session ID, using file-based persistence."""
        if not session:
            raise ValueError("Session ID cannot be empty")
        if session not in self.session_store:
            self.session_store[session] = FileChatMessageHistory(f"history_{session}.json")
        return self.session_store[session]