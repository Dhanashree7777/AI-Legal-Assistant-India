import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


class RAGService:
    def __init__(self):
        # 1. Initialize Google Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.db_path = "./legal_faiss_db"
        self.vector_store = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Loads local FAISS store if available."""
        if os.path.exists(self.db_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS store: {e}")
                self.vector_store = None

    def process_and_create_embeddings(self, file_path: str) -> dict:
        """Parses a newly uploaded PDF, creates embeddings, and merges into FAISS."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)
        total_chunks = len(chunks)

        if total_chunks == 0:
            return {"status": "error", "message": "No readable text found in document."}

        # Safe API batching
        batch_size = 10
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                self.vector_store.add_documents(batch)
            
            # Save incrementally
            self.vector_store.save_local(self.db_path)
            
            # Rate limiting pause if more batches remain
            if i + batch_size < total_chunks:
                time.sleep(20)

        return {
            "status": "success",
            "total_chunks": total_chunks,
            "filename": os.path.basename(file_path)
        }

    def get_retriever(self):
        if self.vector_store is None:
            self._load_vector_store()
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized or database is empty.")
        return self.vector_store.as_retriever(search_kwargs={"k": 5})