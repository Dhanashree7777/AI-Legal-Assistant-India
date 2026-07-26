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
        # 1. Initialize Google Embeddings (using the supported 3072-dimension model)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.db_path = "./legal_faiss_db"
        self.vector_store = None

        # 2. Automatically load existing database if it exists
        if os.path.exists(self.db_path):
            print(f"Loading existing FAISS database from {self.db_path}...")
            # Note: allow_dangerous_deserialization=True is required for loading trusted local FAISS files
            self.vector_store = FAISS.load_local(
                folder_path=self.db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("No existing FAISS database found. Ready to process new documents.")


    def process_and_create_embeddings(self, file_path="./data/Constitution_of_India.pdf"):
        print(f"\nLoading document: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(pages)
        total_chunks = len(chunks)
        print(f"Total chunks generated: {total_chunks}")

        # Safe API Batching (10 chunks at a time, 30s pause) to prevent rate limit crashes
        batch_size = 10
        pause_time = 30

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i} to {min(i + batch_size, total_chunks)} of {total_chunks}...")

            if self.vector_store is None:
                # Create a new FAISS vector store in memory for the first batch
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                # Process new batch and merge it into the existing store
                temp_store = FAISS.from_documents(batch, self.embeddings)
                self.vector_store.merge_from(temp_store)

            # Save progress incrementally to disk so no data is lost during network interruptions
            self.vector_store.save_local(self.db_path)
            print(f"Successfully saved progress to {self.db_path}.")

            # Pause to respect Google API rate limits (skip pause on the very last batch)
            if i + batch_size < total_chunks:
                print(f"Pausing for {pause_time} seconds to cool down API quota...\n")
                time.sleep(pause_time)

        print(f"Finished processing and embedding all {total_chunks} chunks successfully!")


    def get_retriever(self):
        if self.vector_store is None:
            raise ValueError("The vector store is empty! Please process a document first.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 8
            }
        )
        return retriever


if __name__ == "__main__":
    rag_service = RAGService()

    # Step 1: Run ONLY ONCE for each document. 
    # Uncomment the document you want to process, run the script, then comment it out again.
    # rag_service.process_and_create_embeddings("./data/Constitution_of_India.pdf")
    # rag_service.process_and_create_embeddings("./data/BNS.pdf")
    # rag_service.process_and_create_embeddings("./data/BNSS.pdf")
    # rag_service.process_and_create_embeddings("./data/Indian_Evidence_Act.pdf")

    # Step 2: Test Retrieval (ensure you have processed at least one doc first)
    if rag_service.vector_store is not None:
        print("\nTesting Retriever...")
        retriever = rag_service.get_retriever()
        
        docs = retriever.invoke(
            "What is Article 21 of the Constitution?"
        )

        print("\n--- Retrieved Results ---")
        for doc in docs:
            print(doc.page_content[:500])
            print("-" * 100)
    else:
        print("\nNo data found in FAISS. Please uncomment a processing line above and run again to build the database.")