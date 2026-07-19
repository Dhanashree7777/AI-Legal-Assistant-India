import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


class RAGService:
    def __init__(self):

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name="legal_documents",
            persist_directory="./legal_vector_db"
        )


    def process_and_create_embeddings(
            self,
            file_path="./data/Constitution_of_India.pdf"
    ):

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )

        chunks = splitter.split_documents(pages)

        self.vector_store.add_documents(chunks)

        print(f"Added {len(chunks)} chunks.")


    def get_retriever(self):

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 8
            }
        )

        return retriever


if __name__ == "__main__":

    rag_service = RAGService()

    # Run only once for each document
    rag_service.process_and_create_embeddings("./data/Constitution_of_India.pdf")
    # rag_service.process_and_create_embeddings("./data/BNS.pdf")
    # rag_service.process_and_create_embeddings("./data/BNSS.pdf")
    # rag_service.process_and_create_embeddings("./data/Indian_Evidence_Act.pdf")

    retriever = rag_service.get_retriever()

    docs = retriever.invoke(
        "What is Article 21 of the Constitution?"
    )

    for doc in docs:
        print(doc.page_content[:500])
        print("-" * 100)