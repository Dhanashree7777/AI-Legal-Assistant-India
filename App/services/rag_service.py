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

    def process_and_create_embeddings(self, file_path="./data/Constitution_of_India.pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(pages)
        self.vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {os.path.basename(file_path)}")

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k":5}
        )

    def retrieve_with_sources(self, query:str):
        docs=self.get_retriever().invoke(query)
        results=[]
        for d in docs:
            results.append({
                "content":d.page_content,
                "metadata":d.metadata
            })
        return results

if __name__=="__main__":
    rag=RAGService()
    rag.process_and_create_embeddings("./data/Constitution_of_India.pdf")