import os
from App.services.rag_service import RAGService

DATA_DIR = "data"

rag = RAGService()

for filename in os.listdir(DATA_DIR):

    if filename.lower().endswith(".pdf"):

        path = os.path.join(DATA_DIR, filename)

        print(f"Processing: {filename}")

        try:
            result = rag.process_and_create_embeddings(path)
            print(result)

        except Exception as e:
            print(f"Failed: {filename}: {e}")

print("Document indexing completed.")