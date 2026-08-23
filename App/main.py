import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from App.workflows.legal_workflow import workflow
from App.services.chat_memory import ChatMemory
from App.services.rag_service import RAGService

# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

# ---------------------------------------------------------
# Initialize Memory & RAG Services
# ---------------------------------------------------------
chat_memory = ChatMemory()
rag_service = RAGService()


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------
app = FastAPI(
    title="AI Legal Assistant API",
    version="0.1.0",
    description="API for AI Legal Assistant for Indian Laws"
)


# ---------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {"message": "The AI Legal Assistant API is live......"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "OK"}


# ---------------------------------------------------------
# CRUD Section
# ---------------------------------------------------------
data = set()

@app.post("/add-item")
def add_new_item(item: str) -> dict:
    data.add(item)
    return {"message": f"{item} successfully added to the database"}

@app.get("/display-all-items")
def display_all_items() -> dict:
    return {"message": f"Following items are in the database: {data}"}

@app.get("/verify-item/{item_name}")
def verify_item(item_name: str) -> dict:
    if item_name in data:
        return {"message": f"{item_name} is present in the database"}
    return {"message": f"{item_name} is not present in the database"}

@app.put("/update_item")
def update_item(existing_item: str, new_item: str) -> dict:
    if existing_item not in data:
        return {"message": f"{existing_item} is not present in the database"}
    data.remove(existing_item)
    data.add(new_item)
    return {"message": f"{existing_item} has been updated to {new_item}"}

@app.delete("/delete-item")
def delete_item(item: str) -> dict:
    if item not in data:
        return {"message": f"{item} is not present in the database"}
    data.remove(item)
    return {"message": f"{item} has been deleted from the database"}

@app.delete("/delete-all-items")
def delete_all_items() -> dict:
    data.clear()
    return {"message": "All items have been deleted from the database"}


# ---------------------------------------------------------
# Document Upload & Ingestion Endpoint
# ---------------------------------------------------------
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("./data", exist_ok=True)
        file_path = os.path.join("./data", file.filename)

        # Save incoming file to local disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process and embed into FAISS vector database
        result = rag_service.process_and_create_embeddings(file_path)

        if result.get("status") == "success":
            return {
                "status": "success",
                "message": f"Successfully ingested {file.filename} ({result.get('total_chunks')} chunks added)."
            }
        else:
            return {
                "status": "error",
                "message": result.get("message", "Document ingestion failed.")
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ---------------------------------------------------------
# AI Legal Workflow Endpoint
# ---------------------------------------------------------
@app.post("/execute-workflow")
def process_workflow(request: QueryRequest):
    try:
        query = request.query.strip()
        if not query:
            return {
                "answer": "",
                "sources": [],
                "status": "error",
                "message": "Query cannot be empty."
            }

        context = chat_memory.get_context()
        if context:
            full_prompt = (
                "Here is our recent conversation history:\n"
                f"{context}\n\n"
                "Now, please answer this follow-up question "
                f"based on Indian Law: {query}"
            )
        else:
            full_prompt = query

        result = workflow.invoke({
            "messages": [
                HumanMessage(content=full_prompt)
            ]
        })

        final_answer = result["messages"][-1].content
        chat_memory.add(question=query, answer=final_answer)

        return {
            "answer": final_answer,
            "sources": [],
            "status": "success"
        }

    except Exception as e:
        return {
            "answer": "",
            "sources": [],
            "status": "error",
            "message": str(e)
        }


# ---------------------------------------------------------
# Clear Chat Memory Endpoint
# ---------------------------------------------------------
@app.post("/clear-memory")
def clear_memory():
    chat_memory.clear()
    return {"message": "Conversation history has been successfully cleared."}