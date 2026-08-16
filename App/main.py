from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from App.workflows.legal_workflow import workflow
from App.services.chat_memory import ChatMemory


# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------------
# Initialize Chat Memory
# ---------------------------------------------------------

chat_memory = ChatMemory()


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="AI Legal Assistant API",
    version="0.1.0",
    description="API for AI Legal Assistant for Indian Laws"
)


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "message": "The AI Legal Assistant API is live......"
    }


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    return {
        "status": "OK"
    }


# ---------------------------------------------------------
# Simple Database / CRUD Section
# ---------------------------------------------------------

data = set()


@app.post("/add-item")
def add_new_item(item: str) -> dict:
    data.add(item)

    return {
        "message": f"{item} successfully added to the database"
    }


@app.get("/display-all-items")
def display_all_items() -> dict:
    print(f"The data is : {data}")

    return {
        "message": f"Following items are in the database: {data}"
    }


@app.get("/verify-item/{item_name}")
def verify_item(item_name: str) -> dict:

    if item_name in data:
        return {
            "message": f"{item_name} is present in the database"
        }

    return {
        "message": f"{item_name} is not present in the database"
    }


@app.put("/update_item")
def update_item(
    existing_item: str,
    new_item: str
) -> dict:

    if existing_item not in data:
        return {
            "message": f"{existing_item} is not present in the database"
        }

    data.remove(existing_item)
    data.add(new_item)

    return {
        "message": f"{existing_item} has been updated to {new_item}"
    }


@app.delete("/delete-item")
def delete_item(item: str) -> dict:

    if item not in data:
        return {
            "message": f"{item} is not present in the database"
        }

    data.remove(item)

    return {
        "message": f"{item} has been deleted from the database"
    }


@app.delete("/delete-all-items")
def delete_all_items() -> dict:

    data.clear()

    return {
        "message": "All items have been deleted from the database"
    }


# ---------------------------------------------------------
# AI LEGAL WORKFLOW
# ---------------------------------------------------------

@app.post("/execute-workflow")
def process_workflow(request: QueryRequest):

    try:

        # Get and clean user's question
        query = request.query.strip()

        if not query:
            return {
                "answer": "",
                "sources": [],
                "status": "error",
                "message": "Query cannot be empty."
            }

        # -------------------------------------------------
        # Get previous conversation
        # -------------------------------------------------

        context = chat_memory.get_context()

        # -------------------------------------------------
        # Build context-aware prompt
        # -------------------------------------------------

        if context:

            full_prompt = (
                "Here is our recent conversation history:\n"
                f"{context}\n\n"
                "Now, please answer this follow-up question "
                f"based on Indian Law: {query}"
            )

        else:
            full_prompt = query

        # -------------------------------------------------
        # Execute LangGraph / Legal Workflow
        # -------------------------------------------------

        result = workflow.invoke({
            "messages": [
                HumanMessage(content=full_prompt)
            ]
        })

        # -------------------------------------------------
        # Extract final response
        # -------------------------------------------------

        final_answer = result["messages"][-1].content

        # -------------------------------------------------
        # Save conversation to memory
        # -------------------------------------------------

        chat_memory.add(
            question=query,
            answer=final_answer
        )

        # -------------------------------------------------
        # Return standardized API response
        # -------------------------------------------------

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
# CLEAR CHAT MEMORY
# ---------------------------------------------------------

@app.post("/clear-memory")
def clear_memory():

    chat_memory.clear()

    return {
        "message": "Conversation history has been successfully cleared."
    }