from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from App.workflows.legal_workflow import workflow
# Add this to your imports at the top of main.py
from App.services.chat_memory import ChatMemory

# Initialize memory globally (place this somewhere near the top, before your endpoints)
chat_memory = ChatMemory()

app = FastAPI(
    title="AI Legal Assistant API", 
    version="0.1.0",
    description="API for AI Legal Assistant for Indian Laws"
)

@app.get("/")
def root() -> dict:
    return {"message": "The AI Legal Assistant API is live......"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "OK"}


# ------------------------------------------------------------------
# Post Method -> Create a new data
# ------------------------------------------------------------------

data = set()

@app.post("/add-item")
def add_new_item(item: str) -> dict:
    data.add(item)
    return {"message": f"{item} successfully added to the database"}


# ------------------------------------------------------------------
# Get Method -> Read data from the database
# ------------------------------------------------------------------

@app.get("/display-all-items")
def display_all_items() -> dict:
    print(f"The data is : {data}")
    return {"message": f"Following items are in the database: {data}"}


@app.get("/verify-item/{item_name}")
def verify_item(item_name: str) -> dict:
    if item_name in data:
        return {"message": f"{item_name} is present in the database"}
    return {"message": f"{item_name} is not present in the database"}


# ------------------------------------------------------------------
# Put Method -> Update data in the database
# ------------------------------------------------------------------

@app.put("/update_item")
def update_item(existing_item: str, new_item: str) -> dict:
    if existing_item not in data:
        return {"message": f"{existing_item} is not present in the database"}
    data.remove(existing_item)
    data.add(new_item)
    return {"message": f"{existing_item} has been updated to {new_item}"}


# ------------------------------------------------------------------
# Delete Method -> Delete data in the database
# ------------------------------------------------------------------

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


@app.post("/execute-workflow")
def process_workflow(query: str):
    context = chat_memory.get_context()
    
    if context:
        full_prompt = f"Here is our recent conversation history:\n{context}\n\nNow, please answer this follow-up question based on Indian Law: {query}"
    else:
        full_prompt = query

    result = workflow.invoke({"messages": [HumanMessage(content=full_prompt)]})
    
    final_answer = result["messages"][-1].content
    
    chat_memory.add(question=query, answer=final_answer)
    return final_answer

@app.post("/clear-memory")
def clear_memory():
    chat_memory.clear()
    return {"message": "Conversation history has been successfully cleared."}