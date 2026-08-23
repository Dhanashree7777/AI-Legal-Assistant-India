# AI-Legal-Assistant-India

# AI Legal Assistant for Indian Laws

A Retrieval-Augmented Generation (RAG) system designed to answer queries on Indian Constitutional Law, the IT Act, and the updated Criminal laws (BNS, BNSS, BSA). Built with LangChain, LangGraph, and a local FAISS vector database to ensure accurate, hallucination-free legal responses.

## Prerequisites

* **Python 3.11 or newer** (Virtual environment like `myenv` recommended).
* **API Keys**: 
  * **Groq API Key** (for fast reasoning via Llama 3 models).
  * **Google Gemini API Key** (for generating document embeddings).

## Backend Setup

1. **Activate your virtual environment:**
   * Windows: `myenv\Scripts\activate`.
   * Mac/Linux: `source myenv/bin/activate`.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. Configure Environment Variables:
    Create a .env file in the root directory and add your keys. Ensure .env is listed in your .gitignore
    ```GROQ_API_KEY=your_groq_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
4. Start the FastAPI Backend:
    ```uvicorn App.main:app --reload

## Frontend Setup

1. Open a second terminal window and activate your virtual environment.

2. Launch the Streamlit App:
    ```streamlit run app.py
    The UI will open at `http://localhost:8501`.

## Running Tests
1. Build/Test Vector Database: Ingests PDFs from the /data  folder into FAISS
    ```python App/services/rag_service.py
2.Test Agent Connection: Verifies LLM API connections and system prompt
    ```python App/agents/legal_agent.py
3. Test LangGraph Workflow: Validates routing between the agent and RAG tools
    ```python App/workflows/legal_workflow.py

## Useful Commands
1. Restart FastAPI after code changes during development: `uvicorn main:app --reload`
2. Clear Chat Memory: Click "Clear Chat History" in the Streamlit sidebar.
3. Git Version Control:
    ```git add .
    git commit -m "Update project files"
    git push origin main
    git pull origin main 
    ```
4. Export dependencies: 
    ```pip freeze > requirements.txt