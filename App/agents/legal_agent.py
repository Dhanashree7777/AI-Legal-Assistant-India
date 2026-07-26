import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the RAGService from your services folder
from services.rag_service import RAGService

load_dotenv()

# --- TOOL 1: Legal Document Retriever ---
@tool
def get_legal_document_data(query: str):
    """
    This is a RAG tool which retrieves information about Indian laws,
    specifically Constitutional, Civil, and updated Criminal laws (BNS, BNSS, BSA).
    """
    rag_service = RAGService()
    retriever = rag_service.get_retriever()
    response = retriever.invoke(query)
    return response

# --- THE AGENT ---
class IndianLegalAssistantAgent:
    def __init__(self):
        # Using Groq's supported Llama 3.3 model
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )   
        self.tools = [get_legal_document_data]
        self.tools_by_names = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def agent_node(self, state: dict) -> dict:
        messages = state["messages"]
        system_prompt = SystemMessage(content=(
            "You are a highly specialized AI Legal Assistant for Indian Laws.\n"
            "Your instructions are strict:\n"
            "1. ALWAYS use the get_legal_document_data tool to find statutory text.\n"
            "2. Do NOT hallucinate. If the retrieved text doesn't explicitly state the answer, do not guess.\n"
            "3. NEVER provide definitive legal advice; always append a disclaimer stating you do not replace professional counsel.\n"
            "4. You MUST compile and output a final comprehensive response based ONLY on retrieved data.\n"
            "Ensure the phrase INDIAN LEGAL ADVISORY REPORT is clearly printed at the start of your final response."
        ))
        
        full_message = [system_prompt] + messages
        response = self.llm_with_tools.invoke(full_message)
        return {"messages": [response]}

    def tool_node(self, state: dict) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        tool_outputs = []
        
        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                tool_function = self.tools_by_names.get(tool_name)
                if tool_function:
                    tool_result = tool_function.invoke(tool_args)
                else:
                    tool_result = f"Error: Tool {tool_name} not found"
                    
                tool_outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        return {"messages": tool_outputs}