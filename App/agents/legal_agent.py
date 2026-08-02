import sys
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.rag_service import RAGService
from prompts.legal_prompt import LEGAL_SYSTEM_PROMPT

load_dotenv()


@tool
def get_legal_document_data(query: str):
    """
    Retrieve relevant information from Indian legal documents including
    Constitution of India, BNS, BNSS, BSA and the IT Act.
    """
    try:
        rag_service = RAGService()
        retriever = rag_service.get_retriever()
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        return f"Retrieval Error: {e}"


class IndianLegalAssistantAgent:
    def __init__(self):
        # Using Groq's Llama 3.3 model [cite: 2, 19]
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
        system_prompt = SystemMessage(content=LEGAL_SYSTEM_PROMPT)
        response = self.llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def tool_node(self, state: dict) -> dict:
        outputs = []
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls"):
            for call in last_message.tool_calls:
                name = call["name"]
                args = call["args"]
                tool_id = call["id"]

                try:
                    tool = self.tools_by_names[name]
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"Tool execution failed: {e}"

                outputs.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=name
                    )
                )

        return {"messages": outputs}