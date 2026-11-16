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
        @tool
        def get_legal_document_data(query: str):
            """
            Retrieve relevant information from Indian legal documents including
            Constitution of India, BNS, BNSS, BSA and the IT Act.
            """
            try:
                retriever = rag_service.get_retriever()
                docs = retriever.invoke(query)
                return docs
            except Exception as e:
                return f"Retrieval Error: {e}"
        retriever = rag_service.get_retriever()
        docs = retriever.invoke(query)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in docs
        ]
    except Exception as e:
        return f"Retrieval Error: {e}"


class IndianLegalAssistantAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
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

# Add this at the very bottom of legal_agent.py
if __name__ == "__main__":
    print("Testing Indian Legal Assistant Agent...")
    
    # 1. Initialize the agent
    agent = IndianLegalAssistantAgent()
    print("✅ Agent initialized successfully with Groq!")
    
    # 2. Check if the tool is bound
    print(f"✅ Tools loaded: {list(agent.tools_by_names.keys())}")
    
    # 3. Create a fake LangGraph state to test the agent node
    from langchain_core.messages import HumanMessage
    test_state = {"messages": [HumanMessage(content="What is Article 21?")]}
    
    print("\n⏳ Asking the agent: 'What is Article 21?'...")
    response = agent.agent_node(test_state)
    
    # 4. Print what the AI decided to do (it should decide to call a tool or answer)
    ai_message = response["messages"][0]
    if ai_message.tool_calls:
        print(f"✅ The AI decided to use the RAG tool to search for: {ai_message.tool_calls[0]['args']['query']}")
    else:
        print(f"✅ AI Response: {ai_message.content}")