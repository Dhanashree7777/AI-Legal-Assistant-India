import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage

load_dotenv()

@tool
def get_legal_document_data(query: str): # Function Object -> get_legal_document_data
    """
    This is a RAG tool which retrieves information about Indian laws,
    specifically Constitutional, Civil, and updated Criminal laws (BNS, BNSS, BSA).

    Arguments:
    query -> The section, legal concept, or question that will fetch particular statutory data.
    """
    # Self-contained mock response so you don't need the external services folder yet
    mock_response = f"[Mocked System Response]: Found baseline statutory information for query: '{query}'. Under the new Bharatiya Nyaya Sanhita (BNS), the offense of cheating, formerly Section 420 of the IPC, is now covered under Section 318."
    
    return mock_response

class IndianLegalAssistantAgent:
    def __init__(self):
        # Using the updated, currently supported Groq model
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.tools = [get_legal_document_data]
        self.tools_by_names = {t.name: t for t in self.tools} # {Tool Name: Tool's Function Object}

        self.llm_with_tools = llm.bind_tools(self.tools)

    def agent_node(self, state: dict) -> dict: # Thinking
        messages = state["messages"]

        system_prompt = SystemMessage(content=(
            "You are a highly specialized AI Legal Assistant for Indian Laws. Your instructions are strict:\n"
            "1. Use the `get_legal_document_data` tool to fetch statutory text, especially when mapping IPC to BNS.\n"
            "2. DO NOT hallucinate. If the retrieved text doesn't explicitly state the answer, do not guess.\n"
            "3. NEVER provide definitive legal advice; always append a disclaimer stating you do not replace professional counsel.\n"
            "4. You MUST compile and output a final `INDIAN LEGAL ADVISORY REPORT` based ONLY on the retrieved data.\n"
            "5. Get legal data tool argument; query -> The section or legal concept that will fetch particular statutory data.\n"
            "6. DO NOT GIVE GENERIC QUERY TO THE TOOL. The tool searches exact Bare Acts. Be very specific on what law you require.\n"
            "7. Example queries for the tool call: Query = 'What is the BNS section for cheating?', 'What does Article 14 of the Constitution state?', 'How does IPC 420 map to the new BNS?'\n"
            "Ensure the phrase `INDIAN LEGAL ADVISORY REPORT` is clearly printed at the start of your final report."
        ))   

        full_message = [system_prompt] + messages
        response = self.llm_with_tools.invoke(full_message)
        print("\n\n\n", response)
        return {"messages": [response]}    

    def tool_node(self, state: dict) -> dict: # Acting
        messages = state["messages"]
        last_message = messages[-1]
        print("\n\n\n", last_message)
        tool_outputs = []

        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                print("\n\n\n", tool_name, tool_args, tool_id)

                tool_function = self.tools_by_names.get(tool_name)

                if tool_function:
                    tool_result = tool_function.invoke(tool_args)
                else:
                    tool_result = f"Error: Tool `{tool_name}` not found"

                tool_outputs.append(
                    ToolMessage(content=(str(tool_result)), 
                                tool_call_id=tool_id,
                                name=tool_name)
                )

        print("\n\n\n", tool_outputs)
        return {"messages": tool_outputs}

if __name__ == "__main__":
    docs = get_legal_document_data.invoke('{"query":"What is the new BNS section for cheating instead of IPC 420?"}')
    print(docs)