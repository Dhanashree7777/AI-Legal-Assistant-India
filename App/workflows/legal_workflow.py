import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated, Literal

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from agents.legal_agent import IndianLegalAssistantAgent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    legal_report: str


def route_legal_assistant(state: AgentState) -> Literal["legal_tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM wants to use a tool, go to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "legal_tools"

    # Otherwise, save the final report and end
    state["legal_report"] = last_message.content
    return "__end__"


legal_agent = IndianLegalAssistantAgent()

# Workflow data will be stored and passed around AgentState
workflow_builder = StateGraph(AgentState)

# Nodes
workflow_builder.add_node("legal_assistant", legal_agent.agent_node)
workflow_builder.add_node("legal_tools", legal_agent.tool_node)

# Start → Agent
workflow_builder.add_edge(START, "legal_assistant")

# Agent → Tool OR End
workflow_builder.add_conditional_edges(
    "legal_assistant",
    route_legal_assistant,
    {
        "legal_tools": "legal_tools",
        "__end__": END,
    },
)

# Tool → Agent
workflow_builder.add_edge("legal_tools", "legal_assistant")

# Compile workflow
workflow = workflow_builder.compile()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    input = {
        "messages": [
            HumanMessage(
                content="What is the new BNS section corresponding to IPC Section 420?"
            )
        ]
    }

    print("\n\n--------------------------OUTPUT STARTS HERE--------------\n\n")

    output = workflow.invoke(input)

    print(output["messages"][-1].content)

    print("\n\n--------------------------OUTPUT ENDS HERE-----------------\n\n")

    print("\n\n-------------------------START DEBUGGING HERE---------------\n\n")

    print(output)

    print("\n\n-------------------------END DEBUGGING HERE-----------------\n\n")

# result = workflow.invoke(
#     {
#         "messages": [
#             HumanMessage(content=full_prompt)
#         ]
#     },
#     config={
#         "recursion_limit": 10
#     }
# )