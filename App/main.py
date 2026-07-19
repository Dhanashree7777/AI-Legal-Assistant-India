import os
from langchain_core.messages import HumanMessage
from workflows.legal_workflow import workflow

def main():
    """
    Main execution loop for the AI Legal Assistant.
    Provides an interactive terminal chat interface.
    """
    print("=" * 70)
    print("AI Legal Assistant for Indian Laws")
    print("Supports: Constitution of India, BNS, BNSS, BSA.")
    print("Type 'quit' or 'exit' to close the application.")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['quit', 'exit']:
            print("\nShutting down the Legal Assistant. Goodbye!")
            break

        if not user_input.strip():
            continue

        print("Analyzing legal documents and reasoning...")

        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        try:
            output = workflow.invoke(initial_state)
            final_response = output["messages"][-1].content
            print(f"\nAI Assistant:\n\n{final_response}")

        except Exception as e:
            print(f"\nAn error occurred during processing: {e}")

if __name__ == "__main__":
    main()