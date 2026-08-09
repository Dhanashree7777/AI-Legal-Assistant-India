from collections import deque
from typing import List, Dict

class ChatMemory:
    """
    Stores recent conversation history for the AI Legal Assistant.
    """

    def __init__(self, max_history: int = 10):
        self.history = deque(maxlen=max_history)

    def add(self, question: str, answer: str) -> None:
        self.history.append({
            "question": question,
            "answer": answer
        })

    def get_context(self) -> str:
        if not self.history:
            return ""

        context = []
        for chat in self.history:
            context.append(f"User: {chat['question']}")
            context.append(f"Assistant: {chat['answer']}")
        return "\n".join(context)

    def clear(self) -> None:
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

memory = ChatMemory()

