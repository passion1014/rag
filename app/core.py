from typing import List, Tuple, TypedDict
from langserve.pydantic_v1 import BaseModel, Field

class AgentState(TypedDict):
    chat_history: List[Tuple[str, str]]
    question: str
    context: str
    response: str
    
# class ChatHistory(BaseModel):
#     chat_history: List[Tuple[str, str]] = Field(
#         ...,
#         extra={"widget": {"type": "chat", "input": "question"}},
#     )
#     question: str
