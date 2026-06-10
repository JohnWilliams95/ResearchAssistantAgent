from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class Document(TypedDict, total=False):
    content: str
    source: str
    title: Optional[str]
    url: Optional[str]


class WorkflowState(TypedDict, total=False):
    query: str
    intent: str
    messages: Annotated[Sequence[BaseMessage], lambda left, right: list(left) + list(right)]

    search_results: list[Document]
    arxiv_results: list[Document]
    retrieval_results: list[Document]

    skill_results: list[str]

    plan: str
    partial_answers: list[str]
    final_answer: str

    error: Optional[str]