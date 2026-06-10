from agents.base import BaseAgent
from state.workflow_state import WorkflowState, Document
from retrieval.vector_store import VectorStore
from config.settings import settings


class RetrievalAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.0)

    @property
    def name(self) -> str:
        return "retrieval_agent"

    @property
    def description(self) -> str:
        return "从本地 ChromaDB 知识库中检索与用户问题最相关的文档片段"

    @property
    def system_prompt(self) -> str:
        return """你是一个知识检索专家。你管理一个向量知识库（ChromaDB + sentence-transformers），
能够根据用户问题进行语义检索，返回最相关的知识片段。

知识库中预置了 LangGraph、AutoGen、MCP、Multi-Agent System、RAG、ChromaDB 等领域知识。
如果知识库为空，会自动初始化预置知识。"""

    def __call__(self, state: WorkflowState) -> dict:
        store = VectorStore()
        if store.count() == 0:
            self._seed_knowledge_base(store)

        results = store.search(state["query"], n_results=settings.MAX_RETRIEVAL_RESULTS)
        docs = [
            Document(
                content=r.get("content", ""),
                source=f"chroma:{r.get('metadata', {}).get('source', 'unknown')}",
                title=f"KB Doc (score: {r.get('score', 0):.3f})",
            )
            for r in results
        ]
        return {"retrieval_results": docs}

    @staticmethod
    def _seed_knowledge_base(store: VectorStore):
        docs = [
            "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It allows developers to create complex agent workflows with cycles and state management. Key features include: stateful graphs, checkpointing, streaming, and human-in-the-loop support.",
            "AutoGen is a framework developed by Microsoft for building multi-agent conversations. It supports customizable agents, group chats, and integration with various tools and APIs. AutoGen focuses on agent communication patterns and task delegation.",
            "LangGraph vs AutoGen: LangGraph emphasizes fine-grained control over agent workflows with explicit state management, while AutoGen emphasizes high-level agent conversation patterns. LangGraph is more of a low-level orchestration framework, and AutoGen is more of an agent communication framework.",
            "Model Context Protocol (MCP) is an open protocol by Anthropic that standardizes how applications provide context to LLMs. It uses a client-server architecture where MCP servers expose resources, tools, and prompts through a standardized JSON-RPC interface. Key advantages: tool interoperability, dynamic discovery, and separation of concerns between tool providers and consumers.",
            "A multi-agent system (MAS) is a distributed system composed of multiple interacting intelligent agents. In the context of LLMs, each agent can have specialized roles, access different tools, and collaborate to solve complex tasks. LangGraph and AutoGen are both frameworks for building such systems.",
            "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and feeds them as context to an LLM to produce more accurate and grounded responses.",
            "ChromaDB is an open-source embedding database optimized for AI applications. It stores vector embeddings and supports similarity search, making it ideal for semantic search and RAG applications.",
        ]
        for i, doc in enumerate(docs):
            store.add_documents(
                contents=[doc],
                metadata_list=[{"source": "builtin_knowledge"}],
                ids=[f"seed_{i}"],
            )