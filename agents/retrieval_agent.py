import json
from langchain_core.messages import HumanMessage, SystemMessage
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
        return """你是一个知识检索专家。你的任务分为两步：

第一步：查询改写。将用户的自然语言问题改写为更适合向量语义检索的查询。
向量检索偏好：名词短语、关键术语、概念词汇，而非完整的问句。
输出 JSON：{"retrieval_query": "改写后的检索查询"}

第二步：结果评估。评估检索到的知识片段与用户问题的相关性，过滤掉不相关的内容。
输出 JSON：{"evaluated_results": [{"content": "...", "score": 0.95, "reason": "相关原因"}]}

知识库中预置了 LangGraph、AutoGen、MCP、Multi-Agent System、RAG、ChromaDB 等领域知识。
如果知识库为空，会自动初始化预置知识。"""

    def __call__(self, state: WorkflowState) -> dict:
        query = state["query"]

        rewrite_response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"用户问题：{query}\n\n请先输出查询改写的 JSON（第一步）。"),
        ])
        rewritten = self._parse_json(rewrite_response.content)
        retrieval_query = rewritten.get("retrieval_query", query)

        store = VectorStore()
        if store.count() == 0:
            self._seed_knowledge_base(store)

        results = store.search(retrieval_query, n_results=settings.MAX_RETRIEVAL_RESULTS)
        docs = [
            Document(
                content=r.get("content", ""),
                source=f"chroma:{r.get('metadata', {}).get('source', 'unknown')}",
                title=f"KB Doc (score: {r.get('score', 0):.3f})",
            )
            for r in results
        ]

        evaluate_response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_evaluate_prompt(query, docs)),
        ])
        evaluated = self._parse_json(evaluate_response.content)
        evaluated_results = evaluated.get("evaluated_results", [])

        if evaluated_results:
            final_docs = [
                Document(
                    content=r.get("content", ""),
                    source="chroma:evaluated",
                    title=f"KB Doc (relevance: {r.get('score', 'N/A')})",
                )
                for r in evaluated_results
            ]
        else:
            final_docs = docs

        return {"retrieval_results": final_docs}

    def _build_evaluate_prompt(self, query: str, docs: list[Document]) -> str:
        items = []
        for i, doc in enumerate(docs, 1):
            items.append(f"[片段 {i}] {doc.get('content', '')}")

        items_text = "\n\n".join(items) if items else "（无检索结果）"

        return f"""用户问题：{query}

以下是向量检索返回的知识片段：

{items_text}

请评估每个片段与用户问题的相关性，过滤掉不相关的内容，输出评估后的 JSON（第二步）。"""

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

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