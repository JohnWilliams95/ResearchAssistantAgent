import asyncio
import json
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base import BaseAgent
from state.workflow_state import WorkflowState, Document
from tools.web_search_tool import WebSearchTool
from tools.arxiv_tool import ArXivMCPSearch
from config.settings import settings


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.1)

    @property
    def name(self) -> str:
        return "research_agent"

    @property
    def description(self) -> str:
        return "通过网络搜索(Tavily)和学术论文检索(ArXiv MCP)获取外部信息，收集回答用户问题所需的参考资料"

    @property
    def system_prompt(self) -> str:
        return """你是一个信息检索专家。你的任务分为两步：

第一步：查询改写。将用户的自然语言问题改写为更适合搜索引擎和学术论文检索的查询词。
- web_query: 用于 Web 搜索的关键词（简洁、精准，提取核心概念）
- arxiv_query: 用于 ArXiv 论文检索的查询（可以用英文，包含关键术语和学术词汇）

第二步：结果筛选。从搜索结果中筛选出与用户问题最相关的信息，去除无关或低质量内容。
保留最有价值的信息片段，每个结果保留核心摘要（不超过200字）。

请以 JSON 格式输出：
{"web_query": "...", "arxiv_query": "..."}

后续你会收到搜索结果，届时请以 JSON 格式输出筛选结果：
{"filtered_web": [{"title": "...", "content": "...", "url": "..."}], "filtered_arxiv": [{"title": "...", "content": "..."}]}"""

    def __call__(self, state: WorkflowState) -> dict:
        query = state["query"]

        rewrite_response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"用户问题：{query}\n\n请先输出查询改写的 JSON。"),
        ])
        rewritten = self._parse_json(rewrite_response.content)
        web_query = rewritten.get("web_query", query)
        arxiv_query = rewritten.get("arxiv_query", query)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        web_raw, arxiv_raw = loop.run_until_complete(
            self._research_async(
                web_query,
                arxiv_query,
                settings.MAX_SEARCH_RESULTS,
                settings.MAX_ARXIV_RESULTS,
            )
        )

        search_docs = self._format_web_to_documents(web_raw)
        arxiv_docs = self._parse_arxiv_result(arxiv_raw, settings.MAX_ARXIV_RESULTS)

        filter_response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_filter_prompt(query, search_docs, arxiv_docs)),
        ])
        filtered = self._parse_json(filter_response.content)

        filtered_web = filtered.get("filtered_web", [])
        filtered_arxiv = filtered.get("filtered_arxiv", [])

        final_web = [
            Document(
                content=r.get("content", ""),
                source="web",
                title=r.get("title", ""),
                url=r.get("url", ""),
            )
            for r in filtered_web
        ] if filtered_web else search_docs

        final_arxiv = [
            Document(
                content=r.get("content", ""),
                source="arxiv",
                title=r.get("title", ""),
                url=r.get("url", ""),
            )
            for r in filtered_arxiv
        ] if filtered_arxiv else arxiv_docs

        return {"search_results": final_web, "arxiv_results": final_arxiv}

    def _build_filter_prompt(self, query: str, search_docs: list[Document], arxiv_docs: list[Document]) -> str:
        web_items = []
        for i, doc in enumerate(search_docs, 1):
            web_items.append(f"[Web {i}] 标题: {doc.get('title', 'N/A')}\n内容: {doc.get('content', '')}\n链接: {doc.get('url', '')}")

        arxiv_items = []
        for i, doc in enumerate(arxiv_docs, 1):
            arxiv_items.append(f"[ArXiv {i}] 标题: {doc.get('title', 'N/A')}\n摘要: {doc.get('content', '')}")

        web_text = "\n\n".join(web_items) if web_items else "（无 Web 搜索结果）"
        arxiv_text = "\n\n".join(arxiv_items) if arxiv_items else "（无 ArXiv 论文结果）"

        return f"""用户问题：{query}

以下是搜索到的原始结果：

=== Web 搜索结果 ===
{web_text}

=== ArXiv 论文结果 ===
{arxiv_text}

请从以上结果中筛选出与用户问题最相关的信息，输出筛选后的 JSON。"""

    async def _research_async(self, web_query: str, arxiv_query: str, max_web: int, max_arxiv: int):
        loop = asyncio.get_event_loop()

        def sync_web():
            ws = WebSearchTool()
            return ws.search(query=web_query, max_results=max_web)

        async def async_arxiv():
            searcher = ArXivMCPSearch()
            try:
                return await searcher.search(query=arxiv_query, max_results=max_arxiv)
            finally:
                await searcher.close()

        web_results = await loop.run_in_executor(None, sync_web)
        arxiv_results = await async_arxiv()
        return web_results, arxiv_results

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
    def _format_web_to_documents(results: list[dict]) -> list[Document]:
        return [
            Document(
                content=r.get("content", ""),
                source="web",
                title=r.get("title", ""),
                url=r.get("url", ""),
            )
            for r in results
        ]

    @staticmethod
    def _parse_arxiv_result(raw: str, max_results: int) -> list[Document]:
        docs = []
        if not raw or raw.startswith("[ArXiv Error]") or raw.startswith("[Error]"):
            return docs
        lines = raw.split("\n")
        current = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                if current:
                    docs.append(Document(
                        content=current.get("summary", ""),
                        source="arxiv",
                        title=current.get("title", ""),
                        url=current.get("url", ""),
                    ))
                current = {"title": line.replace("Title:", "").strip()}
            elif line.startswith("Authors:"):
                current["authors"] = line.replace("Authors:", "").strip()
            elif line.startswith("URL:") or line.startswith("Link:"):
                current["url"] = line.replace("URL:", "").replace("Link:", "").strip()
            elif line.startswith("Summary:") or line.startswith("Abstract:"):
                current["summary"] = line.replace("Summary:", "").replace("Abstract:", "").strip()
            elif current.get("summary"):
                current["summary"] += " " + line
            else:
                current["summary"] = line
        if current:
            docs.append(Document(
                content=current.get("summary", ""),
                source="arxiv",
                title=current.get("title", ""),
                url=current.get("url", ""),
            ))
        return docs[:max_results]