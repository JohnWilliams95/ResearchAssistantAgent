import asyncio
from langchain_core.messages import HumanMessage
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
        return """你是一个信息检索专家。你会同时：
1. 使用网络搜索引擎(Tavily)搜索最新网页信息
2. 通过 ArXiv MCP Server 检索相关学术论文

你需要确保搜索结果与用户问题高度相关，并保留最有价值的信息片段。"""

    def __call__(self, state: WorkflowState) -> dict:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        web_raw, arxiv_raw = loop.run_until_complete(
            self._research_async(
                state["query"],
                settings.MAX_SEARCH_RESULTS,
                settings.MAX_ARXIV_RESULTS,
            )
        )

        search_docs = self._format_web_to_documents(web_raw)
        arxiv_docs = self._parse_arxiv_result(arxiv_raw, settings.MAX_ARXIV_RESULTS)

        return {"search_results": search_docs, "arxiv_results": arxiv_docs}

    async def _research_async(self, query: str, max_web: int, max_arxiv: int):
        loop = asyncio.get_event_loop()

        def sync_web():
            ws = WebSearchTool()
            return ws.search(query=query, max_results=max_web)

        async def async_arxiv():
            searcher = ArXivMCPSearch()
            try:
                return await searcher.search(query=query, max_results=max_arxiv)
            finally:
                await searcher.close()

        web_results = await loop.run_in_executor(None, sync_web)
        arxiv_results = await async_arxiv()
        return web_results, arxiv_results

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