import json
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base import BaseAgent
from state.workflow_state import WorkflowState, Document
from tools.web_search_tool import web_search
from tools.arxiv_tool import arxiv_search
from config.settings import settings


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.1)
        self.tools = [web_search, arxiv_search]

    @property
    def name(self) -> str:
        return "research_agent"

    @property
    def description(self) -> str:
        return "通过网络搜索(Tavily)和学术论文检索(ArXiv MCP)获取外部信息，收集回答用户问题所需的参考资料"

    @property
    def system_prompt(self) -> str:
        return """你是一个信息检索专家。你可以使用以下工具：

- web_search: 搜索网页获取实时信息。输入搜索关键词，返回相关网页结果（标题、链接、摘要）。
- arxiv_search: 搜索 ArXiv 学术论文。输入查询关键词，返回相关论文（标题、作者、摘要、链接）。

根据用户问题，分析是否需要搜索。如果需要，调用相应工具获取信息。
如果用户问题是对比类或学术类问题，建议同时调用两个工具。
如果用户问题是通用问题，根据需要选择性调用。"""

    def __call__(self, state: WorkflowState) -> dict:
        query = state["query"]

        llm_with_tools = self.llm.bind_tools(self.tools)

        response = llm_with_tools.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"用户问题：{query}"),
        ])

        web_results = []
        arxiv_results = []

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "web_search":
                    result = web_search.invoke(tool_args)
                    web_results = json.loads(result) if isinstance(result, str) else result
                elif tool_name == "arxiv_search":
                    result = arxiv_search.invoke(tool_args)
                    arxiv_results = json.loads(result) if isinstance(result, str) else result

        search_docs = self._format_web_to_documents(web_results)
        arxiv_docs = self._parse_arxiv_result(arxiv_results, settings.MAX_ARXIV_RESULTS)

        filter_response = self.llm.invoke([
            SystemMessage(content="你是一个信息检索专家。根据搜索结果筛选出与用户问题最相关的信息。"),
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