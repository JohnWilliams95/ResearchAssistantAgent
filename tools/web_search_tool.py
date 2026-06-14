import json
from langchain_core.tools import tool
from tavily import TavilyClient
from config.settings import settings


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """搜索网页获取实时信息。输入搜索关键词，返回相关网页结果（标题、链接、摘要）。

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数，默认5
    """
    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            })
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as e:
        return json.dumps([{"title": "Search Error", "url": "", "content": str(e), "score": 0.0}])