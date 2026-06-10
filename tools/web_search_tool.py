from typing import Optional
from tavily import TavilyClient
from config.settings import settings


class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self._client = TavilyClient(api_key=api_key or settings.TAVILY_API_KEY)

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_raw_content: bool = False,
    ) -> list[dict]:
        if max_results is None:
            max_results = settings.MAX_SEARCH_RESULTS
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                include_raw_content=include_raw_content,
            )
            results = response.get("results", [])
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                })
            return formatted
        except Exception as e:
            return [{"title": "Search Error", "url": "", "content": str(e), "score": 0.0}]