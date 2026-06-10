import asyncio
from tools.mcp_manager import MCPSessionManager
from config.settings import settings


class ArXivMCPSearch:
    def __init__(self):
        self._manager: MCPSessionManager | None = None

    async def _ensure_connected(self):
        if self._manager is None:
            self._manager = MCPSessionManager()
            await self._manager.connect(
                command=settings.ARXIV_MCP_COMMAND,
                args=settings.ARXIV_MCP_ARGS.split(),
            )
        return self._manager

    async def search(self, query: str, max_results: int | None = None) -> str:
        if max_results is None:
            max_results = settings.MAX_ARXIV_RESULTS
        try:
            manager = await self._ensure_connected()
            result = await manager.call_tool(
                "search_papers",
                {"query": query, "max_results": max_results},
            )
            return result
        except Exception as e:
            return f"[ArXiv Error] {e}"

    async def close(self):
        if self._manager:
            await self._manager.close()
            self._manager = None


def arxiv_sync_search(query: str, max_results: int | None = None) -> str:
    searcher = ArXivMCPSearch()
    try:
        return asyncio.run(searcher.search(query, max_results))
    finally:
        asyncio.run(searcher.close())