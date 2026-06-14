import asyncio
import json
from langchain_core.tools import tool
from tools.mcp_manager import MCPSessionManager
from config.settings import settings


@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """搜索ArXiv学术论文。输入查询关键词，返回相关论文（标题、作者、摘要、链接）。

    Args:
        query: 搜索关键词（建议使用英文）
        max_results: 最大返回论文数，默认5
    """
    try:
        async def _search():
            manager = MCPSessionManager()
            try:
                await manager.connect(
                    command=settings.ARXIV_MCP_COMMAND,
                    args=settings.ARXIV_MCP_ARGS.split(),
                )
                result = await manager.call_tool(
                    "search_papers",
                    {"query": query, "max_results": max_results},
                )
                return result
            finally:
                await manager.close()

        return asyncio.run(_search())
    except Exception as e:
        return json.dumps([{"title": "ArXiv Error", "content": str(e)}])