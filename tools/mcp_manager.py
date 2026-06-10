import asyncio
import json
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPSessionManager:
    def __init__(self):
        self._session: Optional[ClientSession] = None
        self._tools: list[dict] = []

    async def connect(
        self, command: str, args: list[str]
    ) -> Optional[ClientSession]:
        server_params = StdioServerParameters(command=command, args=args)
        try:
            self._read, self._write = await asyncio.wait_for(
                stdio_client(server_params).__aenter__(), timeout=30.0
            )
            self._session = await asyncio.wait_for(
                ClientSession(self._read, self._write).__aenter__(), timeout=30.0
            )
            await asyncio.wait_for(self._session.initialize(), timeout=30.0)

            tools_result = await asyncio.wait_for(
                self._session.list_tools(), timeout=30.0
            )
            self._tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in tools_result.tools
            ]
            return self._session
        except asyncio.TimeoutError:
            raise RuntimeError(f"MCP server connection timed out: {command} {' '.join(args)}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}")

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            raise RuntimeError("MCP session not initialized")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments), timeout=60.0
            )
            if result.content:
                texts = []
                for c in result.content:
                    if hasattr(c, "text"):
                        texts.append(c.text)
                return "\n".join(texts)
            return json.dumps(result.content, ensure_ascii=False)
        except asyncio.TimeoutError:
            return f"[Error] Tool call timed out: {tool_name}"
        except Exception as e:
            return f"[Error] Tool call failed: {tool_name} - {e}"

    def get_tools(self) -> list[dict]:
        return self._tools

    async def close(self):
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()