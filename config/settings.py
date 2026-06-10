import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=False)


class Settings:
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_MODEL_PROVIDER: str = os.getenv("LLM_MODEL_PROVIDER", "openai")

    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db")
    )
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "research_docs")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    ARXIV_MCP_COMMAND: str = os.getenv("ARXIV_MCP_COMMAND", "uvx")
    ARXIV_MCP_ARGS: str = os.getenv("ARXIV_MCP_ARGS", "arxiv-mcp-server")

    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    MAX_ARXIV_RESULTS: int = int(os.getenv("MAX_ARXIV_RESULTS", "5"))
    MAX_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_RETRIEVAL_RESULTS", "3"))


settings = Settings()