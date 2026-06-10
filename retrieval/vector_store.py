import uuid
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from retrieval.embedder import Embedder


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        contents: list[str],
        metadata_list: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]
        if metadata_list is None:
            metadata_list = [{} for _ in contents]

        embeddings = Embedder.embed(contents)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadata_list,
        )
        return ids

    def search(
        self, query: str, n_results: int | None = None
    ) -> list[dict]:
        if n_results is None:
            n_results = settings.MAX_RETRIEVAL_RESULTS

        query_embedding = Embedder.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
        )

        formatted = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            formatted.append({
                "id": ids_list[i] if i < len(ids_list) else "",
                "content": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas_list[i] if i < len(metas_list) else {},
                "score": 1 - distances_list[i] if i < len(distances_list) else 0.0,
            })
        return formatted

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)