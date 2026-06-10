from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    _instance: SentenceTransformer | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._instance is None:
            cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._instance

    @classmethod
    def embed(cls, texts: list[str]) -> list[list[float]]:
        model = cls.get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        return cls.embed([query])[0]