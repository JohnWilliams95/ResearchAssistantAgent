import re
from pathlib import Path
from typing import Optional


class DocumentLoader:
    @staticmethod
    def load_text_file(filepath: Path) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 500, chunk_overlap: int = 100
    ) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) > chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para

        if current.strip():
            chunks.append(current.strip())

        merged = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_end = " ".join(chunks[i - 1].split()[-50:])
                chunk = prev_end + "\n" + chunk
            merged.append(chunk)

        return merged

    @staticmethod
    def load_and_index_directory(
        directory: Path, vector_store, pattern: str = "*.txt"
    ) -> int:
        count = 0
        for filepath in directory.glob(pattern):
            text = DocumentLoader.load_text_file(filepath)
            chunks = DocumentLoader.chunk_text(text)
            ids = vector_store.add_documents(
                contents=chunks,
                metadata_list=[{"source": str(filepath)} for _ in chunks],
            )
            count += len(ids)
        return count