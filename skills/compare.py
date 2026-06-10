from langchain.chat_models import init_chat_model
from config.settings import settings
from skills.base import BaseSkill


class CompareSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "compare"

    @property
    def description(self) -> str:
        return "对比分析两个或多个事物的异同、优劣"

    def execute(self, context: dict) -> str:
        query = context.get("query", "")
        search_results = context.get("search_results", [])
        arxiv_results = context.get("arxiv_results", [])

        context_text = self._build_context(search_results, arxiv_results)
        llm = init_chat_model(
            settings.LLM_MODEL,
            model_provider=settings.LLM_MODEL_PROVIDER,
            api_key=settings.LLM_API_KEY or None,
            temperature=settings.LLM_TEMPERATURE,
        )

        prompt = f"""你是一个专业的技术对比分析师。请基于以下参考资料，对用户的问题进行全面、结构化的对比分析。

用户问题：{query}

参考资料：
{context_text}

请按以下格式输出对比结果：
1. **概述**：简要介绍要对比的各方
2. **维度对比**：从多个维度（功能、性能、易用性、生态等）逐一对比
3. **优劣势总结**：列出各方的核心优势和劣势
4. **适用场景建议**：给出不同场景下的选型建议

如果参考资料不足，请基于你的知识进行补充，但需要标注哪些来自参考资料，哪些来自你的知识。"""

        response = llm.invoke(prompt)
        return response.content

    @staticmethod
    def _build_context(search_results: list, arxiv_results: list) -> str:
        parts = []
        for r in search_results:
            title = r.get("title", "")
            content = r.get("content", "")
            if title or content:
                parts.append(f"[Web] {title}\n{content}")
        for r in arxiv_results:
            title = r.get("title", "")
            content = r.get("content", "")
            if title or content:
                parts.append(f"[ArXiv] {title}\n{content}")
        return "\n\n---\n\n".join(parts) if parts else "（无参考资料）"