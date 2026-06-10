from langchain.chat_models import init_chat_model
from config.settings import settings
from skills.base import BaseSkill


class SummarizeSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "summarize"

    @property
    def description(self) -> str:
        return "对多来源信息进行摘要和提炼"

    def execute(self, context: dict) -> str:
        query = context.get("query", "")
        search_results = context.get("search_results", [])
        arxiv_results = context.get("arxiv_results", [])

        all_text = []
        for r in search_results:
            content = r.get("content", "")
            if content:
                all_text.append(content)
        for r in arxiv_results:
            content = r.get("content", "")
            if content:
                all_text.append(content)

        source_text = "\n\n---\n\n".join(all_text) if all_text else "（无参考资料）"

        llm = init_chat_model(
            settings.LLM_MODEL,
            model_provider=settings.LLM_MODEL_PROVIDER,
            api_key=settings.LLM_API_KEY or None,
            temperature=settings.LLM_TEMPERATURE,
        )

        prompt = f"""你是一个专业的信息提炼专家。请从参考资料中提取与用户问题相关的关键信息，整理成结构化的摘要。

用户问题：{query}

参考资料：
{source_text}

要求：
1. 提取关键事实和数据点
2. 去重合并相同信息
3. 按重要性排序
4. 用简洁清晰的语言表述
5. 如果参考资料不足以回答问题，请明确说明"""

        response = llm.invoke(prompt)
        return response.content