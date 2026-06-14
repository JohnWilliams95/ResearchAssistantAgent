from langchain_core.messages import HumanMessage, SystemMessage
from agents.base import BaseAgent
from state.workflow_state import WorkflowState


class SynthesisAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.3)

    @property
    def name(self) -> str:
        return "synthesis_agent"

    @property
    def description(self) -> str:
        return "综合所有 Agent 收集的信息（Web搜索、ArXiv论文、本地知识库、技能分析），去重、验证一致性，生成结构化高质量答案"

    @property
    def system_prompt(self) -> str:
        return """你是一个专业的科研助手。你的任务是基于多个 Agent 收集的信息，综合生成高质量答案。

要求：
1. 综合所有信息源（网络搜索、学术论文、本地知识库、技能分析）
2. 标注信息来源，引用时注明 [来源类型]
3. 如果信息有矛盾，指出差异并给出判断
4. 答案要结构化、清晰、专业
5. 如果信息不足以完整回答，明确指出不足
6. 对于对比类问题，给出对比表格
7. 对于论文检索，给出论文列表及摘要"""

    def __call__(self, state: WorkflowState) -> dict:
        sections = []

        search_results = state.get("search_results", [])
        if search_results:
            sections.append("## Web 搜索结果\n")
            for i, r in enumerate(search_results, 1):
                title = r.get("title", "N/A")
                content = r.get("content", "")
                url = r.get("url", "")
                sections.append(f"### {i}. {title}\n{content}\n来源: {url}\n")

        arxiv_results = state.get("arxiv_results", [])
        if arxiv_results:
            sections.append("## ArXiv 论文结果\n")
            for i, r in enumerate(arxiv_results, 1):
                title = r.get("title", "N/A")
                content = r.get("content", "")
                sections.append(f"### {i}. {title}\n{content}\n")

        retrieval_results = state.get("retrieval_results", [])
        if retrieval_results:
            sections.append("## 本地知识库检索\n")
            for i, r in enumerate(retrieval_results, 1):
                title = r.get("title", "N/A")
                content = r.get("content", "")
                sections.append(f"### {i}. {title}\n{content}\n")

        skill_results = state.get("skill_results", [])
        if skill_results:
            sections.append("## 技能分析结果\n")
            for i, r in enumerate(skill_results, 1):
                sections.append(f"### 分析 {i}\n{r}\n")

        context = "\n".join(sections) if sections else "（没有获取到任何信息源）"

        prompt = f"""用户问题：{state['query']}

以下是从多个 Agent 收集的参考资料：

{context}

请基于以上信息，生成一份综合、高质量的回答。"""

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ])

        return {
            "final_answer": response.content,
            "messages": [response],
        }