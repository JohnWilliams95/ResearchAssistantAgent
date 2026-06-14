import json
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base import BaseAgent
from state.workflow_state import WorkflowState
from skills.skill_pool import SkillPool


class SkillExecutorAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.1)
        self._skill_pool = SkillPool()

    @property
    def name(self) -> str:
        return "skill_executor"

    @property
    def description(self) -> str:
        return "根据用户意图自动选择合适的原子技能（对比分析/摘要提炼等）并执行"

    @property
    def system_prompt(self) -> str:
        skills_desc = "\n".join(
            f'- {s["name"]}: {s["description"]}' for s in self._skill_pool.list_skills()
        )
        return f"""你是一个技能调度专家。你管理以下原子技能：

{skills_desc}

根据用户意图和上下文，决定需要调用哪些技能。输出 JSON 格式的调用计划：
{{"skills": [{{"name": "技能名", "params": {{"query": "传给技能的查询"}}}}]}}

规则：
- 可以组合多个技能（按顺序执行）
- 如果没有合适的技能，返回 {{"skills": []}}
- params 中的 query 应该是适合该技能处理的查询文本，可以基于用户原始问题进行调整
- 分析可用技能的能力，选择最匹配用户意图的技能组合"""

    def __call__(self, state: WorkflowState) -> dict:
        query = state["query"]
        intent = state.get("intent", "general")

        plan_response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_plan_prompt(state)),
        ])
        plan = self._parse_json(plan_response.content)
        skills_to_run = plan.get("skills", [])

        results = []
        for skill_call in skills_to_run:
            skill_name = skill_call.get("name", "")
            skill_params = skill_call.get("params", {})
            skill_params.setdefault("query", query)
            skill_params["search_results"] = state.get("search_results", [])
            skill_params["arxiv_results"] = state.get("arxiv_results", [])
            skill_params["retrieval_results"] = state.get("retrieval_results", [])

            result = self._skill_pool.execute(skill_name, skill_params)
            results.append(result)

        return {"skill_results": results}

    def _build_plan_prompt(self, state: WorkflowState) -> str:
        query = state["query"]
        intent = state.get("intent", "general")

        skills_desc = "\n".join(
            f'- {s["name"]}: {s["description"]}' for s in self._skill_pool.list_skills()
        )

        context_parts = []
        search_results = state.get("search_results", [])
        arxiv_results = state.get("arxiv_results", [])
        retrieval_results = state.get("retrieval_results", [])

        if search_results:
            context_parts.append(f"Web 搜索结果: {len(search_results)} 条")
        if arxiv_results:
            context_parts.append(f"ArXiv 论文: {len(arxiv_results)} 条")
        if retrieval_results:
            context_parts.append(f"知识库检索: {len(retrieval_results)} 条")

        context_text = "\n".join(context_parts) if context_parts else "（暂无辅助信息）"

        return f"""用户问题：{query}
用户意图：{intent}

可用技能：
{skills_desc}

已有信息来源：
{context_text}

请分析用户意图，决定需要调用哪些技能，输出调用计划的 JSON。"""

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}