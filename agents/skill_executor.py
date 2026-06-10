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
        return f"""你是一个技能调度专家。你管理以下原子技能：

{chr(10).join(f'- {s["name"]}: {s["description"]}' for s in self._skill_pool.list_skills())}

根据用户意图，自动选择合适的技能并执行。"""

    def __call__(self, state: WorkflowState) -> dict:
        intent = state.get("intent", "general")

        skill_map = {
            "compare": "compare",
            "summarize": "summarize",
        }
        skill_name = skill_map.get(intent)

        if skill_name is None:
            return {"skill_results": []}

        result = self._skill_pool.execute(skill_name, {
            "query": state["query"],
            "search_results": state.get("search_results", []),
            "arxiv_results": state.get("arxiv_results", []),
            "retrieval_results": state.get("retrieval_results", []),
        })

        return {"skill_results": [result]}