from typing import Optional
from skills.base import BaseSkill
from skills.compare import CompareSkill
from skills.summarize import SummarizeSkill


class SkillPool:
    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}
        self._register_default_skills()

    def _register_default_skills(self):
        self.register(CompareSkill())
        self.register(SummarizeSkill())

    def register(self, skill: BaseSkill):
        self._skills[skill.name] = skill

    def get(self, skill_name: str) -> Optional[BaseSkill]:
        return self._skills.get(skill_name)

    def execute(self, skill_name: str, context: dict) -> str:
        skill = self.get(skill_name)
        if skill is None:
            return f"[Error] Unknown skill: {skill_name}"
        return skill.execute(context)

    def list_skills(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]