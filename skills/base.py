from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> str:
        ...