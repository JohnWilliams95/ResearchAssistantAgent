from abc import ABC, abstractmethod
from langchain.chat_models import init_chat_model
from config.settings import settings
from state.workflow_state import WorkflowState


class BaseAgent(ABC):
    def __init__(self, temperature: float | None = None):
        self._temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = init_chat_model(
                settings.LLM_MODEL,
                model_provider=settings.LLM_MODEL_PROVIDER,
                api_key=settings.LLM_API_KEY or None,
                temperature=self._temperature,
            )
        return self._llm

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def __call__(self, state: WorkflowState) -> dict:
        pass