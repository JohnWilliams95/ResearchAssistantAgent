from langchain_core.messages import HumanMessage
from agents.base import BaseAgent
from state.workflow_state import WorkflowState


class IntentAnalyzerAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "intent_analyzer"

    @property
    def description(self) -> str:
        return "分析用户问题的意图类别，将问题路由到合适的下游 Agent"

    @property
    def system_prompt(self) -> str:
        return """你是一个意图分析专家。你的任务是将用户的自然语言问题分类到以下意图之一：

- compare: 对比/比较两个或多个事物
- summarize: 总结/摘要信息
- find_paper: 查找论文/文献/学术资料
- explain: 解释概念/技术/原理
- general: 通用问题

只返回意图类别名称（单个词），不要返回其他内容。"""

    def __call__(self, state: WorkflowState) -> dict:
        query = state["query"]
        response = self.llm.invoke([
            HumanMessage(content=self.system_prompt),
            HumanMessage(content=f"用户问题：{query}"),
        ])
        intent = response.content.strip().lower()
        valid = {"compare", "summarize", "find_paper", "explain", "general"}
        if intent not in valid:
            intent = "general"
        return {"intent": intent, "messages": [response]}