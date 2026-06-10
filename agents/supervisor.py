from langgraph.graph import StateGraph, END
from state.workflow_state import WorkflowState
from agents.intent_analyzer import IntentAnalyzerAgent
from agents.research import ResearchAgent
from agents.retrieval import RetrievalAgent
from agents.skill_executor import SkillExecutorAgent
from agents.synthesis import SynthesisAgent


# ── 实例化所有 Agent（每个都有独立的 System Prompt 和 LLM 实例）──

intent_analyzer = IntentAnalyzerAgent()
research_agent = ResearchAgent()
retrieval_agent = RetrievalAgent()
skill_executor = SkillExecutorAgent()
synthesis_agent = SynthesisAgent()

ALL_AGENTS = {
    intent_analyzer.name: intent_analyzer,
    research_agent.name: research_agent,
    retrieval_agent.name: retrieval_agent,
    skill_executor.name: skill_executor,
    synthesis_agent.name: synthesis_agent,
}


# ── 路由函数 ──


def route_after_intent(state: WorkflowState) -> list[str]:
    intent = state.get("intent", "general")
    nodes = []
    if intent in ("compare", "find_paper", "general"):
        nodes.append(research_agent.name)
    if intent in ("explain", "general"):
        nodes.append(retrieval_agent.name)
    if intent in ("compare", "summarize"):
        nodes.append(skill_executor.name)
    if not nodes:
        nodes.append(research_agent.name)
    return nodes


def route_to_synthesis(state: WorkflowState) -> str:
    return synthesis_agent.name


# ── 构建 Graph ──


def build_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    workflow.add_node(intent_analyzer.name, intent_analyzer)
    workflow.add_node(research_agent.name, research_agent)
    workflow.add_node(retrieval_agent.name, retrieval_agent)
    workflow.add_node(skill_executor.name, skill_executor)
    workflow.add_node(synthesis_agent.name, synthesis_agent)

    workflow.set_entry_point(intent_analyzer.name)

    workflow.add_conditional_edges(
        intent_analyzer.name,
        route_after_intent,
        {
            research_agent.name: research_agent.name,
            retrieval_agent.name: retrieval_agent.name,
            skill_executor.name: skill_executor.name,
        },
    )

    workflow.add_conditional_edges(
        research_agent.name,
        route_to_synthesis,
        {synthesis_agent.name: synthesis_agent.name},
    )
    workflow.add_conditional_edges(
        retrieval_agent.name,
        route_to_synthesis,
        {synthesis_agent.name: synthesis_agent.name},
    )
    workflow.add_conditional_edges(
        skill_executor.name,
        route_to_synthesis,
        {synthesis_agent.name: synthesis_agent.name},
    )

    workflow.add_edge(synthesis_agent.name, END)

    return workflow.compile()


def list_agents() -> list[dict]:
    return [
        {"name": a.name, "description": a.description}
        for a in ALL_AGENTS.values()
    ]