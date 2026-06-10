"""Research Assistant Agent — Multi-Agent System Demo

一个基于 LangGraph 的多 Agent 科研助手系统，支持：
- 自然语言问题理解与意图分析
- Web 搜索 (Tavily) + ArXiv 论文检索 (MCP) 双通道信息获取
- 本地 ChromaDB 知识库语义检索
- 可插拔原子技能（对比分析、摘要提炼）
- 多源信息综合与高质量答案生成

Usage:
    # 交互模式
    python main.py

    # 运行预设 demo 场景
    python main.py --demo
"""

import sys
import asyncio
from agents.supervisor import build_workflow
from state.workflow_state import WorkflowState

DEMO_QUERIES = [
    "对比 LangGraph 与 AutoGen 的最新特性",
    "2024年 AI Agent 领域有哪些重要进展？",
    "什么是 MCP (Model Context Protocol) 协议？",
]


def run_query(query: str, verbose: bool = True) -> str:
    app = build_workflow()
    initial: WorkflowState = {
        "query": query,
        "intent": "",
        "messages": [],
        "search_results": [],
        "arxiv_results": [],
        "retrieval_results": [],
        "skill_results": [],
        "plan": "",
        "partial_answers": [],
        "final_answer": "",
        "error": None,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Query: {query}")
        print(f"{'='*60}")

    result = app.invoke(initial)

    if verbose:
        intent = result.get("intent", "unknown")
        print(f"\n  Intent: {intent}")
        print(f"  Web Results: {len(result.get('search_results', []))} items")
        print(f"  ArXiv Results: {len(result.get('arxiv_results', []))} items")
        print(f"  KB Results: {len(result.get('retrieval_results', []))} items")
        print(f"  Skill Results: {len(result.get('skill_results', []))} items")
        print(f"\n{'─'*60}")
        print("  Final Answer:")
        print(f"{'─'*60}")
        print(result.get("final_answer", "No answer generated."))
        print(f"{'='*60}\n")

    return result.get("final_answer", "")


def interactive_mode():
    print("\n" + "=" * 60)
    print("  Research Assistant Agent — 多 Agent 科研助手")
    print("  Type 'quit' to exit, 'demo' to run demo scenarios")
    print("=" * 60)

    while True:
        try:
            query = input("\n  You: ").strip()
            if query.lower() == "quit":
                print("  Goodbye!")
                break
            if query.lower() == "demo":
                run_demo()
                continue
            if not query:
                continue
            run_query(query)
        except KeyboardInterrupt:
            print("\n  Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


def run_demo():
    print("\n  Running demo scenarios...")
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n  --- Demo {i}/{len(DEMO_QUERIES)} ---")
        run_query(query)


def main():
    if "--demo" in sys.argv:
        run_demo()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()