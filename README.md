# Research Assistant Agent

一个基于 LangGraph 的多 Agent 科研助手系统，能够接收用户的自然语言问题，自动规划、检索内外知识、调用外部工具（MCP）、执行原子技能（Skills），最终生成高质量答案。

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                      用户输入                         │
│         "对比 LangGraph 与 AutoGen 的最新特性"         │
└────────────────────────┬─────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────┐
│              Intent Analyzer Agent                    │
│        意图分类: compare / summarize / find_paper      │
│                  / explain / general                   │
└──────────┬──────────────┬──────────────┬─────────────┘
           ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Research   │ │  Retrieval   │ │    Skill     │
│    Agent     │ │    Agent     │ │   Executor   │
│ (Tavily +    │ │ (ChromaDB +  │ │ (Compare /   │
│  ArXiv MCP)  │ │  Embeddings) │ │  Summarize)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       └────────────────┴────────────────┘
                         ▼
┌──────────────────────────────────────────────────────┐
│              Synthesis Agent                         │
│    综合多源信息 → 去重 → 验证 → 生成结构化答案        │
└──────────────────────────────────────────────────────┘
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 编排 | LangGraph | 有状态图编排，Supervisor 模式路由 |
| LLM | init_chat_model (LangChain) | 通过 `LLM_MODEL_PROVIDER` 切换 OpenAI / DeepSeek / Ollama 等 |
| 向量库 | ChromaDB | 本地持久化，余弦相似度检索 |
| Embedding | sentence-transformers | `all-MiniLM-L6-v2`，本地推理无需 API |
| MCP | mcp python SDK | ArXiv MCP Server 标准化工具协议 |
| 搜索 | Tavily Search API | 实时网页搜索 |
| 学术 | ArXiv MCP Server | 论文检索 |

## 项目结构

```
ResearchAssistantAgent/
├── main.py                     # 入口：交互模式 / --demo
├── requirements.txt            # Python 依赖
├── .env                        # 环境变量（不提交）
├── .env.example                # 环境变量模板
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py             # 统一配置加载（从 .env 读取）
│
├── state/
│   ├── __init__.py
│   └── workflow_state.py       # LangGraph 共享状态定义 (TypedDict)
│
├── agents/                     # 多 Agent 架构核心
│   ├── __init__.py
│   ├── base.py                 # BaseAgent 抽象基类（延迟初始化 LLM）
│   ├── intent_analyzer_agent.py      # Agent 1: 意图分析
│   ├── research_agent.py             # Agent 2: 外部信息检索（Tavily + ArXiv）
│   ├── retrieval_agent.py            # Agent 3: 本地知识库检索（ChromaDB）
│   ├── skill_executor_agent.py       # Agent 4: 技能调度执行
│   ├── synthesis_agent.py            # Agent 5: 多源信息综合
│   └── supervisor.py           # 编排层：构建 Graph + 路由逻辑
│
├── tools/                      # 工具层
│   ├── __init__.py
│   ├── mcp_manager.py          # MCP Client 生命周期管理
│   ├── arxiv_tool.py           # ArXiv MCP Server 封装
│   ├── web_search_tool.py      # Tavily 搜索封装
│   └── tool_registry.py        # 工具索引
│
├── skills/                     # 原子技能池
│   ├── __init__.py
│   ├── base.py                 # BaseSkill 抽象基类
│   ├── compare.py              # 对比分析技能
│   ├── summarize.py            # 摘要提炼技能
│   └── skill_pool.py           # 技能注册 / 执行管理
│
├── retrieval/                  # 知识检索层
│   ├── __init__.py
│   ├── embedder.py             # sentence-transformers 封装（单例）
│   ├── vector_store.py         # ChromaDB 封装
│   └── document_loader.py      # 文档加载 / 分块
│
└── data/
    └── chroma_db/              # ChromaDB 持久化目录（自动创建）
```

## Agent 说明

### BaseAgent

所有 Agent 的抽象基类，提供：

- **延迟初始化 LLM**：首次调用 `self.llm` 时才创建 `init_chat_model` 实例，避免导入时就需要有效的 API Key
- 统一接口：`name` / `description` / `system_prompt` / `__call__(state)`

### Intent Analyzer Agent

分析用户问题意图，分类为 `compare` / `summarize` / `find_paper` / `explain` / `general`，驱动下游路由。

### Research Agent

并行调用 Tavily Web 搜索和 ArXiv MCP Server 论文检索，收集外部信息。通过 MCP 协议与 ArXiv Server 通信。

### Retrieval Agent

在本地 ChromaDB 知识库中进行语义检索。首次运行时自动写入 LLM/Agent 领域种子知识，无需额外准备文档即可演示。

### Skill Executor Agent

根据意图自动选择并执行原子技能（Compare / Summarize）。技能通过 `SkillPool` 注册，可插拔扩展。

### Synthesis Agent

综合所有 Agent 收集的信息（Web、ArXiv、知识库、技能分析），去重、验证一致性，生成结构化高质量答案。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入你的 API Key：

```env
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
LLM_MODEL_PROVIDER=openai
TAVILY_API_KEY=your-tavily-key
```

### 3. 运行

```bash
# 交互模式
python main.py

# 运行预设 Demo 场景（3 个典型问题）
python main.py --demo
```

## 预设 Demo 场景

| # | 问题 | 触发 Agent | 触发技能 |
|---|------|-----------|---------|
| 1 | 对比 LangGraph 与 AutoGen 的最新特性 | Research + Skill | Compare |
| 2 | 2024年 AI Agent 领域有哪些重要进展？ | Research + Retrieval | - |
| 3 | 什么是 MCP (Model Context Protocol) 协议？ | Research + Retrieval | - |

## 切换 LLM 提供商

只需修改 `.env` 中的两个值：

| 提供商 | LLM_MODEL | LLM_MODEL_PROVIDER | 备注 |
|--------|-----------|-------------------|------|
| OpenAI | gpt-4o-mini | openai | 默认 |
| DeepSeek | deepseek-chat | deepseek | 需 `pip install langchain-deepseek` |
| Ollama | llama3 | ollama | 本地部署，无需 API Key |

## 扩展方式

### 新增 Agent

1. 在 `agents/` 下创建新文件，继承 `BaseAgent`
2. 实现 `name` / `description` / `system_prompt` / `__call__`
3. 在 `agents/supervisor.py` 中实例化并注册到 Graph

### 新增 Skill

1. 在 `skills/` 下创建新文件，继承 `BaseSkill`
2. 实现 `name` / `description` / `execute`
3. 在 `skills/skill_pool.py` 的 `_register_default_skills` 中注册

### 新增工具

1. 在 `tools/` 下创建封装
2. 在对应 Agent 中调用

## License

MIT
