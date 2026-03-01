# CrewAI 深度研究報告：框架分析與 AI Agent 框架比較

## 目錄

1. [CrewAI 是什麼？](#1-crewai-是什麼)
2. [核心架構與運作原理](#2-核心架構與運作原理)
3. [支援的 LLM 供應商](#3-支援的-llm-供應商)
4. [工具生態系](#4-工具生態系)
5. [與主流 AI Agent 框架的完整比較](#5-與主流-ai-agent-框架的完整比較)
6. [各框架 GitHub 數據對比](#6-各框架-github-數據對比)
7. [適用場景分析](#7-適用場景分析)
8. [本機安裝指南](#8-本機安裝指南)
9. [結論與建議](#9-結論與建議)
10. [雲端部署指南 — 以 GCP 為例](#10-雲端部署指南--以-gcp-為例)
11. [開源模型自架方案 — 降低 LLM API 成本](#11-開源模型自架方案--降低-llm-api-成本)

---

## 1. CrewAI 是什麼？

**CrewAI** 是一個輕量、高效的 Python 多 AI Agent 自動化框架，由 Joao Moura 創建，目前版本為 **1.10.1a1**。它完全從零打造，**不依賴 LangChain 或任何其他 Agent 框架**，以獨立、精簡、高效能為核心設計理念。

### 核心定位

- **角色扮演式多 Agent 協作**：每個 Agent 有明確的角色 (role)、目標 (goal) 和背景故事 (backstory)
- **團隊協作模式**：將 AI Agent 組織成「船員」(Crew) 團隊，模擬真實人類團隊的運作方式
- **生產級就緒**：從設計之初就考量企業部署場景，支援快取、速率限制、錯誤處理等

### 關鍵數據

| 指標 | 數據 |
|------|------|
| GitHub Stars | **44,000+** |
| 認證開發者 | 100,000+ |
| 授權模式 | MIT License (開源) |
| Python 版本要求 | >=3.10, <3.14 |
| 套件管理工具 | UV (Astral) |

---

## 2. 核心架構與運作原理

CrewAI 的架構由兩大核心模式組成：**Crews（團隊模式）** 和 **Flows（流程模式）**。

### 2.1 Crews — 自主團隊協作

```
┌─────────────────────────────────────────┐
│                  Crew                    │
│                                         │
│  ┌──────────┐  ┌──────────┐            │
│  │  Agent 1  │  │  Agent 2  │  ...      │
│  │ (研究員)  │  │ (分析師)  │            │
│  │ role      │  │ role      │            │
│  │ goal      │  │ goal      │            │
│  │ backstory │  │ backstory │            │
│  │ tools[]   │  │ tools[]   │            │
│  └─────┬─────┘  └─────┬─────┘            │
│        │              │                  │
│  ┌─────▼─────┐  ┌─────▼─────┐           │
│  │   Task 1   │──│   Task 2   │  ...     │
│  │description │  │description │           │
│  │expected_out│  │expected_out│           │
│  └────────────┘  └────────────┘           │
│                                          │
│  Process: sequential / hierarchical      │
└──────────────────────────────────────────┘
```

**四大核心元件：**

| 元件 | 說明 | 原始碼位置 |
|------|------|-----------|
| **Agent** | 自主實體，有角色、目標、背景故事 | `lib/crewai/src/crewai/agent/core.py` |
| **Task** | 工作單元，定義描述和預期輸出 | `lib/crewai/src/crewai/task.py` |
| **Crew** | 團隊容器，協調 Agent 執行 Task | `lib/crewai/src/crewai/crew.py` |
| **Process** | 執行策略（順序/層級式） | `lib/crewai/src/crewai/process.py` |

**執行流程 (Process) 類型：**

- **Sequential（順序執行）**：Task 依序執行，前一個 Task 的輸出成為下一個的上下文
- **Hierarchical（層級管理）**：Manager Agent 自動協調任務分配與驗證

### 2.2 Flows — 事件驅動工作流

Flows 是生產環境用的精確控制架構，支援：

```python
class AdvancedFlow(Flow[MyState]):
    @start()              # 入口點
    def begin(self): ...

    @listen(begin)        # 監聽事件
    def process(self, data): ...

    @router(process)      # 條件路由
    def decide(self): ...
```

**Flow 核心裝飾器：**

| 裝飾器 | 功能 |
|--------|------|
| `@start()` | 定義流程入口點 |
| `@listen()` | 監聽其他方法完成事件 |
| `@router()` | 根據條件進行路由分支 |
| `and_()` | 所有條件滿足時觸發 |
| `or_()` | 任一條件滿足時觸發 |

**Crews + Flows 的組合威力：**

Flows 可以將多個 Crews 串聯起來，在保持 Agent 自主性的同時，提供精確的工作流控制、狀態管理和條件分支。

### 2.3 記憶系統 (Memory)

- **統一記憶 (Unified Memory)**：具 LLM 分析能力的智慧記憶
- **可插拔儲存**：預設 LanceDB，支援自訂後端
- **自適應回想 (Recall Flow)**：動態深度的記憶檢索
- **評分機制**：時效性、語意相似度、重要性多維評分

### 2.4 知識庫 / RAG

- 內建 ChromaDB 和 Qdrant 向量資料庫支援
- 多種 Embedding 提供者（OpenAI、Cohere、HuggingFace、Voyage AI 等）
- 語意搜尋和上下文注入

### 2.5 事件系統與可觀測性

- 事件匯流排 (Event Bus) 架構
- OpenTelemetry 整合的分散式追蹤
- 覆蓋 Agent、Task、Flow、LLM、Tool、Memory 等全面的事件類型

---

## 3. 支援的 LLM 供應商

CrewAI 提供原生和透過 LiteLLM 的廣泛 LLM 支援：

| 供應商 | 原生支援 | 安裝方式 | 模型範例 |
|--------|---------|---------|---------|
| **OpenAI** | 預設 | 內建 | GPT-4o, GPT-4-turbo, o1, o3 |
| **Anthropic** | 原生 Provider | `crewai[anthropic]` | Claude 3 Opus/Sonnet/Haiku |
| **Google Gemini** | 原生 Provider | `crewai[google-genai]` | Gemini 2.0, 1.5 |
| **Azure OpenAI** | 原生 Provider | 內建 | Azure GPT-4 變體 |
| **AWS Bedrock** | 原生 Provider | `crewai[bedrock]` | Claude, Titan, Llama |
| **IBM Watson** | 選用依賴 | `crewai[watson]` | WatsonX 模型 |
| **Ollama** | 透過 LiteLLM | `crewai[litellm]` | Llama, Mistral (本地) |
| **100+ 其他模型** | 透過 LiteLLM | `crewai[litellm]` | 各種開源/商用模型 |

---

## 4. 工具生態系

CrewAI 提供 **77+ 預建工具**（`crewai-tools` 套件），涵蓋：

### 搜尋工具
SerperDev, BraveSearch, EXASearch, TavilySearch, Google (SerpApi), Linkup

### 網頁爬取
Firecrawl, Jina, Browserbase, HyperBrowser, Selenium, Scrapfly, Stagehand

### 資料處理
CSV, JSON, PDF, DOCX, Excel, XML, TXT 搜尋工具

### 程式碼
CodeInterpreter, GitHub Search, CodeDocs Search

### 雲端整合
AWS S3 (讀/寫), Bedrock Knowledge Base, Bedrock Agent

### 第三方整合
Zapier, Composio, MultiOn, LlamaIndex, MongoDB, Weaviate, Qdrant, Couchbase, Snowflake, Databricks, SingleStore

### AI 增強工具
DALL-E (圖像生成), Vision Tool, OCR Tool, AI Mind Tool

### MCP 支援
原生 Model Context Protocol (MCP) 整合，可連接任何 MCP 伺服器

---

## 5. 與主流 AI Agent 框架的完整比較

### 5.1 CrewAI vs LangGraph (LangChain)

| 維度 | CrewAI | LangGraph |
|------|--------|-----------|
| **架構模式** | 角色導向的團隊協作 | 有向圖狀態機 |
| **設計哲學** | 「告訴它做什麼」—預組裝機器人 | 「自己組裝」—樂高積木盒 |
| **學習曲線** | 低：直覺的角色/任務概念 | 高：需學習圖論和狀態管理 |
| **狀態管理** | 隱式，每角色隔離 + 共享 Crew 存儲 | 顯式一等公民，可序列化、持久化 |
| **效能** | 某些 QA 任務快 5.76x | 延遲最低，Token 使用最少 |
| **上手到部署** | 快 40%：最快的 Time-to-Production | 需較多前期設定 |
| **適合場景** | 業務流程自動化、團隊協作 | 需要精確控制和重播的生產系統 |
| **生態系** | 77+ 內建工具 | LangChain 龐大生態（47M+ 下載） |
| **GitHub Stars** | ~44,000 | ~25,000 |

### 5.2 CrewAI vs AutoGen (Microsoft)

| 維度 | CrewAI | AutoGen |
|------|--------|--------|
| **架構模式** | 角色導向團隊 | 對話式多 Agent |
| **協作方式** | 任務分配與委派 | 結構化輪流對話 |
| **核心優勢** | 快速原型、直覺 API | 迭代式優化、共識建構 |
| **額外功能** | — | AutoGen Studio (無程式碼工具) |
| **效能** | 中等 | 最慢（對話共識開銷大） |
| **除錯難度** | 中等（抽象層可能吞錯誤） | 較高（版本文件混亂） |
| **適合場景** | 結構化業務工作流 | 程式碼生成、辯論/討論場景 |

### 5.3 CrewAI vs OpenAI Agents SDK

| 維度 | CrewAI | OpenAI Agents SDK |
|------|--------|-------------------|
| **架構模式** | 角色導向團隊 | 輕量工具導向 |
| **學習門檻** | 低 | 最低（<20 行程式碼即可運行） |
| **LLM 綁定** | 模型無關（100+） | 限 OpenAI 模型 |
| **多 Agent 支援** | 原生完整支援 | 透過 Handoff 機制 |
| **適合場景** | 生產級多 Agent 系統 | 快速原型、OpenAI 生態內開發 |
| **社群成熟度** | 成熟，44K Stars | 新但成長快 |

### 5.4 CrewAI vs Agno (前身 PhiData)

| 維度 | CrewAI | Agno |
|------|--------|------|
| **架構模式** | 角色導向團隊 | 純 Python 組合式 |
| **效能** | 良好 | 極快（Agent 實例化快 ~10,000x） |
| **多模態** | 基本支援 | 原生支援（文字/圖像/音訊/影片） |
| **多 Agent** | 原生完整支援 | 正在追趕中 |
| **文件品質** | 結構化、入門友善 | 優秀但因更名有斷連問題 |
| **託管平台** | $99/月起 | $150/月起 (AgentOS) |
| **適合場景** | 複雜團隊工作流 | 高效能、低延遲、多模態應用 |

### 5.5 CrewAI vs Google ADK (Agent Development Kit)

| 維度 | CrewAI | Google ADK |
|------|--------|------------|
| **架構模式** | 角色導向團隊 | 模組化圖基多 Agent |
| **成熟度** | 穩定 (v1.10) | 新 (v0.1.x) |
| **雲端整合** | 靈活（多雲端） | 深度整合 Google Cloud / Vertex AI |
| **串流支援** | 基本 | 雙向音訊/影片串流 |
| **適合場景** | 快速原型、團隊工作流 | Google 生態企業部署 |
| **社群規模** | 大且成熟 | 成長中但較小 |

### 5.6 其他值得注意的框架

| 框架 | 特色 | 適合場景 |
|------|------|---------|
| **Semantic Kernel** (Microsoft) | 企業 AI 編排，深度整合 Azure | 已使用 Microsoft 技術棧的企業 |
| **Haystack** (deepset) | Pipeline 導向 | RAG 和文件搜尋為主的應用 |
| **MetaGPT** | 模擬軟體公司結構 | 軟體開發自動化 |
| **PydanticAI** | 型別安全優先 | 需要嚴格型別驗證的應用 |
| **DSPy** | 程式化 LLM 呼叫最佳化 | 需要最佳化 Prompt 的研究場景 |

---

## 6. 各框架 GitHub 數據對比

| 框架 | GitHub Stars (2026) | 核心架構 | 授權 |
|------|-------------------|---------|------|
| **LangChain** | ~90,000+ | LLM 應用框架（非純 Agent） | MIT |
| **CrewAI** | ~44,000+ | 角色導向多 Agent 團隊 | MIT |
| **AutoGen** | ~40,000+ | 對話式多 Agent | MIT |
| **LangGraph** | ~25,000+ | 有向圖狀態機 | MIT |
| **Agno** | ~20,000+ | 純 Python 組合式 | MIT |
| **Google ADK** | ~18,000+ | 模組化多 Agent | Apache 2.0 |
| **MetaGPT** | ~15,000+ | 模擬組織結構 | MIT |

> 注：Agent 框架 GitHub Stars 超過 1,000 的專案從 2024 年的 14 個增長到 2025 年的 89 個 —— 增長了 535%。

---

## 7. 適用場景分析

### 選擇 CrewAI 的情境

- 需要**多個專業 Agent 協作**的業務工作流
- 追求**快速原型開發**和上手部署（比 LangGraph 快 40%）
- 需要豐富的**內建工具生態**（77+ 工具）
- 希望用**直覺的角色/任務模型**來描述 Agent 系統
- 需要同時具備**自主性 (Crews)** 和**精確控制 (Flows)** 的靈活性

### 不適合 CrewAI 的情境

- 需要**極度精確的狀態管理和重播**（考慮 LangGraph）
- 只需要**單一 Agent 快速原型**（考慮 OpenAI Agents SDK）
- 需要**最低延遲和多模態**處理（考慮 Agno）
- 已深度整合 **Google Cloud** 生態（考慮 Google ADK）
- 需要**對話式辯論/共識**場景（考慮 AutoGen）

---

## 8. 本機安裝指南

### 8.1 前置需求

```bash
# 1. 確認 Python 版本 (需要 >=3.10, <3.14)
python3 --version

# 2. 安裝 UV 套件管理器（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 8.2 方式一：從 PyPI 安裝（使用者）

```bash
# 基本安裝
uv pip install crewai

# 安裝含工具套件
uv pip install 'crewai[tools]'

# 安裝含特定功能
uv pip install 'crewai[tools,anthropic,litellm]'
```

### 8.3 方式二：從原始碼安裝（開發者）

```bash
# 1. Clone 專案
git clone https://github.com/crewAIInc/crewAI.git
cd crewAI

# 2. 建立虛擬環境
uv venv

# 3. 啟動虛擬環境
source .venv/bin/activate   # Linux/macOS
# 或
.venv\Scripts\activate      # Windows

# 4. 鎖定並同步依賴
uv lock
uv sync

# 5. 安裝 pre-commit hooks
pre-commit install
```

### 8.4 建立第一個 Crew 專案

```bash
# 使用 CLI 建立專案骨架
crewai create crew my-first-crew
cd my-first-crew

# 設定環境變數
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 安裝專案依賴
crewai install

# 執行 Crew
crewai run
```

### 8.5 開發模式下的驗證

```bash
# 執行測試（需在專案根目錄 /home/user/crewAI）
uv run pytest .

# 靜態型別檢查
uvx mypy src

# 建置套件
uv build

# 安裝本地建置版本
uv pip install dist/*.tar.gz
```

### 8.6 常見問題排解

| 問題 | 解決方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'tiktoken'` | `uv pip install 'crewai[embeddings]'` |
| 建置 tiktoken 輪子失敗 | 安裝 Rust 編譯器；`uv pip install tiktoken --prefer-binary` |
| Poetry 相關錯誤 | 執行 `crewai update` |

---

## 9. 結論與建議

### CrewAI 的核心優勢

1. **直覺的心智模型**：角色/任務的比喻讓非技術人員也能理解
2. **雙模架構**：Crews (自主性) + Flows (精確控制) 的獨特組合
3. **豐富生態**：77+ 工具、MCP 支援、多 LLM 供應商
4. **快速部署**：比 LangGraph 快 40% 的 Time-to-Production
5. **獨立架構**：不依賴 LangChain，更精簡高效

### 2026 年趨勢

AI Agent 框架正朝向 **Agentic Mesh（代理網格）** 發展 —— 未來不是選擇單一框架，而是走向模組化生態系，例如：
- 用 **LangGraph** 做「大腦」進行狀態管理
- 用 **CrewAI** 組建「行銷團隊」處理特定工作流
- 用 **OpenAI Agents SDK** 做快速子任務

### 最終建議

| 你的需求 | 推薦框架 |
|---------|---------|
| 第一次嘗試 AI Agent | **OpenAI Agents SDK** |
| 業務流程自動化 | **CrewAI** |
| 精確狀態控制的生產系統 | **LangGraph** |
| 高效能多模態應用 | **Agno** |
| Google Cloud 深度整合 | **Google ADK** |
| 對話式辯論/共識建構 | **AutoGen** |
| 企業級 Azure 整合 | **Semantic Kernel** |

---

## 10. 雲端部署指南 — 以 GCP 為例

CrewAI 可以透過多種方式部署到雲端。以下是從簡單到進階的完整方案。

### 10.1 部署架構選項一覽

| 方案 | 複雜度 | 成本 | 適合場景 |
|------|--------|------|---------|
| **Cloud Run + FastAPI** | 低 | 低（按用量計費） | 原型、小型生產 |
| **GKE (Kubernetes)** | 高 | 中-高 | 大規模生產環境 |
| **Compute Engine + Docker** | 中 | 中 | 需完全控制的場景 |
| **CrewAI AMP Enterprise** | 低 | 高（訂閱制） | 企業級託管方案 |
| **Cloud Run + A2A 協定** | 中 | 中 | 多框架 Agent 互通 |

### 10.2 推薦方案：Google Cloud Run + FastAPI（最快上手）

**架構圖：**

```
使用者/前端
    │
    ▼
┌──────────────┐     ┌──────────────────┐
│ Cloud Run    │────▶│ Secret Manager   │
│ (FastAPI +   │     │ (API Keys)       │
│  CrewAI)     │     └──────────────────┘
│              │
│  Container   │────▶ OpenAI / Gemini / Claude API
│  (Docker)    │
└──────────────┘
    │
    ▼
┌──────────────┐
│ Cloud Storage│ (結果儲存，選用)
└──────────────┘
```

#### Step 1：建立 FastAPI 包裝層

```python
# app.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時初始化
    yield
    # 關閉時清理

app = FastAPI(title="CrewAI Service", lifespan=lifespan)

class CrewRequest(BaseModel):
    topic: str
    model: str = "gpt-4o"

class CrewResponse(BaseModel):
    result: str
    tokens_used: int | None = None

@app.post("/api/crew/run", response_model=CrewResponse)
async def run_crew(request: CrewRequest):
    try:
        researcher = Agent(
            role="Senior Researcher",
            goal=f"Research {request.topic} thoroughly",
            backstory="Expert researcher with deep analytical skills",
            llm=request.model,
            verbose=True,
        )
        writer = Agent(
            role="Content Writer",
            goal="Create a comprehensive report",
            backstory="Skilled writer who turns research into clear reports",
            llm=request.model,
            verbose=True,
        )

        research_task = Task(
            description=f"Research the topic: {request.topic}",
            expected_output="Detailed research findings with key points",
            agent=researcher,
        )
        writing_task = Task(
            description="Write a report based on research findings",
            expected_output="Well-structured report in markdown format",
            agent=writer,
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return CrewResponse(result=str(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

#### Step 2：建立 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# Cloud Run 預設使用 PORT 環境變數
ENV PORT=8080

# 啟動 FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Step 3：requirements.txt

```txt
crewai>=1.10.0
crewai-tools>=1.10.0
fastapi>=0.115.0
uvicorn>=0.34.0
python-dotenv>=1.0.0
```

#### Step 4：部署到 Cloud Run

```bash
# 1. 設定 GCP 專案
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-east1"          # 台灣最近的區域
export SERVICE_NAME="crewai-service"

# 2. 啟用必要 API
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com

# 3. 將 API Key 存入 Secret Manager
echo -n "sk-your-openai-key" | \
    gcloud secrets create openai-api-key --data-file=-

# 4. 建置並推送映像檔
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# 5. 部署到 Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --timeout 300 \
    --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
    --allow-unauthenticated  # 或移除此行改用 IAM 認證

# 6. 取得服務 URL
gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format "value(status.url)"
```

#### Step 5：測試

```bash
# 測試 API
curl -X POST https://your-service-url/api/crew/run \
    -H "Content-Type: application/json" \
    -d '{"topic": "AI Agent 框架趨勢", "model": "gpt-4o"}'
```

### 10.3 進階方案：GKE (Kubernetes) 部署

適合需要高可用、自動擴縮、多服務編排的生產環境。

```
┌─────────────────────────────────────────────┐
│              GKE Cluster                    │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ CrewAI  │  │ CrewAI  │  │ CrewAI  │    │
│  │ Pod 1   │  │ Pod 2   │  │ Pod N   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │          │
│  ┌────▼────────────▼────────────▼────┐     │
│  │        Kubernetes Service          │     │
│  └────────────────┬──────────────────┘     │
│                   │                         │
│  ┌────────────────▼──────────────────┐     │
│  │         Ingress / Load Balancer    │     │
│  └────────────────────────────────────┘     │
│                                             │
│  ┌──────────┐  ┌──────────┐                │
│  │  Redis   │  │ Cloud SQL│  (任務佇列/狀態)│
│  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────┘
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewai-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crewai
  template:
    metadata:
      labels:
        app: crewai
    spec:
      containers:
      - name: crewai
        image: gcr.io/YOUR_PROJECT/crewai-service:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: crewai-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: crewai-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: crewai
```

```bash
# GKE 部署指令
# 1. 建立 GKE 叢集
gcloud container clusters create crewai-cluster \
    --region asia-east1 \
    --num-nodes 3 \
    --machine-type e2-standard-4

# 2. 建立 Secret
kubectl create secret generic crewai-secrets \
    --from-literal=openai-api-key="sk-your-key"

# 3. 部署
kubectl apply -f k8s/deployment.yaml

# 4. 查看狀態
kubectl get pods
kubectl get services
```

### 10.4 使用 Gemini 模型的 GCP 原生方案

如果想完全使用 Google 生態，可以用 Gemini 模型取代 OpenAI：

```python
from crewai import Agent, LLM

# 使用 Gemini（需安裝 crewai[google-genai]）
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

agent = Agent(
    role="Researcher",
    goal="Research the given topic",
    backstory="Expert researcher",
    llm=gemini_llm,
)
```

> Google 提供了官方 quickstart 範例：[google-gemini/crewai-quickstart](https://github.com/google-gemini/crewai-quickstart)

### 10.5 A2A 協定：跨框架 Agent 互通

Google 的 Agent-to-Agent (A2A) 協定讓不同框架的 Agent 可以互相溝通：

```
┌──────────────┐  A2A  ┌──────────────┐
│  CrewAI      │◄─────►│  Google ADK  │
│  Research    │       │  Summarizer  │
│  Agent       │       │  Agent       │
│ (Cloud Run)  │       │ (Cloud Run)  │
└──────────────┘       └──────────────┘
```

CrewAI 已原生支援 A2A（`crewai[a2a]` 套件），可以：
- 讓 CrewAI Agent 以微服務方式獨立部署
- 與 Google ADK、LangGraph 等其他框架的 Agent 互相呼叫
- 每個 Agent 可獨立開發、部署、擴縮

### 10.6 生產環境注意事項

| 項目 | 建議 |
|------|------|
| **記憶體** | 最少 1-2 GB，複雜 Crew 需 4 GB+ |
| **超時設定** | Crew 執行可能需要數分鐘，Cloud Run 預設 5 分鐘（可調至 60 分鐘） |
| **API Key 管理** | 使用 GCP Secret Manager，切勿硬編碼 |
| **非同步處理** | 長時間任務用 Cloud Tasks / Pub/Sub 排隊 |
| **監控** | 整合 Cloud Monitoring + CrewAI 內建的 OpenTelemetry |
| **成本控制** | Cloud Run 按請求計費；注意 LLM API 呼叫費用 |
| **區域選擇** | 台灣用戶建議 `asia-east1`（彰化機房） |
| **瀏覽器工具** | 若 Agent 用 Playwright 等爬蟲工具，需額外安裝 Chromium |

### 10.7 GCP 方案成本估算

| 元件 | 月費（估算） |
|------|-------------|
| Cloud Run (低用量) | ~$5-30 |
| Cloud Run (中用量) | ~$30-150 |
| GKE 叢集 (3 節點) | ~$200-400 |
| Secret Manager | ~$0.06/secret |
| Cloud Build | 120 分鐘/天免費 |
| **LLM API 呼叫** | **依用量，通常是最大成本** |

> 注意：LLM API 費用通常遠超基礎設施費用。GPT-4o 約 $2.5-10/百萬 token，Claude Sonnet 約 $3-15/百萬 token。

### 10.8 其他雲端方案比較

| 雲端 | 推薦服務 | 優勢 |
|------|---------|------|
| **GCP** | Cloud Run / GKE | Gemini 原生整合、A2A 協定 |
| **AWS** | ECS / EKS / Lambda | Bedrock 整合、最大市佔率 |
| **Azure** | ACA / AKS | OpenAI 服務原生整合 |
| **Railway** | 一鍵部署 | 最簡單，適合原型 |
| **Fly.io** | 全球分佈 | 低延遲部署 |

---

## 11. 開源模型自架方案 — 降低 LLM API 成本

### 11.1 為什麼要用開源模型？

使用商用 LLM API（OpenAI GPT-4o、Anthropic Claude 等）是 CrewAI 部署的**最大持續成本**。以典型的多 Agent 工作流為例：

| 使用場景 | 每月 API 估算費用 |
|---------|------------------|
| 輕度使用（每天 50 次 Crew 執行） | $100-300 |
| 中度使用（每天 500 次） | $1,000-3,000 |
| 重度使用（每天 5,000 次） | $10,000+ |

**自架開源模型可將 LLM 成本從「按 token 付費」轉為「固定基礎設施費用」，** 高用量場景下可節省 3-5 倍成本。

### 11.2 CrewAI 原生支援開源模型

CrewAI 透過 **LiteLLM** 整合支援幾乎所有開源模型。原始碼 `lib/crewai/src/crewai/llm.py` 中的 LLM 類別會自動路由：

- **原生 Provider（直連）**：OpenAI、Anthropic、Google、Azure、Bedrock
- **LiteLLM Fallback（100+ 模型）**：Ollama、vLLM、HuggingFace、LM Studio 等所有 OpenAI 相容 API

使用方式極其簡單：

```python
from crewai import Agent, LLM

# 方式 1：透過 Ollama 使用本地模型
agent = Agent(
    role="Researcher",
    goal="Research the given topic",
    backstory="Expert researcher",
    llm=LLM(
        model="ollama/llama3.3",
        base_url="http://localhost:11434"    # Ollama 預設端口
    ),
)

# 方式 2：透過 vLLM 使用自架模型
agent = Agent(
    role="Analyst",
    goal="Analyze data",
    backstory="Data expert",
    llm=LLM(
        model="openai/meta-llama/Llama-3.3-70B-Instruct",
        base_url="http://your-vllm-server:8000/v1",  # vLLM OpenAI 相容 API
        api_key="token-abc123",                       # vLLM 的 token（任意值即可）
    ),
)

# 方式 3：同一 Crew 中混用不同模型
researcher = Agent(
    role="Researcher",
    llm=LLM(model="ollama/llama3.3", base_url="http://localhost:11434"),
    # ...
)
writer = Agent(
    role="Writer",
    llm=LLM(model="gpt-4o"),  # 寫作任務用商用模型
    # ...
)
```

### 11.3 推薦開源模型（2026 年）

| 模型 | 參數量 | VRAM 需求 | 最適場景 | 效能等級 |
|------|--------|-----------|---------|---------|
| **Llama 3.3 70B** | 70B | 40-80 GB | 通用任務（效能接近 GPT-4o） | 頂級 |
| **Qwen 2.5 72B** | 72B | 40-80 GB | 多語言（含繁中）、程式碼 | 頂級 |
| **DeepSeek V3** | 671B MoE | 80+ GB | 推理、數學、程式碼 | 頂級 |
| **Mistral Small 24B** | 24B | 16 GB | 平衡效能/成本 | 高 |
| **Llama 3.2 8B** | 8B | 6-8 GB | 簡單任務、低成本 | 中 |
| **Mistral 7B** | 7B | 6 GB | 基礎任務、預算最低 | 中 |
| **Gemma 3 27B** | 27B | 18 GB | Google 生態、多模態 | 高 |

> 量化技巧：使用 4-bit 量化（GGUF/AWQ）可將 VRAM 需求降低 50-75%，品質損失極小。例如 Llama 3.3 70B 從 ~140 GB 降至 ~35 GB。

### 11.4 推理引擎比較：Ollama vs vLLM

| 維度 | Ollama | vLLM |
|------|--------|------|
| **定位** | 「LLM 界的 Docker」 | 高吞吐量生產推理引擎 |
| **安裝複雜度** | 極低（一行指令） | 中等 |
| **吞吐量** | ~41 TPS | ~793 TPS（**19x 差距**） |
| **並發 128 用戶延遲** | 673ms (P99) | <100ms (P99) |
| **記憶體效率** | 一般 | PagedAttention 降低 40%+ 碎片 |
| **API 相容** | OpenAI 相容 | OpenAI 相容 |
| **適合場景** | 開發/原型/個人 | 生產/多用戶/高並發 |
| **模型格式** | GGUF（自動量化） | HuggingFace、AWQ、GPTQ |

**決策原則**：開發用 Ollama → 生產用 vLLM。兩者都暴露 OpenAI 相容 API，CrewAI 程式碼只需改 `base_url`。

### 11.5 GCP 部署架構

#### 方案 A：Cloud Run GPU + Ollama（最簡單）

Google Cloud 官方支援在 Cloud Run 上掛載 GPU 跑 Ollama：

```
┌──────────────────────────────────────────┐
│            Cloud Run (GPU)               │
│                                          │
│  ┌──────────┐     ┌──────────────────┐  │
│  │ CrewAI   │────▶│ Ollama (Sidecar) │  │
│  │ FastAPI  │     │ Llama 3.3 / Qwen │  │
│  │ :8080    │     │ :11434           │  │
│  └──────────┘     └──────────────────┘  │
│                          │              │
│                    ┌─────▼─────┐        │
│                    │ NVIDIA L4 │        │
│                    │   GPU     │        │
│                    └───────────┘        │
└──────────────────────────────────────────┘
```

```bash
# 1. 使用 Google 官方文件的 Ollama + GPU Cloud Run 部署
# 參考：https://cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama

# 建立包含 Ollama + 模型的 Docker 映像
cat > Dockerfile.ollama <<'EOF'
FROM ollama/ollama:latest

# 預下載模型到映像中（加速冷啟動）
RUN ollama serve & sleep 5 && ollama pull llama3.3 && killall ollama

EXPOSE 11434
CMD ["ollama", "serve"]
EOF

# 建置並部署
gcloud builds submit --tag gcr.io/$PROJECT_ID/ollama-llama
gcloud run deploy ollama-service \
    --image gcr.io/$PROJECT_ID/ollama-llama \
    --region us-central1 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --memory 24Gi \
    --cpu 8 \
    --no-cpu-throttling \
    --port 11434
```

#### 方案 B：GCE GPU VM + vLLM（高效能生產）

```
┌─────────────────────────────────────────────────┐
│         GCE VM (GPU: A100 / L4)                 │
│                                                  │
│  ┌─────────────────────────────────────────────┐│
│  │ Docker Compose                               ││
│  │                                              ││
│  │  ┌──────────┐     ┌──────────────────────┐  ││
│  │  │ CrewAI   │────▶│ vLLM Server          │  ││
│  │  │ FastAPI  │     │ Llama 3.3 70B (AWQ)  │  ││
│  │  │ :8080    │     │ OpenAI API :8000     │  ││
│  │  └──────────┘     └──────────────────────┘  ││
│  │                          │                   ││
│  │                    ┌─────▼─────┐             ││
│  │                    │ NVIDIA    │             ││
│  │                    │ A100 80GB │             ││
│  │                    └───────────┘             ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Llama-3.3-70B-Instruct
      --quantization awq
      --max-model-len 8192
      --gpu-memory-utilization 0.9
      --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  crewai:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=token-not-needed
      - OPENAI_API_BASE=http://vllm:8000/v1
    depends_on:
      - vllm
```

```bash
# GCE GPU VM 部署指令
# 1. 建立 GPU VM
gcloud compute instances create crewai-vllm-server \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --boot-disk-size=200GB \
    --image-family=common-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE

# 2. SSH 進入並啟動
gcloud compute ssh crewai-vllm-server --zone=us-central1-a

# 3. 安裝 Docker + NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. 啟動服務
docker compose up -d
```

#### 方案 C：GKE + vLLM（大規模生產）

```yaml
# k8s/vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-3.3-70B-Instruct"
        - "--quantization"
        - "awq"
        - "--max-model-len"
        - "8192"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "80Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "40Gi"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: vllm
```

CrewAI 的 Agent 只需指向內部服務：

```python
llm = LLM(
    model="openai/meta-llama/Llama-3.3-70B-Instruct",
    base_url="http://vllm-service:8000/v1",
    api_key="not-needed",
)
```

### 11.6 GCP GPU 定價與成本對比

| GPU 型號 | 按需費用/小時 | 月費（24/7） | 適合模型 |
|---------|-------------|-------------|---------|
| **NVIDIA T4** (16 GB) | ~$0.35 | ~$252 | 7-8B 模型 |
| **NVIDIA L4** (24 GB) | ~$0.70 | ~$504 | 8-24B 模型 |
| **NVIDIA A100** (40 GB) | ~$3.00 | ~$2,160 | 70B 量化模型 |
| **NVIDIA A100** (80 GB) | ~$4.00 | ~$2,880 | 70B 完整模型 |
| **NVIDIA H100** (80 GB) | ~$11.00 | ~$7,920 | 大型模型、高吞吐量 |

> Spot/搶佔式 VM 可降價 60-91%。A100 Spot 約 $1.0-1.5/hr，T4 Spot 約 $0.11/hr。

### 11.7 成本損益分析

**場景：每天 500 次 Crew 執行（每次約 5,000 token input + 2,000 token output）**

| 方案 | 月費 | 年費 |
|------|------|------|
| **GPT-4o API** | ~$1,500-2,500 | ~$18,000-30,000 |
| **Claude Sonnet API** | ~$1,200-2,000 | ~$14,400-24,000 |
| **A100 自架 Llama 70B** | ~$2,160（固定） | ~$25,920 |
| **A100 Spot 自架** | ~$720-1,080 | ~$8,640-12,960 |
| **L4 自架 Llama 8B** | ~$504（固定） | ~$6,048 |
| **L4 Spot 自架** | ~$150-250 | ~$1,800-3,000 |

**損益轉折點**：

- **低用量（<$500/月 API）**：用商用 API 更划算，不值得自架
- **中用量（$500-2,000/月 API）**：自架開始有優勢，特別是用 Spot VM
- **高用量（>$2,000/月 API）**：自架幾乎必然更便宜，且無 rate limit

### 11.8 混合策略（推薦）

**最佳實踐：不需要全部替換，用混合模型策略：**

```python
from crewai import Agent, LLM

# 簡單任務用小型開源模型（成本極低）
simple_llm = LLM(model="ollama/llama3.2:8b", base_url="http://ollama:11434")

# 複雜推理用大型開源模型
reasoning_llm = LLM(
    model="openai/meta-llama/Llama-3.3-70B-Instruct",
    base_url="http://vllm:8000/v1",
    api_key="not-needed",
)

# 最高品質輸出用商用模型
premium_llm = LLM(model="gpt-4o")

# 分配策略
researcher = Agent(role="Researcher", llm=simple_llm, ...)      # 大量搜尋：用便宜模型
analyst = Agent(role="Analyst", llm=reasoning_llm, ...)          # 分析推理：用中等模型
writer = Agent(role="Writer", llm=premium_llm, ...)              # 最終輸出：用頂級模型
```

這種混合策略可以在保持輸出品質的同時，**降低 60-80% 的 LLM 成本**。

### 11.9 注意事項

| 項目 | 說明 |
|------|------|
| **Tool Calling** | 並非所有開源模型都支援 function calling，建議用 Llama 3.3、Mistral、Qwen 2.5 等有完整工具支援的模型 |
| **冷啟動** | GPU VM 需要 1-5 分鐘載入模型，建議保持常駐或用 Spot 搶佔式策略 |
| **品質差異** | 8B 模型在複雜推理上顯著弱於 GPT-4o，建議先測試 |
| **繁中支援** | Qwen 2.5 對繁體中文支援最佳；Llama 3.3 次之 |
| **量化損失** | 4-bit 量化在簡單任務上品質損失 <5%，複雜推理可能 10-15% |
| **維運成本** | 自架需要人力維護 GPU 驅動、模型更新、監控 |

---

## 參考來源

- [CrewAI GitHub Repository](https://github.com/crewAIInc/crewAI)
- [CrewAI Official Documentation](https://docs.crewai.com)
- [A Detailed Comparison of Top 6 AI Agent Frameworks in 2026 - Turing](https://www.turing.com/resources/ai-agent-frameworks)
- [AI Agent Frameworks Compared (2026) - Arsum](https://arsum.com/blog/posts/ai-agent-frameworks/)
- [Open Source AI Agent Frameworks Compared - OpenAgents](https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared)
- [AutoGen vs CrewAI vs LangGraph vs OpenAI - Galileo](https://galileo.ai/blog/autogen-vs-crewai-vs-langgraph-vs-openai-agents-framework)
- [OpenAI Agents SDK vs LangGraph vs Autogen vs CrewAI - Composio](https://composio.dev/blog/openai-agents-sdk-vs-langgraph-vs-autogen-vs-crewai)
- [CrewAI vs Phidata (Agno) - nolist.ai](https://nolist.ai/compare/crewai-vs-phidata-agno)
- [Agno vs CrewAI - Keywords AI](https://www.keywordsai.co/market-map/compare/agno-vs-crewai)
- [Top 10 Most Starred AI Agent Frameworks on GitHub (2026) - Medium](https://techwithibrahim.medium.com/top-10-most-starred-ai-agent-frameworks-on-github-2026-df6e760a950b)
- [Definitive Guide to Agentic Frameworks in 2026 - SoftMax Data](https://blog.softmaxdata.com/definitive-guide-to-agentic-frameworks-in-2026-langgraph-crewai-ag2-openai-and-more/)
- [Top 7 Agentic AI Frameworks in 2026 - AlphaMatch](https://www.alphamatch.ai/blog/top-agentic-ai-frameworks-2026)
- [Top 10 Agentic AI Frameworks In 2026 - Aitude](https://www.aitude.com/top-agentic-ai-frameworks-2026/)
- [How to Deploy CrewAI to Production - DEV Community](https://dev.to/vhalasi/how-to-deploy-crewai-to-production-445f)
- [How to Deploy CrewAI to Production - Crewship](https://www.crewship.dev/blog/deploy-crewai-to-production)
- [CrewAI Deployment Guide: Production Implementation - Wednesday](https://www.wednesday.is/writing-articles/crewai-deployment-guide-production-implementation)
- [Google Gemini CrewAI Quickstart](https://github.com/google-gemini/crewai-quickstart)
- [Unlocking Multi-Agent A2A on Google Cloud - Google Dev](https://discuss.google.dev/t/unlocking-multi-agent-a2a-how-to-connect-crewai-and-adk-on-google-cloud/265858)
- [Quickstart: Deploy FastAPI to Cloud Run - Google Cloud](https://docs.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-fastapi-service)
- [KAgent: Deploying Custom AI Agents from CrewAI](https://kagent.dev/blog/crewai-byo-agent)
- [CrewAI LLM Connections 官方文件](https://docs.crewai.com/en/learn/llm-connections)
- [Self-Hosted LLM Guide: Setup, Tools & Cost Comparison (2026)](https://blog.premai.io/self-hosted-llm-guide-setup-tools-cost-comparison-2026/)
- [Ollama vs llama.cpp vs vLLM: 2026 Comparison](https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide)
- [vLLM vs Ollama - Northflank](https://northflank.com/blog/vllm-vs-ollama-and-how-to-run-them)
- [Run LLM on Cloud Run GPUs with Ollama - Google Cloud](https://docs.cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama)
- [GCP GPU Pricing](https://cloud.google.com/compute/gpus-pricing)
- [Cloud GPU Pricing Comparison 2026 - Nerd Level Tech](https://nerdleveltech.com/cloud-gpu-pricing-comparison-2026-aws-vs-gcp-vs-azure-for-ai-training)
- [Guide to Local LLMs in 2026 - SitePoint](https://www.sitepoint.com/definitive-guide-local-llms-2026-privacy-tools-hardware/)
- [Best Open Source LLMs 2026 - Contabo](https://contabo.com/blog/open-source-llms/)
