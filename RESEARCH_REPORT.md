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
