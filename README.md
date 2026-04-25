# 智能文档问答系统 (企业知识库) 🧠

这是一个基于 RAG (检索增强生成) 架构构建的企业级智能文档问答系统。系统集成了文档解析、语义切片、向量检索、多轮对话状态管理以及前端流式渲染，能够将各类复杂的规范文档（如带有表格、多级标题的 PDF/Word）转化为可交互的动态知识库。

## ✨ 核心特性

- **📄 动态知识库扩容**：支持前端一键上传本地文档，后端自动调用大模型进行版面分析与 Markdown 结构化提取，实时切片并追加至向量数据库。
- **🧠 LangGraph 多轮记忆**：底层采用图状态机 (StateGraph) 管理对话上下文，精准理解用户的指代与连续提问。
- **⚡ 流式输出体验 (SSE)**：打通后端大模型流式响应与前端 Server-Sent Events，实现丝滑的“打字机”实时输出效果。
- **📊 智能切片与 Markdown 渲染**：依据文档标题层级进行语义切块 (Semantic Chunking)，保留表格与列表的上下文结构；前端内置 `markdown-it`，完美还原复杂国标表格与技术参数。

## 🛠️ 技术栈

### 后端 (Backend)
- **Web 框架**: FastAPI
- **AI 编排**: LangChain, LangGraph
- **向量数据库**: FAISS (本地化高性能检索)
- **大模型接入**: 阿里云通义千问 (DashScope)
  - 对话模型：`qwen-turbo`
  - 复杂文档解析：`qwen-long`
  - 向量化模型：`text-embedding-v3`

### 前端 (Frontend)
- **框架**: Vue 3 (Composition API, `<script setup>`)
- **构建工具**: Vite
- **渲染引擎**: `markdown-it`

---

## 📂 项目结构

```text
.
├── ai-knowledge-base/         # Vue 3 前端工程目录
│   ├── src/
│   │   ├── App.vue            # 核心对话与上传组件
│   │   └── ...
│   ├── package.json           # 前端依赖配置
│   └── vite.config.js         # Vite 配置文件
├── faiss_gb47372_index/       # FAISS 向量数据库持久化目录
│   ├── index.faiss
│   └── index.pkl
├── api.py                     # FastAPI 后端主服务程序
├── .env                       # 环境变量配置文件 (存放 API Key)
├── .gitignore                 # Git 忽略配置
└── README.md                  # 项目说明文档
🚀 快速启动
1. 环境准备
确保已安装 Python 3.10+。

确保已安装 Node.js (LTS 版本)。

准备好阿里云百炼 (DashScope) 的 API Key。

2. 后端服务部署
在项目根目录下，安装必要的 Python 依赖：

Bash
pip install fastapi uvicorn pydantic python-multipart requests python-dotenv langchain langchain-community langchain-text-splitters langgraph faiss-cpu dashscope
创建 .env 文件并填入你的 API Key：

代码段
DASHSCOPE_API_KEY=sk-你的通义千问APIKEY
启动 FastAPI 后端服务：

Bash
python api.py
后端服务将运行在 http://0.0.0.0:8000

3. 前端服务部署
新开一个终端窗口，进入前端工程目录：

Bash
cd ai-knowledge-base
安装前端依赖：

Bash
npm install
启动 Vite 开发服务器：

Bash
npm run dev
打开浏览器访问 http://localhost:5173/ 即可体验系统。

💡 使用指南
查阅现有知识：系统启动后，直接在底部的输入框中提问（例如：“热失控的判定条件是什么？”）。

上传新文档：点击页面右上角的 ➕ 上传新知识文档 按钮，选择本地的 PDF 或 Word 文件。系统会在后台自动完成解析、切片与入库，完成后即可对新文档进行提问。

📅 未来演进方向
[ ] 将底层 FAISS 替换为 Milvus 或 Qdrant 以支持十亿级向量的高并发检索。

[ ] 引入 OCR 多模态模型 (如 Qwen-VL)，实现对复杂电路图和纯图片 PDF 的问答支持。

[ ] 增加用户鉴权模块 (JWT) 与历史会话的云端持久化存储 (MySQL/PostgreSQL)。