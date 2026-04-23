import os
import json
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # 1. 必须引入这个中间件
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 全局初始化
# ==========================================
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")
os.environ["DASHSCOPE_API_KEY"] = api_key 

print("正在初始化 AI 组件...")
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = FAISS.load_local(
    folder_path="faiss_gb47372_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# 注意：开启流式输出 streaming=True
llm = ChatTongyi(model="qwen-turbo", temperature=0.1, streaming=True)

# ==========================================
# 2. 定义 LangGraph 逻辑 (保持不变)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

def retrieve_node(state: AgentState):
    latest_message = state["messages"][-1].content
    docs = retriever.invoke(latest_message)
    context_str = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context_str}

def generate_node(state: AgentState):
    messages = state["messages"]
    context = state["context"]
    system_prompt = f"""你是一个专业的企业知识库 AI 助手。
请严格基于以下提供的【参考文档上下文】来回答用户的问题。如果文档中没有相关信息，请直接回答不知道。

【参考文档上下文】:
{context}
"""
    conversation = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(conversation)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# ==========================================
# 3. 搭建 FastAPI 服务
# ==========================================
app = FastAPI(title="智能文档问答系统 API", version="1.0")

# 2. 关键修复：配置 CORS 中间件，必须放在路由定义之前！
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有前端地址跨域访问
    allow_credentials=True,
    allow_methods=["*"], # 允许所有请求方法，包括 OPTIONS、GET、POST 等
    allow_headers=["*"], # 允许所有自定义请求头
)

# 定义数据模型
class ChatRequest(BaseModel):
    query: str
    session_id: str

# 3. 路由定义
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式输出接口：使用 Server-Sent Events (SSE) 逐字推流
    """
    async def event_generator():
        config = {"configurable": {"thread_id": request.session_id}}
        input_message = HumanMessage(content=request.query)
        
        async for event in graph_app.astream_events({"messages": [input_message]}, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'chunk': content}, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("启动 FastAPI 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)