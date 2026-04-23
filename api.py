import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
from fastapi.responses import StreamingResponse
# ==========================================
# 1. 全局初始化 (服务启动时只加载一次)
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
llm = ChatTongyi(model="qwen-turbo", temperature=0.1,streaming=True)

# ==========================================
# 2. 定义 LangGraph 逻辑
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
# 编译为全局可用的 app
graph_app = workflow.compile(checkpointer=memory)

# ==========================================
# 3. 搭建 FastAPI 服务
# ==========================================
app = FastAPI(title="智能文档问答系统 API", version="1.0")

# 定义前端传过来的数据格式
class ChatRequest(BaseModel):
    query: str
    session_id: str  # 让前端传 session_id，用于区分不同用户的多轮对话

# 定义返回给前端的数据格式
class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 使用前端传来的 session_id 作为 LangGraph 的 thread_id
        config = {"configurable": {"thread_id": request.session_id}}
        input_message = HumanMessage(content=request.query)
        
        # 触发图执行
        result = graph_app.invoke({"messages": [input_message]}, config=config)
        
        # 提取最终大模型的回答
        final_ai_message = result["messages"][-1].content
        
        return ChatResponse(
            answer=final_ai_message,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式输出接口：使用 Server-Sent Events (SSE) 逐字推流
    """
    async def event_generator():
        # 1. 配置上下文和输入
        config = {"configurable": {"thread_id": request.session_id}}
        input_message = HumanMessage(content=request.query)
        
        # 2. 核心：使用 astream_events 获取底层大模型的每一次吐字
        # version="v2" 是 LangChain 官方推荐的事件流 API 版本
        async for event in graph_app.astream_events({"messages": [input_message]}, config=config, version="v2"):
            kind = event["event"]
            
            # 我们只关心 "聊天模型正在输出流" 这个事件
            if kind == "on_chat_model_stream":
                # 提取出当前的字/词
                content = event["data"]["chunk"].content
                if content:
                    # 3. 按照标准的 SSE 格式 (data: {json}\n\n) 通过 yield 实时推给前端
                    yield f"data: {json.dumps({'chunk': content}, ensure_ascii=False)}\n\n"
        
        # 4. 生成结束，发送结束标识，方便前端断开连接
        yield "data: [DONE]\n\n"

    # 使用 FastAPI 的 StreamingResponse，指定媒体类型为 event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")
if __name__ == "__main__":
    print("启动 FastAPI 服务...")
    # 运行在 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)