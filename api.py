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
from fastapi import File, UploadFile
import requests
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    接收前端上传的文件，调用大模型解析，切片后动态追加到 FAISS 向量库
    """
    try:
        # 1. 将上传的文件暂存到本地
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        api_key = os.environ.get("DASHSCOPE_API_KEY")

        # 2. 上传至阿里云百炼获取 file_id
        upload_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/files"
        headers_upload = {"Authorization": f"Bearer {api_key}"}
        data = {"purpose": "file-extract"}
        files = {"file": open(temp_file_path, "rb")}
        
        upload_res = requests.post(upload_url, headers=headers_upload, files=files, data=data).json()
        if "id" not in upload_res:
            raise Exception(f"阿里云文件上传失败: {upload_res}")
        file_id = upload_res["id"]

        # 3. 调用 qwen-long 提取 Markdown
        chat_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers_chat = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "qwen-long",
            "messages": [
                {"role": "system", "content": "你是一个严谨的结构化数据提取工程师。请提取用户上传文档的全部内容，严格按 Markdown 格式输出。保留多级标题和表格。"},
                {"role": "system", "content": f"fileid://{file_id}"},
                {"role": "user", "content": "请全面解析这份文档。"}
            ]
        }
        chat_res = requests.post(chat_url, headers=headers_chat, json=payload).json()
        if "choices" not in chat_res:
            raise Exception(f"大模型解析失败: {chat_res}")
        markdown_content = chat_res["choices"][0]["message"]["content"]

        # 4. 语义切片
        headers_to_split_on = [("#", "一级标题"), ("##", "二级标题"), ("###", "三级标题")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(markdown_content)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        final_splits = text_splitter.split_documents(md_header_splits)

        # 5. 动态追加到现有的 FAISS 向量库中，并持久化到本地硬盘
        global vectorstore
        vectorstore.add_documents(final_splits)
        vectorstore.save_local("faiss_gb47372_index")

        # 6. 清理临时文件
        os.remove(temp_file_path)

        return {
            "status": "success", 
            "message": f"文件 {file.filename} 解析入库成功！共新增 {len(final_splits)} 个知识块。"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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