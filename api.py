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
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from fastapi import File, UploadFile
import httpx
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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

@tool
def search_knowledge_base(query: str) -> str:
    """检索企业内部的规范文档上下文。当用户询问与公司制度、技术规范、业务流程等文档相关的问题时，使用此工具获取参考信息。"""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

# 注意：开启流式输出 streaming=True
llm = ChatTongyi(model="qwen-turbo", temperature=0.1, streaming=True)
llm_with_tools = llm.bind_tools([search_knowledge_base])

# ==========================================
# 2. 定义 LangGraph 逻辑 (保持不变)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    intent: str
    search_query: str

def rewrite_query_node(state: AgentState):
    """查询改写节点：结合历史对话将最新提问改写为独立查询语句"""
    messages = state["messages"]
    latest_message = messages[-1].content

    history_str = ""
    for msg in messages[:-1]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        history_str += f"{role}: {msg.content}\n"

    if history_str.strip():
        prompt = f"""以下是历史对话记录：
{history_str}
用户最新提问：{latest_message}

请将用户的最新提问改写为一个完整的、独立的问题，使其脱离上下文也能被准确理解。只输出改写后的问题，不要输出其他内容。"""
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
    else:
        rewritten = latest_message

    return {"search_query": rewritten}

def retrieve_node(state: AgentState):
    search_query = state["search_query"]
    docs = retriever.invoke(search_query)
    context_str = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context_str}

def router_node(state: AgentState):
    """意图识别节点：判断用户输入是'文档问答'还是'日常寒暄'"""
    user_input = state["messages"][-1].content
    prompt = f"""请判断以下用户输入的意图类别，只能从以下两个选项中选择一个：
- 文档问答：用户在询问与文档、知识库相关的问题
- 日常寒暄：用户在进行打招呼、闲聊等非知识库相关的对话

用户输入：{user_input}

请只输出类别名称（文档问答 或 日常寒暄），不要输出其他内容。"""
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip()
    if "寒暄" in intent:
        intent = "日常寒暄"
    else:
        intent = "文档问答"
    return {"intent": intent}

def direct_answer_node(state: AgentState):
    """日常寒暄处理节点：直接给出友好回复"""
    messages = state["messages"]
    system_prompt = "你是一个友好的 AI 助手。用户正在进行日常寒暄或闲聊，请自然、简洁地回应。"
    conversation = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(conversation)
    return {"messages": [response]}

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

def route_intent(state: AgentState) -> str:
    """根据意图识别结果决定路由方向"""
    if state.get("intent") == "日常寒暄":
        return "direct_answer"
    return "rewrite_query"

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("direct_answer", direct_answer_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_intent, {"rewrite_query": "rewrite_query", "direct_answer": "direct_answer"})
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("direct_answer", END)



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
    temp_file_path = f"temp_{file.filename}"
    try:
        # 1. 将上传的文件暂存到本地
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        api_key = os.environ.get("DASHSCOPE_API_KEY")

        # 2. 上传至阿里云百炼获取 file_id
        upload_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/files"
        headers_upload = {"Authorization": f"Bearer {api_key}"}
        data = {"purpose": "file-extract"}

        # ⭐️ 关键修复 1：设置 300 秒（5分钟）的超长超时时间，防止大模型解析超时
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(temp_file_path, "rb") as f:
                files = {"file": (file.filename, f)}
                response = await client.post(upload_url, headers=headers_upload, data=data, files=files)
                upload_res = response.json()
                
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
            chat_res = (await client.post(chat_url, headers=headers_chat, json=payload)).json()
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

        return {
            "status": "success", 
            "message": f"文件 {file.filename} 解析入库成功！共新增 {len(final_splits)} 个知识块。"
        }

    except Exception as e:
        # ⭐️ 关键修复 2：打印详细堆栈到终端
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # ⭐️ 关键修复 3：无论成功还是失败，都确保临时文件被清理
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
# 3. 路由定义
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式输出接口：使用 Server-Sent Events (SSE) 逐字推流
    """
    async def event_generator():
        config = {"configurable": {"thread_id": request.session_id}}
        input_message = HumanMessage(content=request.query)
        
        # ⭐️ 核心改进：使用异步上下文管理器动态连接数据库并编译图
        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
            graph_app = workflow.compile(checkpointer=memory)
            
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