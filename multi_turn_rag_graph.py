import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 加载环境变量与初始化核心组件
# ==========================================
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")
os.environ["DASHSCOPE_API_KEY"] = api_key 

print("1. 正在初始化组件...")
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = FAISS.load_local(
    folder_path="faiss_gb47372_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatTongyi(model="qwen-turbo", temperature=0.1)

# ==========================================
# 2. 定义 LangGraph 的状态 (State)
# ==========================================
# State 是图的“血液”，在各个节点之间传递
class AgentState(TypedDict):
    # messages 列表中保存了所有的历史对话。add_messages 确保新消息是追加而不是覆盖
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # context 保存单次检索到的文档文本
    context: str

# ==========================================
# 3. 定义图的节点 (Nodes)
# ==========================================

def retrieve_node(state: AgentState):
    """节点 1：负责从向量库检索文档"""
    # 获取用户最新的一句话
    latest_message = state["messages"][-1].content
    print(f"\n[执行节点] Retrieve: 正在检索与 '{latest_message}' 相关的文档...")
    
    docs = retriever.invoke(latest_message)
    context_str = "\n\n".join(doc.page_content for doc in docs)
    
    # 将检索到的文档更新到 State 中的 context 字段
    return {"context": context_str}

def generate_node(state: AgentState):
    """节点 2：负责结合上下文和历史对话生成回答"""
    print("[执行节点] Generate: 正在请求大模型生成回答...")
    messages = state["messages"]
    context = state["context"]
    
    # 构造系统提示词，将检索到的背景知识注入其中
    system_prompt = f"""你是一个专业的企业知识库 AI 助手。
请严格基于以下提供的【参考文档上下文】来回答用户的问题。如果文档中没有相关信息，请直接回答不知道。

【参考文档上下文】:
{context}
"""
    # 将系统提示词放在对话历史的最前面，然后带上所有的对话历史请求大模型
    conversation = [SystemMessage(content=system_prompt)] + list(messages)
    response = llm.invoke(conversation)
    
    # 将大模型的回答追加到 State 的 messages 列表中
    return {"messages": [response]}

# ==========================================
# 4. 构建并编译图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 定义边（控制流转顺序）: 开始 -> 检索 -> 生成 -> 结束
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 初始化内存保存器（Checkpointer），这是多轮对话的核心！
memory = MemorySaver()

# 编译图表，注入记忆模块
app = workflow.compile(checkpointer=memory)

# ==========================================
# 5. 启动多轮对话交互终端
# ==========================================
print("\n" + "="*50)
print("🚀 LangGraph 智能文档问答系统已启动！")
print("可以开始连续提问了。输入 'quit' 或 'exit' 退出。")
print("="*50 + "\n")

# 定义一个唯一的 session_id，LangGraph 靠这个 ID 来隔离和读取不同用户的历史记录
config = {"configurable": {"thread_id": "session_123"}}

while True:
    user_input = input("🧑 你: ")
    if user_input.lower() in ['quit', 'exit']:
        print("👋 再见！")
        break
    if not user_input.strip():
        continue
        
    # 构造用户的消息结构
    input_message = HumanMessage(content=user_input)
    
    # 触发 LangGraph 流转
    result = app.invoke({"messages": [input_message]}, config=config)
    
    # result["messages"] 包含了图执行完毕后的所有历史消息，我们取最后一条打印
    final_ai_message = result["messages"][-1].content
    print(f"\n🤖 AI: {final_ai_message}\n")
    print("-" * 50)