import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 加载环境变量与 API Key
# ==========================================
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")
os.environ["DASHSCOPE_API_KEY"] = api_key 

# ==========================================
# 2. 初始化核心组件 (Embedding, 向量库, LLM)
# ==========================================
print("1. 正在初始化组件...")
# 加载 Embedding 和本地 FAISS
embeddings = DashScopeEmbeddings(model="text-embedding-v3")
vectorstore = FAISS.load_local(
    folder_path="faiss_gb47372_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)
# 将 FAISS 转化为检索器，每次召回 Top-3 最相关的片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 初始化通义千问对话大模型 (qwen-turbo 速度快，适合日常对话与生成)
llm = ChatTongyi(model="qwen-turbo", temperature=0.1)

# ==========================================
# 3. 定义 RAG 的提示词模板 (Prompt Template)
# ==========================================
# 明确告诉大模型：只能基于上下文回答，不知道就说不知道
template = """你是一个专业的企业知识库 AI 助手。
请严格基于以下提供的【参考文档上下文】来回答用户的问题。
如果上下文中没有包含能回答该问题的信息，请直接回答“根据提供的文档内容，我无法回答该问题”，绝不要编造任何内容。

【参考文档上下文】:
{context}

用户问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ==========================================
# 4. 构建 LCEL 核心执行链
# ==========================================
# 辅助函数：将检索到的多个 Document 对象中的文本提取出来拼接成一段长字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL 语法链：
# 1. RunnablePassthrough.assign 负责将 retriever 的结果丢给 format_docs 变成纯文本塞入 'context'
# 2. question 原样透传
# 3. 传给 prompt 生成最终提示词
# 4. 交给 llm 生成回答
# 5. StrOutputParser 提取大模型返回的纯文本字符串
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 5. 执行测试
# ==========================================
query = "移动电源发生热失控的判定条件是什么？"
print(f"\n👉 提问: {query}\n")
print("2. 正在检索知识并请求大模型生成回答，请稍候...\n")

# 调用 invoke 触发整个链条
response = rag_chain.invoke(query)

print("🤖 AI 回答:")
print("-" * 50)
print(response)
print("-" * 50)