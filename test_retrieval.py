import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

# ==========================================
# 1. 加载环境变量与 API Key
# ==========================================
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件！")

# 兼容 LangChain 底层调用
os.environ["DASHSCOPE_API_KEY"] = api_key 

# ==========================================
# 2. 实例化 Embedding 模型并加载本地向量库
# ==========================================
print("1. 正在初始化 Embedding 模型...")
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

print("2. 正在加载本地 FAISS 向量库...")
# 注意：新版 LangChain 出于安全考虑，读取本地 pickle 文件必须显式开启 allow_dangerous_deserialization=True
vectorstore = FAISS.load_local(
    folder_path="faiss_gb47372_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True 
)

# ==========================================
# 3. 执行检索测试
# ==========================================
# 设定一个针对这份国标文档的具体问题
query = "移动电源的充放电温度控制有什么要求？或者过温保护怎么触发？"
print(f"\n👉 测试问题: '{query}'\n")

print("3. 正在向量数据库中进行语义检索...\n")
# k=3 表示召回最相似的前 3 个文本块 (Chunks)
results = vectorstore.similarity_search_with_score(query, k=3)

# ==========================================
# 4. 打印召回结果
# ==========================================
print("-" * 50)
for i, (doc, score) in enumerate(results):
    print(f"【召回文档 {i+1}】 (向量距离 L2 Score: {score:.4f}，越小越相似)")
    print(f"📍 来源章节 (Metadata): {doc.metadata}")
    print(f"📄 文本内容: \n{doc.page_content.strip()}")
    print("-" * 50)

print("\n💡 检索测试完成。你可以尝试修改 'query' 变量，测试其他具体的指标问题（例如挤压测试的参数、外壳阻燃要求等）。")