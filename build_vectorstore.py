import os
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
load_dotenv()
# 替换为你的通义千问 API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_API_KEY"] = api_key
MD_FILE_PATH = "GB47372-2026.md"
# ==========================================
# 第一步：读取本地 Markdown 文件
# ==========================================
print("1. 正在读取 Markdown 文件...")
with open(MD_FILE_PATH, "r", encoding="utf-8") as f:
    markdown_document = f.read()

# ==========================================
# 第二步：基于 Markdown 标题进行语义切片
# ==========================================
print("2. 正在按标题进行语义切片...")
# 定义我们要基于哪些级别的标题进行切分，并将标题名映射为 Metadata 的 Key
headers_to_split_on = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False # 保留文本块中的标题文字，防止丢失上下文
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# 【关键防护策略】：二次切分
# 有些章节（比如包含长表格或大段说明的 "5.2 标识和警示说明"）在按标题切分后，可能依然超过大模型最适宜的上下文长度。
# 所以我们需要用 RecursiveCharacterTextSplitter 兜底，按固定字符长度再切一刀。
chunk_size = 600
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap
)
# 这一步会自动继承之前按标题切分时生成的 Metadata
final_splits = text_splitter.split_documents(md_header_splits)

print(f"   切片完成！共切分出 {len(final_splits)} 个文本块 (Chunks)。")
# 打印第一个 Chunk 看看效果
print("\n--- 示例 Chunk ---")
print(f"内容: {final_splits[0].page_content[:100]}...")
print(f"元数据 (Metadata): {final_splits[0].metadata}")
print("------------------\n")

# ==========================================
# 第三步：调用 Embedding API 并存入 FAISS
# ==========================================
print("3. 正在向量化并存入 FAISS 数据库 (需要调用 API，请稍候)...")
# 初始化通义千问的 Embedding 模型
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

# 创建 FAISS 向量索引
vectorstore = FAISS.from_documents(
    documents=final_splits,
    embedding=embeddings
)

# 保存到本地，以便后续 RAG 检索时直接加载
vectorstore.save_local("faiss_gb47372_index")
print("4. 数据入库成功！已保存至当前目录的 'faiss_gb47372_index' 文件夹。")