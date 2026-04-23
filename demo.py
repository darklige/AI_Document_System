import requests
import os
from dotenv import load_dotenv

# 1. 加载 .env 文件中的环境变量
load_dotenv()
# 替换为你的通义千问 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")
FILE_PATH = "GB+47372-2026.pdf"

# ==========================================
# 第一步：将 PDF 文件上传至阿里云百炼服务器
# ==========================================
upload_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/files"
headers_upload = {
    "Authorization": f"Bearer {API_KEY}"
}
data = {"purpose": "file-extract"}
files = {"file": open(FILE_PATH, "rb")}

print("1. 正在上传 PDF 文件...")
upload_response = requests.post(upload_url, headers=headers_upload, files=files, data=data)
upload_res_json = upload_response.json()

if "id" not in upload_res_json:
    print("上传失败：", upload_res_json)
    exit()

file_id = upload_res_json["id"]
print(f"   上传成功！获取到 File ID: {file_id}")

# ==========================================
# 第二步：调用 qwen-long 提取 Markdown
# ==========================================
chat_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
headers_chat = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 构造 payload
payload = {
    "model": "qwen-long",  # 切换为最稳定支持 fileid 语法的 qwen-long
    "messages": [
        {
            "role": "system",
            "content": "你是一个严谨的结构化数据提取工程师。你的任务是提取用户上传文档的全部内容，并严格按照 Markdown 格式输出。请务必保留原文档的多级标题层级（#）、表格结构（使用 Markdown 表格语法）以及列表编号，不要遗漏任何关键的测试参数，不要输出任何前置或后置的客套话。"
        },
        {
            "role": "system",
            # 【关键修复】：fileid 必须是一个独立的 system message！
            "content": f"fileid://{file_id}"
        },
        {
            "role": "user",
            # user 里面只放纯粹的要求
            "content": "请全面解析我上传的这份《移动电源安全技术规范》文档全文。"
        }
    ]
}

print("2. 正在由大模型解析并生成 Markdown (国标文件可能需要几十秒，请耐心等待)...")
chat_response = requests.post(chat_url, headers=headers_chat, json=payload)
chat_res_json = chat_response.json()

if "choices" not in chat_res_json:
    print("解析失败：", chat_res_json)
    exit()

markdown_content = chat_res_json["choices"][0]["message"]["content"]

# ==========================================
# 第三步：将 Markdown 结果保存到本地
# ==========================================
output_filename = "GB47372-2026.md"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"3. 解析完成！已成功保存至 {output_filename}")