import requests
import json

url = "http://127.0.0.1:8000/chat/stream"
payload = {
    "query": "移动电源发生热失控的判定条件是什么？",
    "session_id": "user_streaming_001"
}

print("正在请求流式接口...\n🤖 AI: ", end="")

# 关键：开启 stream=True，维持长连接
with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            # 解析 SSE 格式 (去掉开头的 "data: ")
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                if data_str == "[DONE]":
                    break
                # 解析 JSON 并平滑打印
                data_json = json.loads(data_str)
                print(data_json["chunk"], end="", flush=True)

print("\n\n✅ 流式接收完成！")