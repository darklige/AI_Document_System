<template>
  <div class="chat-container">
    <div class="header">
      <h2>企业知识库智能问答</h2>
    </div>

    <div class="message-list" ref="messageListRef">
      <div 
        v-for="(msg, index) in messages" 
        :key="index" 
        :class="['message-wrapper', msg.role]"
      >
        <div class="message-bubble">
          {{ msg.content }}
        </div>
      </div>
    </div>

    <div class="input-area">
      <input 
        v-model="userInput" 
        @keyup.enter="sendMessage"
        type="text" 
        placeholder="输入您的问题，例如：热失控的判定条件..." 
        :disabled="isLoading"
      />
      <button @click="sendMessage" :disabled="isLoading || !userInput.trim()">
        {{ isLoading ? '思考中...' : '发送' }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue';

// 状态定义
const userInput = ref('');
const isLoading = ref(false);
const messageListRef = ref(null);
const sessionId = 'vue_test_user_01'; // 模拟固定用户，保持多轮对话记忆

// 聊天记录数组：role 可选 'user' 或 'ai'
const messages = ref([
  { role: 'ai', content: '您好！我是企业知识库助手。我已经学习了《GB 47372-2026 移动电源安全技术规范》，请问有什么可以帮您？' }
]);

// 滚动到底部的辅助函数
const scrollToBottom = async () => {
  await nextTick();
  if (messageListRef.value) {
    messageListRef.value.scrollTop = messageListRef.value.scrollHeight;
  }
};

// 核心：发送消息并处理流式响应
const sendMessage = async () => {
  const query = userInput.value.trim();
  if (!query) return;

  // 1. 将用户问题加入界面，清空输入框
  messages.value.push({ role: 'user', content: query });
  userInput.value = '';
  isLoading.value = true;
  scrollToBottom();

  // 2. 预先推入一个空的 AI 消息，用于稍后逐字追加
  messages.value.push({ role: 'ai', content: '' });
  const aiMessageIndex = messages.value.length - 1;

  try {
    // 3. 发送 POST 请求到 FastAPI 流式接口
    const response = await fetch('http://127.0.0.1:8000/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, session_id: sessionId })
    });

    if (!response.ok) throw new Error('网络请求失败');

    // 4. 获取流式读取器并准备解码
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    // 5. 循环读取流数据
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // 解码当前数据块（可能包含多行 SSE 数据）
      const chunkText = decoder.decode(value, { stream: true });
      const lines = chunkText.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          
          if (dataStr === '[DONE]') {
            // 后端推流结束
            break;
          }

          try {
            // 解析 JSON 并追加到最后一条 AI 消息中
            const dataObj = JSON.parse(dataStr);
            if (dataObj.chunk) {
              messages.value[aiMessageIndex].content += dataObj.chunk;
              scrollToBottom(); // 每次吐字都保持页面在最底部
            }
          } catch (e) {
            console.warn('JSON 解析忽略此片段:', dataStr);
          }
        }
      }
    }
  } catch (error) {
    console.error('API 异常:', error);
    messages.value[aiMessageIndex].content = '抱歉，服务出现异常，请稍后再试。';
  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
/* 极简的聊天 UI 样式 */
.chat-container {
  max-width: 800px;
  margin: 20px auto;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  height: 80vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.header {
  padding: 16px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  text-align: center;
}

.header h2 {
  margin: 0;
  font-size: 1.2rem;
  color: #333;
}

.message-list {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background-color: #fafafa;
}

.message-wrapper {
  display: flex;
  margin-bottom: 16px;
}

.message-wrapper.user {
  justify-content: flex-end;
}

.message-bubble {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.5;
  white-space: pre-wrap; /* 保持后端返回的换行符 */
}

.message-wrapper.user .message-bubble {
  background-color: #1890ff;
  color: white;
  border-bottom-right-radius: 2px;
}

.message-wrapper.ai .message-bubble {
  background-color: white;
  border: 1px solid #e0e0e0;
  color: #333;
  border-bottom-left-radius: 2px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.input-area {
  padding: 16px;
  background-color: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
  border-bottom-left-radius: 8px;
  border-bottom-right-radius: 8px;
}

input {
  flex: 1;
  padding: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 1rem;
  outline: none;
}

input:focus {
  border-color: #1890ff;
}

button {
  padding: 0 24px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.3s;
}

button:hover {
  background-color: #40a9ff;
}

button:disabled {
  background-color: #bae0ff;
  cursor: not-allowed;
}
</style>