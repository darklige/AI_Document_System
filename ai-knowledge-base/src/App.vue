<template>
  <div class="chat-container">
    <div class="header">
      <h2>企业知识库智能问答 (Markdown 版)</h2>
      <div class="upload-area">
        <input 
          type="file" 
          ref="fileInputRef" 
          @change="handleFileUpload" 
          accept=".pdf,.docx,.txt" 
          style="display: none;" 
        />
        <button 
          class="upload-btn" 
          @click="triggerFileInput" 
          :disabled="isUploading"
        >
          {{ isUploading ? '📄 正在解析入库...' : '➕ 上传新知识文档' }}
        </button>
      </div>
    </div>
    <div class="message-list" ref="messageListRef">
      <div 
        v-for="(msg, index) in messages" 
        :key="index" 
        :class="['message-wrapper', msg.role]"
      >
        <div 
          class="message-bubble markdown-body" 
          v-html="renderMarkdown(msg.content)"
        ></div>
      </div>
    </div>

    <div class="input-area">
      <input 
        v-model="userInput" 
        @keyup.enter="sendMessage"
        type="text" 
        placeholder="输入您的问题，例如：国标中关于挤压测试的参数表格是什么？" 
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
import MarkdownIt from 'markdown-it';

// 初始化 Markdown 解析器，开启自动换行和链接识别
const md = new MarkdownIt({ breaks: true, linkify: true });

// 渲染函数：如果内容为空，给一个默认的光标提示
const renderMarkdown = (text) => {
  if (!text) return '<span class="cursor-blink">▌</span>';
  return md.render(text);
};

// 状态定义
const userInput = ref('');
const isLoading = ref(false);
const messageListRef = ref(null);
const sessionId = 'vue_test_md_01'; 
const fileInputRef = ref(null);
const isUploading = ref(false);
const triggerFileInput = () => {
  fileInputRef.value.click();
};
const messages = ref([
  { role: 'ai', content: '您好！我已经学习了 **《GB 47372-2026 移动电源安全技术规范》**。\n\n您可以问我关于:\n- 🔋 充放电性能指标\n- 🔥 热失控判定条件\n- 📊 挤压测试参数表格\n\n请问有什么可以帮您？' }
]);

const scrollToBottom = async () => {
  await nextTick();
  if (messageListRef.value) {
    messageListRef.value.scrollTop = messageListRef.value.scrollHeight;
  }
};
// 处理文件上传的核心函数
const handleFileUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  isUploading.value = true;
  // 可以在聊天框里给用户一个提示
  messages.value.push({ 
    role: 'ai', 
    content: `正在后台阅读并学习文件：**${file.name}**，国标/长文档解析可能需要 30-60 秒，请稍候...` 
  });
  scrollToBottom();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://127.0.0.1:8000/upload', {
      method: 'POST',
      body: formData // 注意：使用 FormData 时，fetch 会自动设置正确的 Content-Type
    });

    const result = await response.json();
    
    if (response.ok) {
      messages.value.push({ 
        role: 'ai', 
        content: `✅ **学习完毕！**\n\n${result.message}\n\n现在您可以基于这份新文档向我提问了。` 
      });
    } else {
      throw new Error(result.detail || '上传解析失败');
    }
  } catch (error) {
    console.error('上传异常:', error);
    messages.value.push({ role: 'ai', content: `❌ 文件学习失败：${error.message}` });
  } finally {
    isUploading.value = false;
    event.target.value = ''; // 清空 input，允许重复上传同一个文件
    scrollToBottom();
  }
};
const sendMessage = async () => {
  const query = userInput.value.trim();
  if (!query) return;

  messages.value.push({ role: 'user', content: query });
  userInput.value = '';
  isLoading.value = true;
  scrollToBottom();

  messages.value.push({ role: 'ai', content: '' });
  const aiMessageIndex = messages.value.length - 1;

  try {
    const response = await fetch('http://127.0.0.1:8000/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, session_id: sessionId })
    });

    if (!response.ok) throw new Error('网络请求失败');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunkText = decoder.decode(value, { stream: true });
      const lines = chunkText.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          if (dataStr === '[DONE]') break;

          try {
            const dataObj = JSON.parse(dataStr);
            if (dataObj.chunk) {
              messages.value[aiMessageIndex].content += dataObj.chunk;
              scrollToBottom();
            }
          } catch (e) {
            console.warn('解析忽略:', dataStr);
          }
        }
      }
    }
  } catch (error) {
    console.error('API 异常:', error);
    messages.value[aiMessageIndex].content = '抱歉，服务出现异常，请检查后端服务是否启动。';
  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
/* 基础布局保持不变 */
.chat-container { max-width: 800px; margin: 20px auto; border: 1px solid #e0e0e0; border-radius: 8px; display: flex; flex-direction: column; height: 85vh; font-family: -apple-system, sans-serif; }
.header { padding: 16px; background-color: #f5f5f5; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: bold; }
.message-list { flex: 1; padding: 20px; overflow-y: auto; background-color: #f9f9f9; }
.message-wrapper { display: flex; margin-bottom: 20px; }
.message-wrapper.user { justify-content: flex-end; }
.message-bubble { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; }
.message-wrapper.user .message-bubble { background-color: #1890ff; color: white; border-bottom-right-radius: 2px; }
.message-wrapper.ai .message-bubble { background-color: white; border: 1px solid #e0e0e0; color: #333; border-bottom-left-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.input-area { padding: 16px; background-color: white; border-top: 1px solid #e0e0e0; display: flex; gap: 10px; }
input { flex: 1; padding: 12px; border: 1px solid #d9d9d9; border-radius: 6px; font-size: 1rem; outline: none; }
button { padding: 0 24px; background-color: #1890ff; color: white; border: none; border-radius: 6px; cursor: pointer; transition: 0.3s; }
button:disabled { background-color: #bae0ff; cursor: not-allowed; }
/* 补充的上传按钮样式 */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
}
.upload-btn {
  background-color: #52c41a;
  padding: 8px 16px;
  font-size: 0.9rem;
}
.upload-btn:hover {
  background-color: #73d13d;
}
.upload-btn:disabled {
  background-color: #b7eb8f;
}
/* 闪烁的光标动画 */
.cursor-blink { animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* ============= Markdown 核心样式 =============
  注意：在 Vue 的 <style scoped> 中，动态注入的 HTML 节点需要用 :deep() 穿透才能生效！
*/
.message-wrapper.ai :deep(h1), 
.message-wrapper.ai :deep(h2), 
.message-wrapper.ai :deep(h3) {
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  padding-bottom: 0.3em;
  border-bottom: 1px solid #eaecef;
}

.message-wrapper.ai :deep(p) {
  margin-bottom: 1em;
}

.message-wrapper.ai :deep(ul), 
.message-wrapper.ai :deep(ol) {
  padding-left: 20px;
  margin-bottom: 1em;
}

/* 重点：让国标里的表格漂漂亮亮的 */
.message-wrapper.ai :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 1em;
  font-size: 0.9em;
}
.message-wrapper.ai :deep(th), 
.message-wrapper.ai :deep(td) {
  border: 1px solid #dfe2e5;
  padding: 8px 12px;
}
.message-wrapper.ai :deep(th) {
  background-color: #f6f8fa;
  font-weight: 600;
}
.message-wrapper.ai :deep(tr:nth-child(2n)) {
  background-color: #fbfbfc;
}

/* 代码块和强调 */
.message-wrapper.ai :deep(code) {
  background-color: rgba(27,31,35,0.05);
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: monospace;
}
.message-wrapper.ai :deep(pre) {
  background-color: #f6f8fa;
  padding: 16px;
  overflow: auto;
  border-radius: 6px;
}
.message-wrapper.ai :deep(pre code) {
  background-color: transparent;
  padding: 0;
}
</style>