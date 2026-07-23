const API_BASE = '/api/v1';

let sessionId = null;
let isLoading = false;

const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const typingIndicator = document.getElementById('typing-indicator');
const sessionStatus = document.getElementById('session-status');
const emotionPanel = document.getElementById('emotion-panel');
const closeEmotionBtn = document.getElementById('close-emotion-btn');

const emotionTypeEl = document.getElementById('emotion-type');
const intensityFill = document.getElementById('intensity-fill');
const intensityValue = document.getElementById('intensity-value');
const therapyStageEl = document.getElementById('therapy-stage');

const stageNames = {
    'rapport_building': '建立关系',
    'assessment': '评估问题',
    'goal_setting': '设定目标',
    'intervention': '干预治疗',
    'closure': '结束阶段'
};

const emotionNames = {
    'anxiety': '焦虑',
    'depression': '抑郁',
    'anger': '愤怒',
    'sadness': '悲伤',
    'fear': '恐惧',
    'stress': '压力',
    'neutral': '平静',
    'hope': '希望',
    'relief': '释然',
    'confusion': '困惑'
};

function formatTime(date) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function addMessage(content, isUser) {
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${isUser ? '👤' : '🤖'}</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(content)}</div>
            <div class="message-time">${formatTime(new Date())}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    typingIndicator.classList.add('show');
    scrollToBottom();
}

function hideTyping() {
    typingIndicator.classList.remove('show');
}

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    messageInput.disabled = loading;
}

function updateSessionStatus() {
    if (sessionId) {
        sessionStatus.textContent = `会话: ${sessionId}`;
    } else {
        sessionStatus.textContent = '会话: 未开始';
    }
}

function updateEmotionPanel(data) {
    if (!data || !data.emotion_analysis) return;
    
    const emotion = data.emotion_analysis;
    const emotionName = emotionNames[emotion.primary_emotion] || emotion.primary_emotion;
    const intensity = emotion.intensity || 0;
    const stage = stageNames[data.therapy_stage] || data.therapy_stage;
    
    emotionTypeEl.textContent = emotionName;
    intensityFill.style.width = `${intensity * 10}%`;
    intensityValue.textContent = intensity;
    therapyStageEl.textContent = stage;
    
    emotionPanel.classList.add('show');
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;
    
    addMessage(message, true);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    setLoading(true);
    showTyping();
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId,
                enable_thought_chain: true,
                enable_emotion_analysis: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        sessionId = data.session_id;
        updateSessionStatus();
        
        hideTyping();
        addMessage(data.response, false);
        
        updateEmotionPanel(data);
        
    } catch (error) {
        console.error('Error:', error);
        hideTyping();
        addMessage('抱歉，发生了错误。请稍后重试。', false);
    } finally {
        setLoading(false);
    }
}

function startNewSession() {
    sessionId = null;
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">💬</div>
            <h2>欢迎使用心理咨询AI助手</h2>
            <p>我是一个专业的心理咨询AI，可以倾听你的烦恼，帮助你缓解压力。</p>
            <p>请放心，我们的对话是保密的。你可以随时开始分享你的感受。</p>
        </div>
    `;
    updateSessionStatus();
    emotionPanel.classList.remove('show');
}

messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);
newSessionBtn.addEventListener('click', startNewSession);
closeEmotionBtn.addEventListener('click', () => {
    emotionPanel.classList.remove('show');
});

messageInput.focus();
