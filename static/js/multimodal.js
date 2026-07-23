/**
 * 多模态聊天前端脚本 (v3 — 重构版)
 * ======================================
 *
 * 功能：
 * 1. 文本消息发送（含超时控制、请求取消）
 * 2. 语音录制和识别 → 自动发送
 * 5. 语音回复播放（TTS）
 * 6. 暗色模式切换
 * 7. 安全告警响应
 *
 * 对接后端 API（4 个端点）：
 *   POST /api/v1/chat                              — 文本聊天
 *   POST /api/v1/multimodal/speech-to-text         — 语音转文字
 *   POST /api/v1/multimodal/text-to-speech/base64  — 文字转语音
 */


// ============================================================
// 常量
// ============================================================
const API_BASE = '/api/v1';
const REQUEST_TIMEOUT_MS = 60000;

const emotionCNMap = {
    'angry': '愤怒',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心',
    'sad': '悲伤',
    'surprise': '惊讶',
    'neutral': '平静',
    'unknown': '未知'
};


// ============================================================
// 全局状态（集中管理）
// ============================================================
const STATE = {
    sessionId: null,

    // 文本聊天
    loading: false,
    abortController: null,

    // TTS（默认关闭，用户手动开启）
    ttsEnabled: false,

    // 录音
    mediaRecorder: null,
    audioChunks: [],
    recordingStartTime: null,
    recordingTimer: null,

};


// ============================================================
// DOM 引用
// ============================================================
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const sessionStatus = document.getElementById('session-status');
const newSessionBtn = document.getElementById('new-session-btn');

const voiceBtn = document.getElementById('voice-btn');

const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const ttsToggle = document.getElementById('tts-toggle');
const recordingOverlay = document.getElementById('recording-overlay');
const stopRecordingBtn = document.getElementById('stop-recording-btn');
const recordingTimerDisplay = document.getElementById('recording-timer');
const recordingText = document.getElementById('recording-text');

const emotionPanel = document.getElementById('emotion-panel');
const closeEmotionBtn = document.getElementById('close-emotion-btn');
const emotionType = document.getElementById('emotion-type');
const intensityFill = document.getElementById('intensity-fill');
const intensityValue = document.getElementById('intensity-value');

const profileBtn = document.getElementById('profile-btn');
const profileOverlay = document.getElementById('profile-overlay');
const closeProfileBtn = document.getElementById('close-profile-btn');
const exportBtn = document.getElementById('export-btn');
const exportPassword = document.getElementById('export-password');
const exportStatus = document.getElementById('export-status');


// ============================================================
// 认证（登录/注册）
// ============================================================
const authOverlay = document.getElementById('auth-overlay');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginError = document.getElementById('login-error');
const registerError = document.getElementById('register-error');
const userDisplay = document.getElementById('user-display');
const logoutBtn = document.getElementById('logout-btn');

// Tab 切换
document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.tab;
        loginForm.classList.toggle('hidden', target !== 'login');
        registerForm.classList.toggle('hidden', target !== 'register');
        loginError.classList.add('hidden');
        registerError.classList.add('hidden');
    });
});

// 登录
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginError.classList.add('hidden');
    const btn = loginForm.querySelector('button');
    btn.disabled = true;
    btn.textContent = '登录中...';

    try {
        const resp = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: document.getElementById('login-username').value.trim(),
                password: document.getElementById('login-password').value,
            }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            loginError.textContent = data.detail || '登录失败';
            loginError.classList.remove('hidden');
            return;
        }
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        localStorage.setItem('user_id', data.user_id);
        onAuthSuccess();
    } catch (err) {
        loginError.textContent = '网络错误，请重试';
        loginError.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = '登录';
    }
});

// 注册
registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    registerError.classList.add('hidden');

    const password = document.getElementById('register-password').value;
    const confirm = document.getElementById('register-password-confirm').value;
    if (password !== confirm) {
        registerError.textContent = '两次密码不一致';
        registerError.classList.remove('hidden');
        return;
    }

    const btn = registerForm.querySelector('button');
    btn.disabled = true;
    btn.textContent = '注册中...';

    try {
        const displayName = document.getElementById('register-display-name').value.trim();
        const body = {
            username: document.getElementById('register-username').value.trim(),
            password: password,
        };
        if (displayName) body.display_name = displayName;

        const resp = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) {
            registerError.textContent = data.detail || '注册失败';
            registerError.classList.remove('hidden');
            return;
        }
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        localStorage.setItem('user_id', data.user_id);
        onAuthSuccess();
    } catch (err) {
        registerError.textContent = '网络错误，请重试';
        registerError.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = '注册';
    }
});

function onAuthSuccess() {
    authOverlay.classList.add('hidden');
    fetchUserInfo();
    // 同步知情同意到后端
    if (localStorage.getItem(CONSENT_KEY) === 'true') {
        fetch(`${API_BASE}/auth/consent`, {
            method: 'POST',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ version: '1.0' })
        }).catch(() => {});
    }
}

function fetchUserInfo() {
    fetch(`${API_BASE}/auth/me`, { headers: authHeaders() })
        .then(r => {
            if (r.status === 401) {
                localStorage.removeItem('access_token');
                localStorage.removeItem('refresh_token');
                localStorage.removeItem('user_id');
                authOverlay.classList.remove('hidden');
                return null;
            }
            return r.json();
        })
        .then(user => {
            if (!user) return;
            const name = user.display_name || user.user_id || '';
            userDisplay.textContent = name ? `👤 ${name.substring(0, 16)}` : '';
            logoutBtn.classList.toggle('hidden', !name);
        })
        .catch(() => {});
}

function handleLogout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_id');
    // 不删 CONSENT_KEY — 知情同意在注册之前，与登录状态无关
    userDisplay.textContent = '';
    logoutBtn.classList.add('hidden');
    authOverlay.classList.remove('hidden');
    if (STATE.sessionId) {
        STATE.sessionId = null;
        updateSessionStatus();
    }
}

function isLoggedIn() {
    return !!localStorage.getItem('access_token');
}

// ============================================================
// 知情同意
// ============================================================
const CONSENT_KEY = 'consent_given_v1';

function showConsentModal() {
    document.getElementById('consent-overlay').classList.remove('hidden');
}

function hideConsentModal() {
    document.getElementById('consent-overlay').classList.add('hidden');
}

function onConsentAgree() {
    localStorage.setItem(CONSENT_KEY, 'true');
    hideConsentModal();

    // 已登录则同步到后端 + 刷新用户信息
    if (isLoggedIn()) {
        fetch(`${API_BASE}/auth/consent`, {
            method: 'POST',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ version: '1.0' })
        }).catch(() => {});
        fetchUserInfo();
    } else {
        // 未登录 → 显示登录/注册
        authOverlay.classList.remove('hidden');
    }
}

function onConsentDisagree() {
    document.querySelector('.consent-content').innerHTML =
        '<p style="text-align:center;padding:20px">您需要同意知情同意书才能使用本服务。</p>' +
        '<p style="text-align:center">如需继续，请刷新页面重新授权。</p>';
    document.querySelector('.consent-buttons').innerHTML = '';
}

function authHeaders() {
    const token = localStorage.getItem('access_token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function refreshToken() {
    const refresh = localStorage.getItem('refresh_token');
    if (!refresh) return false;
    try {
        const resp = await fetch(`${API_BASE}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refresh }),
        });
        if (!resp.ok) return false;
        const data = await resp.json();
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        localStorage.setItem('user_id', data.user_id);
        return true;
    } catch {
        return false;
    }
}

/** 带自动 refresh 的 fetch。用法同 fetch()，自动附加 auth header，401 时尝试刷新。 */
async function authFetch(url, options = {}) {
    const headers = { ...(options.headers || {}), ...authHeaders() };
    let resp = await fetch(url, { ...options, headers });

    if (resp.status === 401) {
        const ok = await refreshToken();
        if (ok) {
            // 用新 token 重试一次
            const newHeaders = { ...(options.headers || {}), ...authHeaders() };
            resp = await fetch(url, { ...options, headers: newHeaders });
        } else {
            // refresh 也失败 → 清理并回到登录页
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            localStorage.removeItem('user_id');
            userDisplay.textContent = '';
            logoutBtn.classList.add('hidden');
            authOverlay.classList.remove('hidden');
        }
    }
    return resp;
}


// ============================================================
// 个人中心 & 数据导出
// ============================================================
async function openProfile() {
    profileOverlay.classList.remove('hidden');
    document.getElementById('export-status').classList.add('hidden');
    document.getElementById('export-status').textContent = '';
    exportPassword.value = '';

    // 刷新用户信息
    try {
        const resp = await fetch(`${API_BASE}/auth/me`, { headers: authHeaders() });
        if (resp.ok) {
            const user = await resp.json();
            document.getElementById('info-user-id').textContent = user.user_id || '-';
            document.getElementById('info-display-name').textContent = user.display_name || '-';
            const statusMap = { active: '正常', disabled: '已停用', deleted: '已注销' };
            document.getElementById('info-status').textContent = statusMap[user.status] || user.status || '-';
            document.getElementById('info-consent').textContent = user.consent_at ? '已签署' : '未签署';
            document.getElementById('info-created').textContent = '-';
        }
    } catch (e) {
        console.warn('Failed to fetch user profile:', e);
    }
}

function closeProfile() {
    profileOverlay.classList.add('hidden');
}

async function handleExport() {
    const password = exportPassword.value.trim();
    if (!password) {
        showExportStatus('请输入密码', 'error');
        return;
    }

    exportBtn.disabled = true;
    exportStatus.classList.add('hidden');
    exportStatus.textContent = '';

    try {
        const resp = await fetch(
            `${API_BASE}/user/export?password=${encodeURIComponent(password)}`,
            { headers: authHeaders() }
        );

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            const msg = errData.detail || `请求失败 (${resp.status})`;
            showExportStatus(msg, 'error');
            return;
        }

        const data = await resp.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `personal_data_export_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showExportStatus('数据导出成功，文件已下载。', 'success');
        exportPassword.value = '';
    } catch (e) {
        console.error('Export error:', e);
        showExportStatus('网络错误，请重试。', 'error');
    } finally {
        exportBtn.disabled = false;
    }
}

function showExportStatus(message, type) {
    exportStatus.textContent = message;
    exportStatus.className = `export-status ${type}`;
    exportStatus.classList.remove('hidden');
}


// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    updateTTSToggle();
    initDarkMode();

    const consented = localStorage.getItem(CONSENT_KEY) === 'true';
    const loggedIn = isLoggedIn();

    if (consented && loggedIn) {
        // 已同意 + 已登录 → 直接进聊天
        authOverlay.classList.add('hidden');
        fetchUserInfo();
    } else if (!consented) {
        // 未同意 → 先展示知情同意，隐藏登录
        authOverlay.classList.add('hidden');
        showConsentModal();
    } else {
        // 已同意但未登录 → 展示登录
        authOverlay.classList.remove('hidden');
    }
});

function initEventListeners() {
    sendBtn.addEventListener('click', () => sendMessage());

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', autoResizeTextarea);

    newSessionBtn.addEventListener('click', createNewSession);

    voiceBtn.addEventListener('click', toggleVoiceRecording);
    stopRecordingBtn.addEventListener('click', stopRecording);

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    ttsToggle.addEventListener('click', toggleTTS);

    closeEmotionBtn.addEventListener('click', () => {
        emotionPanel.classList.remove('show');
    });

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleDarkMode);
    }

    // 知情同意按钮
    const agreeBtn = document.getElementById('consent-agree-btn');
    if (agreeBtn) agreeBtn.addEventListener('click', onConsentAgree);
    const disagreeBtn = document.getElementById('consent-disagree-btn');
    if (disagreeBtn) disagreeBtn.addEventListener('click', onConsentDisagree);

    // 登录退出
    if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);

    // 个人中心
    if (profileBtn) profileBtn.addEventListener('click', openProfile);
    if (closeProfileBtn) closeProfileBtn.addEventListener('click', closeProfile);
    if (exportBtn) exportBtn.addEventListener('click', handleExport);
    if (profileOverlay) {
        profileOverlay.addEventListener('click', (e) => {
            if (e.target === profileOverlay) closeProfile();
        });
    }
}


// ============================================================
// 状态管理
// ============================================================
/** 唯一入口：切换加载态，控制输入框和发送按钮 */
function setLoading(loading) {
    STATE.loading = loading;
    sendBtn.disabled = loading;
    messageInput.disabled = loading;
}

function cancelCurrentRequest() {
    if (STATE.abortController) {
        STATE.abortController.abort();
        STATE.abortController = null;
    }
    if (STATE.streamAbortController) {
        STATE.streamAbortController.abort();
        STATE.streamAbortController = null;
    }
}

function updateSessionStatus() {
    if (STATE.sessionId) {
        sessionStatus.textContent = `会话: ${STATE.sessionId.substring(0, 8)}...`;
    } else {
        sessionStatus.textContent = '会话: 未开始';
    }
}


// ============================================================
// 工具函数
// ============================================================
/** 创建 SVG 头像（替代 emoji） */
function createAvatar(isUser) {
    const div = document.createElement('div');
    if (isUser) {
        // 柔和人物剪影 — 圆润、友好
        div.innerHTML = '<svg class="svg-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="7" r="4"/><path d="M4.5 21.5a7.5 7.5 0 0 1 15 0"/></svg>';
    } else {
        // 心形 — 传达关怀、共情、温暖
        div.innerHTML = '<svg class="svg-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.8 13.6c1.5-1.3 3-2.9 3-5.3a5.5 5.5 0 0 0-9.8-3.6L12 7l-2-2.3A5.5 5.5 0 0 0 .2 8.3c0 2.4 1.5 4 3 5.3l7 7a1.4 1.4 0 0 0 1.8 0l7-7Z"/></svg>';
    }
    return div;
}

function htmlEscape(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

function shouldAutoScroll() {
    const threshold = 120;
    return chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight < threshold;
}

function scrollChat() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'auto'
    });
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

function showTyping() {
    typingIndicator.classList.add('show');
    if (shouldAutoScroll()) scrollChat();
}

function hideTyping() {
    typingIndicator.classList.remove('show');
}

/** 按 HTTP 状态码提供不同的错误提示 */
function getErrorMessage(status) {
    if (status === 422) return '消息内容不符合要求，请修改后重试。';
    if (status === 413) return '消息过长，请缩短后重试。';
    if (status >= 500) return '服务器繁忙，请稍后重试。';
    return '抱歉，发生了错误。请稍后重试。';
}

/** 创建带超时 + 自动认证的 fetch */
function fetchWithTimeout(url, options = {}, timeout = REQUEST_TIMEOUT_MS) {
    STATE.abortController = new AbortController();
    const timer = setTimeout(() => STATE.abortController.abort(), timeout);

    return authFetch(url, {
        ...options,
        signal: STATE.abortController.signal
    }).finally(() => clearTimeout(timer));
}


// ============================================================
// 消息渲染
// ============================================================
/**
 * 添加一条消息到聊天区
 * @param {string}  content     — 消息文本
 * @param {boolean} isUser      — 是否为用户消息
 * @param {string}  [audioBase64=null] — TTS 音频 base64（仅 AI 消息）
 */
function addMessage(content, isUser, audioBase64 = null) {
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;

    const avatar = createAvatar(isUser);
    avatar.className = 'message-avatar';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = htmlEscape(content);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });

    messageContent.appendChild(timeDiv);

    if (!isUser && audioBase64) {
        const audioPlayer = document.createElement('div');
        audioPlayer.className = 'audio-player';
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = `data:audio/mpeg;base64,${audioBase64}`;
        audioPlayer.appendChild(audio);
        messageContent.appendChild(audioPlayer);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    chatMessages.appendChild(messageDiv);
    if (shouldAutoScroll()) scrollChat();
}

function showError(content) {
    hideTyping();
    addMessage(content, false);
}


// ============================================================
// 流式消息 DOM 辅助
// ============================================================
/** 创建 AI 气泡，内容区先显示三点缓冲动画 + 思考文案，待首 token 到达后自动替换为文本 */
function createStreamingMessageBubble() {
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai streaming';

    const avatar = createAvatar(false);
    avatar.className = 'message-avatar';

    const content = document.createElement('div');
    content.className = 'message-content';

    // 思考状态行：三点动效 + 文案（类似豆包/DeepSeek）
    const thinkingRow = document.createElement('div');
    thinkingRow.className = 'streaming-thinking-row';

    const dots = document.createElement('div');
    dots.className = 'streaming-dots';
    for (let i = 0; i < 3; i++) {
        dots.appendChild(document.createElement('span'));
    }
    thinkingRow.appendChild(dots);

    const thinkingLabel = document.createElement('span');
    thinkingLabel.className = 'streaming-thinking-label';
    thinkingLabel.textContent = '正在分析...';
    thinkingRow.appendChild(thinkingLabel);

    content.appendChild(thinkingRow);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('zh-CN', {
        hour: '2-digit', minute: '2-digit'
    });
    content.appendChild(timeDiv);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    return messageDiv;
}

/** 更新流式气泡中的思考文案 */
function updateThinkingLabel(messageDiv, text) {
    const label = messageDiv.querySelector('.streaming-thinking-label');
    if (label) {
        label.textContent = text;
    }
}

/** 更新流式气泡文字，保留时间戳 */
function updateStreamingMessage(messageDiv, text) {
    const content = messageDiv.querySelector('.message-content');
    const timeDiv = content.querySelector('.message-time');
    content.innerHTML = htmlEscape(text);
    content.appendChild(timeDiv);
    if (shouldAutoScroll()) scrollChat();
}

/** 完成流式：移除 .streaming，时间戳保留 */
function finalizeStreamingMessage(messageDiv, text) {
    updateStreamingMessage(messageDiv, text);
    messageDiv.classList.remove('streaming');
}

/** 往已完成的气泡追加音频播放器 */
function appendAudioToMessage(messageDiv, audioBase64) {
    const content = messageDiv.querySelector('.message-content');
    const player = document.createElement('div');
    player.className = 'audio-player';
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = `data:audio/mpeg;base64,${audioBase64}`;
    player.appendChild(audio);
    content.appendChild(player);
}


// ============================================================
// 文本聊天
// ============================================================
/**
 * 发送消息到后端，渲染 AI 回复
 * @param {string} [messageOverride] — 外部传入的消息文本（语音识别后自动发送），
 *                                     留空则从输入框读取
 */
async function sendMessage(messageOverride) {
    sendMessageStream(messageOverride);
}

/** 流式消息：fetch + ReadableStream 解析 SSE，逐 token 增量渲染 */
async function sendMessageStream(messageOverride) {
    const message = messageOverride != null
        ? String(messageOverride).trim()
        : messageInput.value.trim();

    if (!message || STATE.loading) return;

    if (!messageOverride) {
        addMessage(message, true);
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }

    setLoading(true);

    // 提前创建 AI 气泡（含三点动效），让同一个 DOM 元素从"思考"平滑过渡到"输出"
    // 避免 typing-indicator → 气泡的 DOM 切换造成的空白断档
    const streamMessageDiv = createStreamingMessageBubble();

    // 强制浏览器绘制一次，确保 streaming 气泡（含三点动效）已渲染到屏幕
    // 避免 authFetch 的微任务阻塞首次 paint，造成"卡住"的观感
    await new Promise(r => requestAnimationFrame(r));

    STATE.streamAbortController = new AbortController();
    let fullText = '';
    let meta = null;

    try {
        const response = await authFetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: STATE.sessionId,
                enable_thought_chain: true,
                enable_emotion_analysis: true
            }),
            signal: STATE.streamAbortController.signal
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                try {
                    const event = JSON.parse(line.slice(6));

                    switch (event.type) {
                        case 'meta':
                            meta = event;
                            STATE.sessionId = event.session_id;
                            updateSessionStatus();
                            if (event.emotion_analysis) {
                                updateEmotionPanel(event.emotion_analysis);
                            }
                            break;

                        case 'status':
                            // 后端状态更新：思考 → 检索 → 生成
                            if (event.phase === 'thinking') {
                                updateThinkingLabel(streamMessageDiv, '正在思考...');
                            } else if (event.phase === 'searching') {
                                updateThinkingLabel(streamMessageDiv, '正在检索知识库...');
                            }
                            break;

                        case 'delta':
                            fullText += event.token;
                            updateStreamingMessage(streamMessageDiv, fullText);
                            break;

                        case 'done':
                            finalizeStreamingMessage(streamMessageDiv, fullText);

                            if (meta && meta.safety_alert) {
                                addMessage(
                                    '⚠️ 系统检测到高风险内容。如果您正处于危机中，请立即拨打心理援助热线：400-161-9995 或 120。',
                                    false
                                );
                            } else if (!fullText) {
                                updateStreamingMessage(streamMessageDiv, '抱歉，未能生成回复，请重试。');
                                streamMessageDiv.classList.remove('streaming');
                            }

                            if (STATE.ttsEnabled && fullText) {
                                const audioBase64 = await fetchTTS(fullText);
                                if (audioBase64) appendAudioToMessage(streamMessageDiv, audioBase64);
                            }
                            break;

                        case 'error':
                            updateStreamingMessage(streamMessageDiv,
                                fullText + '\n\n[回复中断：' + (event.message || '发生错误') + ']');
                            streamMessageDiv.classList.remove('streaming');
                            break;
                    }
                } catch (e) {
                    // 非 JSON 行，跳过
                }
            }
        }

    } catch (error) {
        streamMessageDiv.classList.remove('streaming');
        if (fullText) {
            // 已有部分文本，保留内容
        } else {
            // 尚未收到文本 — 在气泡内展示错误
            if (error.name === 'AbortError') {
                updateStreamingMessage(streamMessageDiv, '请求已取消。');
            } else {
                const status = error.message.startsWith('HTTP ')
                    ? parseInt(error.message.slice(5), 10)
                    : 0;
                console.error('Chat stream error:', error);
                updateStreamingMessage(streamMessageDiv, getErrorMessage(status));
            }
        }
    } finally {
        STATE.streamAbortController = null;
        setLoading(false);
    }
}


// ============================================================
// TTS（文字转语音）
// ============================================================
async function fetchTTS(text) {
    try {
        const response = await fetch(`${API_BASE}/multimodal/text-to-speech/base64`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const data = await response.json();
            return data.audio_base64;
        }
    } catch (e) {
        console.warn('TTS failed:', e);
    }
    return null;
}

function toggleTTS() {
    STATE.ttsEnabled = !STATE.ttsEnabled;
    updateTTSToggle();
}

function updateTTSToggle() {
    ttsToggle.classList.toggle('active', STATE.ttsEnabled);
}


// ============================================================
// 语音录制 & 识别
// ============================================================
async function toggleVoiceRecording() {
    if (STATE.mediaRecorder && STATE.mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        STATE.mediaRecorder = new MediaRecorder(stream);
        STATE.audioChunks = [];

        STATE.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) STATE.audioChunks.push(event.data);
        };

        STATE.mediaRecorder.onstop = async () => {
            // 无论正常停止还是异常停止，都清理录音 UI
            voiceBtn.classList.remove('recording');
            recordingOverlay.classList.remove('show');
            if (STATE.recordingTimer) {
                clearInterval(STATE.recordingTimer);
                STATE.recordingTimer = null;
            }

            const audioBlob = new Blob(STATE.audioChunks, { type: 'audio/webm' });
            await handleSpeechToText(audioBlob);

            stream.getTracks().forEach(track => track.stop());
        };

        STATE.mediaRecorder.start();
        STATE.recordingStartTime = Date.now();

        voiceBtn.classList.add('recording');
        recordingOverlay.classList.add('show');

        STATE.recordingTimer = setInterval(updateRecordingTimer, 1000);

    } catch (error) {
        console.error('Error starting recording:', error);
        if (error.name === 'NotAllowedError') {
            alert('麦克风权限被拒绝。\n\n请在浏览器地址栏左侧点击锁/信息图标 → 开启麦克风权限，\n或在系统设置 → 隐私 → 麦克风中允许浏览器访问。');
        } else if (error.name === 'NotFoundError') {
            alert('未检测到麦克风设备，请检查麦克风是否已连接。');
        } else if (error.name === 'NotReadableError') {
            alert('麦克风被其他应用占用，请关闭其他正在使用麦克风的程序后重试。');
        } else {
            alert('无法访问麦克风：' + error.message);
        }
    }
}

function stopRecording() {
    if (STATE.mediaRecorder && STATE.mediaRecorder.state === 'recording') {
        STATE.mediaRecorder.stop();
    }
    // UI 清理已移至 mediaRecorder.onstop 统一处理
}

function updateRecordingTimer() {
    const elapsed = Math.floor((Date.now() - STATE.recordingStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    recordingTimerDisplay.textContent = `${mins}:${secs}`;
}

/**
 * 将录音 Blob 发送给后端进行语音识别，
 * 识别成功后以识别的文本自动调用 sendMessage
 */
async function handleSpeechToText(audioBlob) {
    // 不调用 setLoading，避免 sendMessage 守卫被阻塞；
    // 仅禁用语音按钮以防重复录制
    voiceBtn.disabled = true;
    showTyping();
    recordingText.textContent = '正在识别语音...';

    try {
        const formData = new FormData();
        // 保留原始文件名（上传文件）或使用默认名（录音）
        const blobName = audioBlob.name || 'recording.webm';
        formData.append('audio', audioBlob, blobName);

        const response = await fetch(`${API_BASE}/multimodal/speech-to-text`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (data.text) {
            addMessage(`🎤 ${data.text}`, true);
            // 统一走 sendMessage 获取 AI 回复（此时 STATE.loading 为 false，守卫通过）
            await sendMessage(data.text);
        } else {
            hideTyping();
            showError('未能识别语音内容，请重试。');
        }

    } catch (error) {
        console.error('STT error:', error);
        hideTyping();
        showError('语音识别失败，请重试。');
    } finally {
        voiceBtn.disabled = false;
        recordingText.textContent = '正在录音...';
    }
}

/**
 * 文件上传处理：安全过滤 → 根据 MIME 类型分流
 *   1. 先送安全端点检测
 *   2. blocked → 显示安全告警，不继续处理
 *   3. 通过 → 情绪识别 / 语音转文字
 */
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const mime = file.type;
    const name = file.name;
    const typeLabel = mime.startsWith('image/') ? '图片'
        : mime.startsWith('audio/') ? '音频' : '视频';
    const safetyEndpoint = mime.startsWith('image/') ? 'image'
        : mime.startsWith('audio/') ? 'audio' : 'video';

    showTyping();
    addMessage(`📎 上传${typeLabel}: ${name} (安全检测中...)`, true);

    try {
        // 1) 安全检测
        const safetyResult = await checkFileSafety(file, safetyEndpoint);
        if (!safetyResult) {
            // 安全检测本身失败，让用户决定是否继续
            hideTyping();
            return;
        }
        if (safetyResult.blocked) {
            hideTyping();
            addMessage(
                '⚠️ 系统检测到高风险内容，已阻止该文件的上传。如果您正处于危机中，请立即拨打心理援助热线：400-161-9995 或 120。',
                false
            );
            event.target.value = '';
            return;
        }

        // 2) 安全通过 → 统一走多模态情绪识别管线
        if (mime.startsWith('image/') || mime.startsWith('audio/') || mime.startsWith('video/')) {
            await sendMultimodalChat(file, mime, name);
        } else {
            hideTyping();
            alert('不支持的文件格式，请上传图片、音频或视频文件。');
        }
    } catch (err) {
        console.error('File upload error:', err);
        hideTyping();
        showError('文件处理失败，请重试。');
    }

    // 重置 input 以允许重复上传同一文件
    event.target.value = '';
}

/**
 * 调用后端安全检测端点
 * @returns {Object|null} 安全检测结果，null 表示检测请求失败
 */
async function checkFileSafety(file, type) {
    try {
        const formData = new FormData();
        formData.append(type, file, file.name);

        const response = await authFetch(`${API_BASE}/multimodal/safety/${type}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.error(`Safety check failed: HTTP ${response.status}`);
            return null;
        }

        const data = await response.json();
        return data;

    } catch (err) {
        console.error('Safety check error:', err);
        return null;
    }
}

/**
 * 统一多模态聊天：读取文件 → POST /multimodal/chat → 渲染响应
 * 包含 ASR、音频/视觉情绪识别、情绪融合、干预管线
 */
async function sendMultimodalChat(file, mime, name) {
    // 读取文件为 base64
    const base64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });

    // 根据 MIME 类型选字段
    const fieldKey = mime.startsWith('image/') ? 'image_data'
        : mime.startsWith('audio/') ? 'audio_data' : 'video_data';
    const typeLabel = mime.startsWith('image/') ? '图片'
        : mime.startsWith('audio/') ? '音频' : '视频';

    setLoading(true);
    showTyping();

    try {
        const body = {
            session_id: STATE.sessionId,
            [fieldKey]: base64,
        };

        const response = await fetchWithTimeout(`${API_BASE}/multimodal/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }, 120000);  // 视频安全+ASR+LLM+TTS 链路长，给 2 分钟

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        STATE.sessionId = data.session_id;
        updateSessionStatus();
        hideTyping();

        // 显示转录文本（如有）
        if (data.transcribed_text) {
            addMessage(`🎤 ${data.transcribed_text}`, true);
        }

        // 显示 AI 回复
        let audioBase64 = null;
        if (STATE.ttsEnabled && data.response) {
            audioBase64 = await fetchTTS(data.response);
        }

        if (data.response) {
            addMessage(data.response, false, audioBase64);
        } else {
            addMessage('抱歉，未能生成回复，请重试。', false);
        }

        // 更新情绪面板
        const emotionInfo = data.fused_emotion || data.emotion || data.audio_emotion || data.visual_emotion;
        if (emotionInfo && emotionInfo.primary_emotion) {
            updateEmotionPanel({
                primary_emotion: emotionInfo.primary_emotion,
                intensity: emotionInfo.confidence || 0
            });
        }

    } catch (error) {
        console.error('Multimodal chat error:', error);
        hideTyping();
        showError('多模态处理失败，请重试。');
    } finally {
        setLoading(false);
    }
}

// ============================================================
// 情绪面板
// ============================================================
/**
 * 更新右侧情绪分析面板
 * @param {Object} emotionState — 情绪数据
 */
function updateEmotionPanel(emotionState) {
    emotionPanel.classList.add('show');

    const primaryEmotion = emotionState.primary_emotion || 'unknown';
    const intensity = Math.round((emotionState.intensity ?? 0.5) * 100);

    emotionType.textContent = emotionCNMap[primaryEmotion] || primaryEmotion;
    intensityFill.style.width = intensity + '%';
    intensityValue.textContent = intensity + '%';
}

function resetEmotionPanel() {
    emotionType.textContent = '-';
    intensityFill.style.width = '0%';
    intensityValue.textContent = '0%';
}


// ============================================================
// 会话管理
// ============================================================
function createNewSession() {
    cancelCurrentRequest();
    STATE.sessionId = null;
    setLoading(false);

    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-illustration">
                <div class="welcome-halo"></div>
                <div class="welcome-icon">💬</div>
            </div>
            <h2>欢迎使用心理咨询AI助手</h2>
            <p>我是一个专业的心理咨询AI，可以倾听你的烦恼，帮助你缓解压力。</p>
            <p>请放心，我们的对话是保密的。你可以随时开始分享你的感受。</p>
            <div class="feature-tags">
                <span class="tag">🎤 语音输入</span>

                <span class="tag">🔊 语音回复</span>
            </div>
        </div>
    `;

    updateSessionStatus();
    resetEmotionPanel();
    emotionPanel.classList.remove('show');
    messageInput.focus();
}


// ============================================================
// 暗色模式
// ============================================================
function initDarkMode() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.body.classList.add('dark');
    }
    updateThemeToggleIcon();

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            document.body.classList.toggle('dark', e.matches);
            updateThemeToggleIcon();
        }
    });
}

function toggleDarkMode() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeToggleIcon();
}

function updateThemeToggleIcon() {
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.innerHTML = document.body.classList.contains('dark')
            ? '<svg class="svg-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
            : '<svg class="svg-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
    }
}
