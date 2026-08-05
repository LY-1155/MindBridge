/**
 * 实时语音对话 — 按住说话 WebSocket 客户端。
 *
 * 用法：
 *   1. 页面加载后调用 VoiceChat.init({ token, sessionId })
 *   2. 用户按住 "语音" 按钮 → 录音 → 松手 → AI 语音回复
 */

const VoiceChat = (() => {
  // ── 状态 ──────────────────────────────────────
  let ws = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let token = "";
  let sessionId = null;
  let audioQueue = [];
  let isPlaying = false;

  // DOM 引用
  let btnTalk = null;
  let textSubtitle = null;
  let emotionBadge = null;
  let chatMessagesEl = null;

  // AI 文本累积
  let aiTextBuffer = "";
  let userTextBuffer = "";

  // ── 聊天集成 ────────────────────────────────────

  /** 往聊天框插入消息（复用 multimodal.js / chat.js 的 addMessage，如果存在的话） */
  function addToChat(content, isUser) {
    // 优先复用页面已有的 addMessage 函数
    if (typeof addMessage === "function") {
      try {
        addMessage(content, isUser);
        return;
      } catch (e) { /* fallthrough */ }
    }
    // 兜底：自己构造 DOM
    if (!chatMessagesEl) {
      chatMessagesEl = document.getElementById("chat-messages");
    }
    if (!chatMessagesEl) return;
    const welcome = chatMessagesEl.querySelector(".welcome-message");
    if (welcome) welcome.remove();
    const div = document.createElement("div");
    div.className = `message ${isUser ? "user" : "ai"}`;
    div.innerHTML = `<div class="message-content">${content}<div class="message-time">${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</div></div>`;
    chatMessagesEl.appendChild(div);
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  }

  // ── WebSocket ──────────────────────────────────

  function connect(wsToken, existingSessionId) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let url = `${proto}://${location.host}/ws/voice?token=${encodeURIComponent(wsToken)}`;
    if (existingSessionId) {
      url += `&session_id=${encodeURIComponent(existingSessionId)}`;
    }
    console.log("[VoiceChat] Connecting to:", url.replace(wsToken, "***"));

    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[VoiceChat] ✅ WebSocket connected");
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        audioQueue.push(new Uint8Array(event.data));
        playNextChunk();
      } else {
        try {
          const msg = JSON.parse(event.data);
          console.log("[VoiceChat] ←", msg.type);
          handleMessage(msg);
        } catch (e) {
          console.warn("[VoiceChat] Invalid JSON message", e);
        }
      }
    };

    ws.onclose = (event) => {
      console.log("[VoiceChat] ❌ WebSocket closed:", event.code, event.reason);
      // auth 失败 → 每 3 秒重试，直到拿到有效 token
      if (event.code === 1008) {
        let attempts = 0;
        const retry = () => {
          const t = localStorage.getItem('access_token');
          if (t && t !== token) {
            console.log("[VoiceChat] Retrying with fresh token (attempt", attempts + 1, ")");
            token = t;
            connect(t, sessionId);
          } else if (attempts < 10) {
            attempts++;
            setTimeout(retry, 3000);
          }
        };
        setTimeout(retry, 3000);
      }
    };

    ws.onerror = (err) => {
      console.error("[VoiceChat] ❌ WebSocket error — 检查服务端是否启动、token 是否有效");
    };
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "connected":
        sessionId = msg.session_id;
        console.log("[VoiceChat] Session:", sessionId);
        break;

      case "status":
        updateSubtitle(`[${msg.phase}]`);
        break;

      case "user_text":
        userTextBuffer = msg.content;
        addToChat("🎤 " + msg.content, true);
        updateSubtitle("✅ " + msg.content);
        break;

      case "text.delta":
        aiTextBuffer += msg.content;
        appendSubtitle(msg.content);
        break;

      case "emotion.result":
        if (emotionBadge && msg.primary_emotion) {
          emotionBadge.textContent = msg.primary_emotion;
          emotionBadge.style.display = "inline-block";
        }
        break;

      case "done":
        if (aiTextBuffer) {
          addToChat(aiTextBuffer, false);
        }
        aiTextBuffer = "";
        userTextBuffer = "";
        updateSubtitle(""); // 清字幕
        break;

      case "error":
        console.error("[VoiceChat] ← error:", msg.message);
        updateSubtitle("❌ " + msg.message);
        setTimeout(() => updateSubtitle(""), 3000);
        aiTextBuffer = "";
        break;

      case "cancelled":
        updateSubtitle("已取消");
        setTimeout(() => updateSubtitle(""), 1500);
        aiTextBuffer = "";
        break;
    }
  }

  // ── 字幕 ────────────────────────────────────────

  function updateSubtitle(text) {
    if (textSubtitle) textSubtitle.textContent = text;
  }

  function appendSubtitle(text) {
    if (textSubtitle) textSubtitle.textContent += text;
  }

  // ── 音频播放 ────────────────────────────────────

  async function playNextChunk() {
    if (isPlaying || audioQueue.length === 0) return;
    isPlaying = true;

    // 创建 MediaSource 或使用 AudioContext 连续播放
    // 简化方案：将 chunk 拼接后一次性播放（适合短回复）
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // 收集所有已到达的 chunk
    const chunks = audioQueue.splice(0, audioQueue.length);
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
    if (totalLength === 0) { isPlaying = false; return; }

    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      combined.set(c, offset);
      offset += c.length;
    }

    try {
      const audioBuffer = await audioCtx.decodeAudioData(combined.buffer.slice());
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      source.onended = () => {
        isPlaying = false;
        // 检查是否有新 chunk 在播放期间到达
        if (audioQueue.length > 0) playNextChunk();
      };
      source.start(0);
    } catch (e) {
      console.warn("[VoiceChat] Audio decode failed (may need full MP3 before playing):", e.message);
      // Edge TTS 产 MP3，AudioContext.decodeAudioData 不完全支持 MP3
      // 回退：用 <audio> 元素播放
      playWithAudioElement(combined);
    }
  }

  function playWithAudioElement(data) {
    const blob = new Blob([data], { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      isPlaying = false;
      if (audioQueue.length > 0) playNextChunk();
    };
    audio.play().catch(console.warn);
  }

  // ── 录音 ────────────────────────────────────────

  async function startRecording() {
    if (isRecording) return;
    audioChunks = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      // 不在录制中推流，松手后一次性发送完整文件（避免 WebM chunk 拼接损坏）
      mediaRecorder.start();
      isRecording = true;
      updateSubtitle("🎤 正在录音...松开结束");
      if (btnTalk) btnTalk.classList.add("recording");
      console.log("[VoiceChat] Recording started");
    } catch (err) {
      console.error("[VoiceChat] Microphone access denied:", err);
      updateSubtitle("无法访问麦克风");
    }
  }

  function stopRecording(cancelled = false) {
    console.log("[VoiceChat] stopRecording called, cancelled:", cancelled, "isRecording:", isRecording, "ws open:", ws && ws.readyState === WebSocket.OPEN);
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;

    mediaRecorder.onstop = () => {
      console.log("[VoiceChat] onstop fired, audioChunks:", audioChunks.length, "cancelled:", cancelled);
      if (!cancelled && ws && ws.readyState === WebSocket.OPEN) {
        const mimeType = mediaRecorder ? mediaRecorder.mimeType : "audio/webm";
        const blob = new Blob(audioChunks, { type: mimeType });
        console.log("[VoiceChat] Blob size:", blob.size, "bytes");
        blob.arrayBuffer().then((buf) => {
          ws.send(buf);
          ws.send(JSON.stringify({ type: "audio.end" }));
          console.log("[VoiceChat] Audio sent:", buf.byteLength, "bytes");
        }).catch((err) => {
          console.error("[VoiceChat] Blob->arrayBuffer failed:", err);
        });
      } else if (cancelled && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "cancel" }));
        console.log("[VoiceChat] Cancelled");
      }
      audioChunks = [];
    };

    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    mediaRecorder = null;

    if (btnTalk) btnTalk.classList.remove("recording");
    updateSubtitle("");
  }

  // ── 按钮事件绑定 ────────────────────────────────

  function bindButton(buttonId, subtitleId, badgeId) {
    btnTalk = document.getElementById(buttonId);
    textSubtitle = document.getElementById(subtitleId);
    if (badgeId) emotionBadge = document.getElementById(badgeId);

    if (!btnTalk) {
      console.warn("[VoiceChat] Button not found:", buttonId);
      return;
    }

    // 鼠标 / 触摸：按住录音
    btnTalk.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startRecording();
    });
    btnTalk.addEventListener("mouseup", (e) => {
      e.preventDefault();
      stopRecording(false);
    });
    btnTalk.addEventListener("mouseleave", (e) => {
      if (isRecording) stopRecording(true); // 滑出 → 取消
    });

    btnTalk.addEventListener("touchstart", (e) => {
      e.preventDefault();
      startRecording();
    });
    btnTalk.addEventListener("touchend", (e) => {
      e.preventDefault();
      stopRecording(false);
    });
    btnTalk.addEventListener("touchcancel", () => {
      if (isRecording) stopRecording(true);
    });

    console.log("[VoiceChat] Button bound:", buttonId);
  }

  // ── 公开 API ────────────────────────────────────

  return {
    init({ token: t, sessionId: sid, buttonId, subtitleId, badgeId }) {
      token = t;
      sessionId = sid;
      connect(t, sid);
      bindButton(buttonId, subtitleId, badgeId);
    },

    disconnect() {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach((t) => t.stop());
      }
      if (ws) ws.close();
      isRecording = false;
      audioQueue = [];
    },

    isConnected() {
      return ws && ws.readyState === WebSocket.OPEN;
    },
  };
})();
