# WebSocket 实时语音对话

在回合制文字聊天基础上，新增 WebSocket 实时语音对话能力：用户按住说话 → 松手 → AI 流式语音回复，语音语调参与情绪判断。

## 为什么

### 为什么加语音对话而不是只保持文字

- 心理筛查场景中，语音是自然的倾诉通道——用户说出来比打字门槛更低，尤其在情绪低落时
- 语音语调（韵律特征）携带文字之外的情绪信息，SenseVoice 已有此能力，不应浪费
- 当前"录音→上传→ASR→文本管道"已跑通，升级为 WebSocket 实时通道改动可控

### 为什么 WebSocket 而不是 HTTP + SSE

- 语音对话需要双向流：前端持续推送音频 chunk（按住期间），服务端流式推送文本 + 音频（回复期间）
- HTTP 分块上传 + SSE 返回是为了一来一回设计的，双向并发不是原生语义
- 后续若要升级 VAD 自动断句或全双工对话，WebSocket 不需要重改传输层
- nginx 已配置 WebSocket 升级透传（见 `nginx/nginx.conf`），部署面无新增阻碍

### 为什么"按住说话"而不是 VAD 自动断句

- 按住说话（push-to-talk）是可控性最高的交互模式，用户可以精确控制哪段话发送
- 心理筛查场景下用户可能长时间沉默（在思考或情绪波动中），VAD 的静音阈值很难适配
- VAD 自动断句是后续升级方向，WebSocket 传输层已为它留好通道

### 为什么串行管线而不是改造并行

- 当前 4 阶段管线（安全→情绪→路由→干预）逻辑成熟、测试覆盖完整
- SenseVoice 一次推理同时产出 ASR 文本 + 语调情绪标签，ASR 和情绪无法进一步拆解并行
- 管线总耗时的瓶颈在 LLM 回复和 TTS 合成，两者本身必须串行（TTS 依赖 LLM 输出文本）

### 为什么流式 TTS + 逐句合成

- LLM 回复 + TTS 合成是延迟大头，整段生成完再播首音延迟可达 3-5 秒
- Edge TTS 底层 `communicate()` 支持 async generator 流式产出，不需要等待全段合成
- 逐句触发（LLM 出到句末标点时送 TTS）平衡了自然度和延迟：不在词中间切断，又比全段生成快

### 为什么降级处理 ASR 空结果

- 心理筛查场景下用户可能哽咽、沉默或无法组织语言——此时文字为空但音频情绪信号有效（如 distress）
- 有情绪信号时走安抚路由给共情回复（纯 TTS，不需 LLM），比冷冰冰的"请重试"更适合
- 完全无信号时返回错误提示，避免静默无回应让用户困惑

### 为什么前后端都支持取消

- 用户按着录音时可能说了半句后悔——滑出按钮取消是移动端常见交互
- 服务端需要清空已缓冲的音频数据，避免残留混入下次录音

## 决议

采用 WebSocket 双向通道承载实时语音对话，具体设计如下：

### 连接生命周期

- 一次对话（一个 session）对应一条 WebSocket 连接
- 连接时通过 URL 参数携带 JWT access token 认证：`wss://host/ws/voice?token=xxx`
- 新会话由服务端创建并返回 session_id；已有会话通过 URL 参数传入

### 消息协议

JSON 控制帧与二进制音频帧混合，以首个字节区分：

**前端 → 服务端：**

| 消息类型 | 格式 | 说明 |
|---|---|---|
| `audio.chunk` | 二进制 | 按住期间持续推送 Opus/WebM 编码的音频片段 |
| `audio.end` | JSON `{"type":"audio.end"}` | 松手，标记录音结束 |
| `cancel` | JSON `{"type":"cancel"}` | 用户滑出按钮取消，服务端清 buffer |

**服务端 → 前端：**

| 消息类型 | 格式 | 说明 |
|---|---|---|
| `status` | JSON `{"type":"status","phase":"..."}` | 管道阶段：asr / safety / emotion / generating |
| `text.delta` | JSON `{"type":"text.delta","content":"..."}` | LLM 流式文本增量（字幕） |
| `audio.delta` | 二进制 | TTS 流式音频 chunk |
| `emotion.result` | JSON `{"type":"emotion.result","primary":"...","confidence":0.x}` | 情绪融合结果 |
| `done` | JSON `{"type":"done","session_id":"..."}` | 本轮完成 |
| `error` | JSON `{"type":"error","message":"..."}` | 错误信息 |

### 管道对接

- 新增 `pipeline/ws_pipeline.py`：封装 WebSocket 版管道服务，内部调 ASR → Safety → Emotion → Router → Intervention → TTS，通过回调推消息
- 输入为音频 bytes，通过回调函数接口与 WebSocket handler 解耦
- 复用现有 `PipelineOrchestrator` 的核心模块，不重写管道逻辑

### 前端录音

- 浏览器 `MediaRecorder` API，产出 Opus 编码的 WebM 格式
- 按住按钮期间定时发送 audio chunk；松手发送 `audio.end`
- 服务端拼接音频 buffer → ffmpeg 转 WAV（16kHz mono）→ 送 SenseVoice

### TTS 流式推送

- LLM 流式输出 token → 按语义边界（句号/问号/感叹号/换行）积累句子 → 遇到句末标点时送 Edge TTS 合成 → 二进制音频 chunk 推前端
- 首句合成即开始播放，后续句子边合成边播

### ASR 空结果降级

- ASR 空 + 有音频情绪信号 → 走安抚路由，给简短共情 TTS 回复（无需 LLM）
- ASR 空 + 无情绪信号 → 推送 error 帧 "未检测到语音，请重试"

### 语音情绪

- 复用现有 SenseVoice 一次推理给出文本 + 语调情绪标签，不做修改
- 音频情绪与文本情绪经 `emotion_fusion.py` 加权融合后作为管道输入

## 后果

### 新增

- `pipeline/ws_pipeline.py`：WebSocket 版管道服务
- `api/routes/ws.py`：WebSocket 端点 `/ws/voice`
- `static/js/voice_chat.js`：前端按住说话 + WebSocket 交互
- 前端录音按钮 UI 组件

### 不变

- 4 阶段管线逻辑（safety/emotion/router/intervention）复用，不修改
- SenseVoice 语音情绪判断不做改动
- 会话管理（session_memory）复用，不修改
- 情绪融合（emotion_fusion）复用，不修改

### 风险

- `edge-tts` 流式 `communicate()` 的逐句触发方式未经充分测试，首音延迟可能不如预期
- WebSocket 连接在移动网络下可能不稳定，需要重连机制（本次不做）
- JWT token 在 URL 参数中传递，需确保仅通过 WSS（TLS 加密）连接
