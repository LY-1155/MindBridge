# CONTEXT

项目「心理异常智能早筛与精准干预」的领域术语表。仅包含概念定义，不含实现细节。

## 核心概念

### 会话（Session）
用户与系统的一次连续对话。会话内维护消息历史和情绪记录，跨轮次共享上下文。

### 管线（Pipeline）
端到端处理流程，按固定顺序串联四个阶段：安全过滤 → 情绪分析 → 智能路由 → 干预闭环。阶段间以 JSON 契约传递数据。

### 安全过滤模块（Safety Module）
管线第一阶段。对输入文本做敏感词匹配和分级（0 通过 / 1 记录 / 2 紧急）。敏感词命中是风险锚点的来源之一；显式危险信号（P0）触发安全短路直接进入危机干预，弱风险信号不触发短路，交由语义安全评估器判定。

### 情绪分析模块（Emotion Module）
管线第二阶段。对用户输入做情绪分析，产出统一的 `EmotionTags`。分析来源可为文本、语音音频、面部视频中的一路或多路，多路信号通过情绪融合层合并。

### 智能路由模块（Router Module）
管线第三阶段。根据情绪标签的 risk、安全过滤结果的等级、情绪类型以及用户意图，将请求分派到四种干预分支之一：通用（general）、知识（knowledge）、安抚（comfort）、危机（crisis）（risk 从低到高排列）。以 risk 为主要判断依据，意图（intent）和安全等级作为修正因子，情绪类型做段内微调。Router 无会话状态，只评判当前轮次。

## 路由子概念

### 通用路由（General Route）
无风险或不需干预场景的默认路径。侧重点为自然对话和信息回应，不使用共情模板或知识灌输。适用于 risk 极低、无明显情绪信号的信息提问、事实查询及日常闲聊。general 路由下 LLM 以助手身份正常回答，不强行将话题转向心理健康。

### 知识路由（Knowledge Route）
低-中等风险场景的干预路径。侧重点为提供情绪管理知识、应对策略或信息澄清。适用于 risk 较低、需要认知重评但情绪尚未明显失控的情况。配合 RAG 知识库检索提供心理学知识科普。

### 安抚路由（Comfort Route）
中-高风险场景的干预路径。侧重点为共情倾听和情绪安抚而非问题解决，不灌输知识或建议。适用于 risk 在 comfort 区间内、情绪明显困扰但未达危机水平的会话。

### 危机路由（Crisis Route）
高风险场景的紧急干预路径。侧重点为安全保障、危机热线引导和即时情感支持。由语义安全评估器裁决为 crisis、risk 高位、安全过滤等级触发升段、或上述叠加触发。被安全短路跳过的请求直接命中此路由。

### 路由段位（Risk Band）
以 risk 值为基础的四个路由区间，决定路由的基线方向。risk ≥ 危机阈值（0.7）落入高危段（crisis），risk ≥ 安抚阈值（0.5）落入中危段（comfort），risk ≥ 知识阈值（0.1）落入低位段（knowledge），其余为底段（general）。段位由 risk 决定，情绪类型不能改变段位归属。路由优先级：高情绪风险先安抚、低情绪风险给知识，避免在用户情绪激动时灌输知识。

### 意图覆盖（Intent Override）
Router 对 intent=information 的纯信息提问（如"什么是利弊分析"、"如何改善睡眠"）的特殊处理：当 risk 段位为 general 时自动提升至 knowledge，使无情绪色彩的知识型提问走 RAG 知识库而非闲聊。仅在 risk 最低段且无情绪信号时生效；有情绪信号时仍由 risk 决定路由。

### 情绪偏向（Emotion Bias）
主情绪类型对路由方向的倾向性影响。在同一个风险段位内，某些情绪更适合特定干预路径（如焦虑适合安抚、困惑适合知识科普）。情绪偏向仅在本段位内生效，不跨段改变路由。

### 路由置信度（Route Confidence）
Router 对分派决策的确定程度（0~1）。受 risk 与阈值的距离、安全等级是否触发升段、情绪偏向是否与段位方向冲突、以及多模态信号是否打架等因素影响。低置信度标识需要更保守的干预策略。

### 干预闭环模块（Intervention Module）
管线第四阶段。根据路由决策生成对用户的回复，包含共情、建议和可选行动项。四种路由对应不同生成策略：general 走 LLM 自然回复（不强行转向心理健康），crisis 走确定性话术模板，comfort 走 LLM 共情生成，knowledge 走 LLM + RAG 知识库检索。模块自身通过 SessionManager 拉取会话历史，不依赖管线传入上下文。

### 干预回复生成器（InterventionReplyGenerator）
干预模块的核心组件。接收管线结论（route / emotion / safety / user_text），按路由分支选择生成策略（general / comfort / knowledge / crisis），组织 prompt 并调用 LLM 适配器产出回复文本。不执行自身的情绪分析或安全判断，管线结论为唯一权威输入。

### 危机话术模板（Crisis Template）
预设的危机应急回复文本，按危机类型（suicide / violence / self_harm / crisis）分类。触发条件：路由为 crisis 或安全短路。危机模板为确定性话术，不经 LLM 生成，确保热线号码和安全引导语不被修改。模板来自 EmergencyPushService。

### 语义安全评估器（Semantic Safety Judge）
判断当前输入是否构成危机的 LLM 组件。读取最近若干轮对话上下文与风险锚点信号，输出三级裁决：crisis（危机）/ probe（需探针确认）/ no_risk（无风险）。静默运行，不直接面向用户。仅在风险锚点触发时被调用。
_Avoid_: 安全裁决器

### 风险锚点（Risk Anchor）
规则层产出的"可能构成风险"的候选信号，是触发语义安全评估器的条件。包括敏感词命中、情绪 risk 超过低阈值、评估器与 SCID 的风险标记。故意设得宽松——宁可多调用评估器，不可漏掉真实风险。
_Avoid_: 敏感词命中（只是其中一种锚点，不是全部）

### 分级响应（Graduated Crisis Response）
语义安全评估器裁决后的分级动作：crisis → 危机路由发确定性模板；probe → 医生继续对话并注入安全探针；no_risk → 正常医生对话。核心原则是"风险词命中不再直接打断对话"。

### 安全探针（Safety Probe）
医生对话中嵌入的结构化风险确认问题（如三选一框架："是真心不想活，还是心里难受，还是都有？"），用于区分"念头"与"计划"。探针问题由 LLM 按当前对话生成，规则提供兜底模板。用户的回答决定是否从探针升级到危机。

### 量表筛查（Scale Screening）
知识路由下由 LLM 判定触发的多轮对话式评估子步骤。选取标准化心理筛查量表（PHQ-9、GAD-7 等），跨多轮自然对话逐题引导用户自评，独立计分模块按量表锚点规则打分。具备双重作用：筛查结果作为知识库检索的强化查询上下文；重度分数或自伤条目可触发路由重评估，上升至危机。筛查结果标注为"自评参考"而非诊断。题目不按原文逐字念，由 LLM 按对话上下文自由组织句式。单次会话最多触发一次，用户连续偏离量表话题时静默放弃。

### 知识库检索（Knowledge Base Retrieval）
知识路由下的一次检索流程。采用统一索引（私有案例 + 公有心理学合并到一个 Chroma 向量库）+ 混合检索（稠密向量语义匹配 + BM25 关键词精确匹配）+ RRF 融合排序。检索前由 LLM 对用户查询做分类，产出 category 和 source 元数据过滤条件，缩小候选集。检索结果作为 prompt 上下文约束 LLM 输出范围，生成回复需注明知识来源。外部 API 兜底已接入：精神科药物名感知触发 Tavily Search API 实时检索（覆盖 56 种常见精神科药物，见 `core/rag/search_fallback.py`），用于补充知识库未覆盖的药物信息。

### 知识库索引（Knowledge Base Index）
私有临床知识和公有心理学知识的统一 Chroma 向量索引。每条知识在入库时标注 source（private/public）和 category 元数据，支持按 source 加权（私有条目 ×1.2 分数加权），以及按 category 过滤。最终 9 类分类（见 ADR 0007）：

- **私有**：clinical（临床咨询技法、家庭治疗、个案概念化）
- **公有**：disorder_knowledge（诊断标准/症状） / coping_strategies（疗法原理+可操作技巧，合并了原 therapy_techniques 和 self_help） / medication_knowledge（精神科药物） / sleep_health（睡眠健康/CBT-I） / trauma_and_stress（创伤知情/心理急救） / grief_and_loss（哀伤/丧失） / relationships（人际/亲密关系/依恋） / psychology_basics（基础概念兜底）

已删除的类别：crisis_intervention（危机路由不走 RAG，删除）、scale_interpretation（量表 JSON 自带解读规则，删除）。

### 查询分类（Query Classification）
检索前由 LLM 对用户原始查询做意图分类，输出应该检索的知识 source 和 category 列表。分类结果作为 Chroma metadata 过滤条件，将候选集从全量缩小到相关子集。分类开销约 200ms，带来的候选集缩减（~1000→~100）显著提升 BM25 在更小集合上的精确度。

### 混合检索（Hybrid Retrieval）
稠密向量检索（百炼 Embedding API → Chroma top-N）+ 关键词检索（jieba 分词 + BM25 top-N），两路结果经 RRF 融合排序后取 top-3。稠密路径解决语义近义（"睡不着觉"→"失眠"），BM25 路径解决精确术语匹配（药名、量表名）。

### 安全短路（Emergency Shortcut）
当安全过滤 blocked 或显式危险信号（P0）命中时，管线跳过情绪分析和路由，直接将 route 置为 crisis、emotion 置为 distress，进入危机干预。此机制确保高危场景零延迟响应。弱风险信号不触发短路，进入语义安全评估器判定。

### TherapyChain（已退役移除）
历史遗留的独立 LLM 对话流程，曾自含情绪分析、安全检查、阶段判定和回复生成。现已移除——`/api/v1/chat` 与 `/api/v1/chat/stream` 均改接合同管线（`pipeline.orchestrator.run_pipeline`），不再存在治疗链路径。

## 情绪分析子概念

### 情绪标签（EmotionTags）
情绪模块输出的统一数据结构，包含：
- `primary_emotion`：主情绪，限定为 8 个标签之一（neutral / anxiety / sadness / anger / fear / stress / happiness / confusion）
- `intensity`：情绪强度，0~1，从语音特征（语速、关键词密度）计算
- `risk`：风险值，0~1，结合安全过滤等级和情绪类别综合计算
- `modality_notes`：各模态的附加元数据
- `intent`：文本意图标签，取值为 information（信息提问）、emotion_expression（情感表达）、casual_chat（闲聊）或 unknown。仅文本情绪引擎产出，用于路由辅助判断

### 语音情绪（Audio Emotion）
从音频信号中提取的情绪信息。使用 SenseVoice 模型从声学特征判断情绪类别，支持 GPU 推理。当音频不可用时以文本关键词做启发式回退。

### 文本情绪（Text Emotion）
从文本内容中提取的情绪信息。采用策略模式，可插拔不同引擎：默认关键词引擎（KeywordEmotionEngine）做零依赖关键词匹配；可选 ONNX 引擎（ONNXEmotionEngine）做语义级情绪分类和意图检测。有音频时作为语音情绪的辅助信号参与融合；无音频时作为唯一信号来源。详见 ADR 0010。

### 文本情绪引擎（Text Emotion Engine）
实现 `TextEmotionEngine` 协议的情绪分类组件，产出 `TextEmotionResult`（含主情绪、置信度、全标签分布、意图标签）。Factory 根据 `EMOTION_ENGINE` 配置选择引擎，ONNX 加载失败时自动降级为关键词引擎，保证服务永不因模型缺失而崩溃。

### 意图检测（Intent Detection）
区分用户输入意图的规则层，由文本情绪引擎产出 intent 标签：
- **information**：信息提问（如"什么是利弊分析"、"怎么缓解焦虑"），触发疑问句式检测
- **emotion_expression**：情感表达（如"我好难过"、"最近压力很大"），第一人称 + 高置信度情绪信号
- **casual_chat**：日常闲聊，纯中性且无提问句式
- **unknown**：无法判定，关键词引擎默认返回

意图标签配合 risk 值参与路由决策：information + low risk 倾向于 knowledge 路由（知识科普），emotion_expression + higher risk 倾向于 comfort 路由（情绪安抚）。

### 视觉情绪（Visual Emotion）
从视频帧中提取的面部表情信息。流程：视频 → 抽帧 → MediaPipe 人脸检测 → HSEmotion 推理（Ekman 7 类：angry / disgust / fear / happy / sad / surprise / neutral）→ 情绪标签映射转为契约 8 标签。多帧结果按时序加权聚合（靠后帧权重更高），无面部帧跳过，有效帧占比低于 30% 时整段视觉信号降级。

### 情绪标签映射（Emotion Label Mapping）
将视觉模型的 Ekman 7 类标签转换为契约 8 类标签的映射规则。直接对应（angry→anger, fear→fear, happy→happiness, sad→sadness, neutral→neutral）保留全置信度；跨域映射（disgust→stress ×0.6, surprise→anxiety ×0.4）施加信心惩罚；anxiety 和 confusion 视觉模型不产生，由文本和音频补充。

### 情绪融合（Emotion Fusion）
将多路情绪信号（语音、文本、视觉）加权合并为统一结果。支持 1~N 路信号输入，单路直接返回，多路按置信度加权聚合。当三路信号的 primary_emotion 全部不同、且最大置信度不超过 0.6 时，信号冲突仲裁介入，融合置信度额外 ×0.7 惩罚。融合结果的 `mixed_signals` 为 True 时，风险值额外 +0.1 预警分。

### 情绪强度（Intensity）
情绪标签的强烈程度，0~1 连续值。由语速因子（字符/秒）和情绪关键词密度因子加权计算，不直接等同于模型置信度。

## 辅助概念

### 语音识别（ASR）
将音频转为文本的过程，默认使用 SenseVoice。与语音情绪分析共享同一模型实例，一次推理同时产出文本和情绪标签。

### Mock / Stub
并行开发的实现策略：
- **Mock**：确定性的假实现，返回固定测试数据，用于单元测试和本地开发
- **Stub**：接口占位，标注了应接入的真实能力的位置。当 Stub 被替换为真实实现后即成为正式模块

### 契约（Contract）
跨模块通信的 JSON 数据格式，定义在 `schemas/contracts/v1.py`。所有模块间传参必须遵循契约模型，变更需升级版本号并通知团队。当前版本 `1.4`。

### 视频预处理（Video Preprocessor）
管线的前置步骤，负责将视频文件解构为管线可用的信号。流程：视频 → ffmpeg 分离音频 → 自适应抽帧（min(时长×1fps, 20) 帧）→ 每帧人脸检测 + 情绪推理 + 标签映射 → 时序加权聚合产出 `visual_emotion` → SenseVoice 一次推理产出 ASR 文本 + `audio_emotion`。输入为视频文件和可选的独立音频文件，输出为 text、audio_emotion、visual_emotion 三路信号。

### 实时语音对话（Real-time Voice Conversation）
管线的一种新入口模式：用户通过按住说话（push-to-talk）发送语音，AI 以流式语音回复，交互载体为 WebSocket 双向通道。与回合制文字对话共享同一管线逻辑（安全→情绪→路由→干预），差异在传输层和交互方式。按住期间推送 Opus/WebM 音频 chunk，松手后走串行管线 → 情绪融合 → LLM 流式生成 → 逐句 TTS 合成 → 二进制音频流推送前端播放。语音语调情绪由 SenseVoice 在 ASR 时一并产出，与文本情绪经情绪融合层合并。详见 ADR 0012。

### WebSocket 连接（WebSocket Connection）
实时语音对话的传输通道。一条 WebSocket 连接对应一个会话，生命周期覆盖整段对话。连接建立时通过 URL 参数携带 JWT access token 认证。消息协议为 JSON 控制帧 + 二进制音频帧混合，以首个字节区分类型。

### 按住说话（Push-to-Talk）
实时语音对话的输入交互模式。用户按住录音按钮期间持续推送音频 chunk（MediaRecorder + Opus/WebM），松手时发送 `audio.end` 触发管道处理。滑出按钮区域松手则发送 `cancel`，服务端清空音频缓冲区。

### 语音取消（Voice Cancel）
按住说话的安全兜底机制。用户在录音过程中滑出按钮区域并松手，前端发送 `cancel` 帧，服务端收到后清空已缓冲的音频数据，不触发管道处理。

### 逐句合成（Sentence-by-Sentence Synthesis）
TTS 流式推送的触发策略。LLM 流式输出 token 时按语义边界积累文本——遇到句号、问号、感叹号或换行符时，将已积累的完整句子送 Edge TTS 合成，边合成边推送音频 chunk。首句合成即前端开始播放，后续句子异步追加。避免在词中间切断的同时保持低首音延迟。

### WebSocket 管道服务（WebSocket Pipeline Service）
WebSocket 版的管道入口（`pipeline/ws_pipeline.py`）。输入为音频 bytes，内部依次调用 ASR → Safety → Emotion → Router → Intervention → TTS，通过回调函数接口推送状态更新（status / text.delta / audio.delta / emotion.result / done / error）至 WebSocket handler。与 HTTP 版管道共享核心模块，差异仅为输出方式（回调推送 vs 返回值）。

### 模态降级（Modality Degradation）
当某路情绪信号无法获取时的静默处理策略。文本（ASR 产出）为必选项，不可降级；音频和视觉为加分项，任一失败时跳过该路、经由剩余信号继续融合，并在 `modality_notes` 中记录降级原因。视觉信号内部，当人脸检测帧占比低于 30% 时整段降级为不可靠。

### 信号冲突仲裁（Signal Conflict Arbitration）
情绪融合阶段对多路信号不一致的检测与惩罚机制。当文本、音频、视觉三路全部给出不同的 primary_emotion，且最高置信度不超过 0.6 时，判定为"高度不一致"，融合结果的置信度额外乘以 0.7。此机制利用信号冲突本身作为诊断信息，而非仅依赖单路置信度。

## 用户体系

### 用户（User）
系统的自然人主体。一个真实自然人对应一个 user，user_id 为永久业务标识（UUID）。用户拥有多条会话、情绪记录、量表筛查记录和安全标记。用户与登录凭证解耦——user 是数据归属，credential 是认证方式。

### 登录凭证（Credential）
用户用于证明身份的方式。与 user_id 解耦，支持多种类型可插拔（password / phone / wechat 等），同一 user 可绑定多个 credential。当前首版使用账号+密码，后续可追加手机号、微信 openid 等，user_id 不变。

### 访问令牌（Access Token）
JWT，有效期 30 分钟。用户登录后获得，每次请求通过 `Authorization: Bearer <token>` header 携带。过期后用 refresh token 刷新。短有效期限制泄露影响窗口。

### 刷新令牌（Refresh Token）
JWT，有效期 30 天。仅用于换取新的 access token，不直接用于业务请求。服务端可吊销。

### 账号软删除（Soft Delete）
用户发起删除请求后，user.status 标记为 deleted，所有 token 立即吊销。30 天后悔期内可人工恢复；期满后定时任务物理删除所有关联数据（credentials、messages、emotion_records、scale_screenings、safety_flags）。

### 用户数据导出（User Data Export）
用户有权导出其全部个人数据（全部会话 + 情绪记录 + 量表筛查 + 安全标记历史），JSON 格式。导出需验证当前密码。前端个人信息页展示关键摘要（情绪趋势、量表历史），完整数据通过 `GET /api/v1/user/export` 下载。

## 安全治理

### 安全标记（Safety Flag）
安全过滤模块产出的 level 值落库记录，独立于情绪记录。每条 flag 包含 level（0/1/2）、blocked、matched_terms 和是否已人审（reviewed）。level=1 的记录不再仅是日志行，而是有生命周期的可查询事件。

### 安全标记累积规则（Safety Flag Accumulation Rule）
纯规则自动化的安全升级机制。同一 user 在滑动时间窗口内 level=1 累计达到阈值次数，自动触发路由升段至 crisis。无需人工介入，人审接口预留。此规则弥补"无 7×24 值班"下 level=1 无人盯防的 gap。

### 字段级加密（Field-Level Encryption）
对数据库中敏感内容字段做 AES 加密后存储。加密范围：messages.content、emotion_records.context、safety_flags.matched_terms。元数据字段（session_id、user_id、emotion 标签名、risk 数值、时间戳）保持明文以支持索引和统计。密钥存于环境变量，不入代码、不入数据库、不入镜像。

### 速率限制（Rate Limiting）
基于 slowapi + Redis 的 per-user/per-IP 频率控制。默认 60 req/min/user；登录注册防暴力破解分别为 3/10 req/min/IP；昂贵操作（视频安全检测、TTS）更低。危机相关端点（emergency-push）永不限流。

### 知情同意（Informed Consent）
首次使用必须弹窗展示知情同意书，用户明确同意后方可使用。内容涵盖：AI 辅助工具的边界（非医疗诊断）、数据收集范围、用户权利（查看、导出、删除）。每条 AI 回复底部标注"AI 辅助回复，非医疗诊断"。

### 结构化日志（Structured Logging）
统一 logger 替代所有 print()。每条日志必须携带 request_id、user_id、session_id 用于关联追踪。三级分类：WARNING（用户不受影响）、ERROR（组件异常但服务可用）、CRITICAL（影响安全/危机判断，需主动 webhook 推送至钉钉/飞书）。

### 操作手册（Runbook）
一页 markdown 运维文档，覆盖：服务挂了如何重启、数据库连不上如何排查、依赖服务（LLM API）挂了如何处理、紧急停服流程。不设 7×24 值班，CRITICAL 告警次日人工审查。
