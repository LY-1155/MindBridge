# 文本情绪识别：策略模式 + ONNX 引擎 + 关键词降级

将 `EmotionService` 中的 `_text_emotion` 关键词匹配替换为可插拔的情绪分类引擎架构，支持 ONNX 小模型语义推理和意图检测，保留关键词引擎作为零依赖 fallback。

## 为什么

### 为什么不能用纯关键词匹配

- 无否定处理："我不焦虑" 命中 "焦虑" → 错误分类为 anxiety
- 无法区分意图："什么是利弊分析" 和 "我很焦虑" 在关键词视角下无法区分前者是信息提问
- 漏检率高："心里堵得慌"、"喘不过气" 等口语化表达不在词表内
- 误检：疑问句式 + 情绪词组合（"怎么缓解焦虑"）被归类为情绪表达

### 为什么选 ONNX 而非调 LLM 做情绪分析

- 延迟：ONNX 本地 CPU 推理 ~15-30ms（R9 8940），LLM 网络调用 200-800ms
- 成本：ONNX 零 API 调用费用；LLM 每条请求都要 token 消耗
- 确定性：ONNX 模型输出可重现，LLM 输出随温度波动
- 管线已有 LLM 调用（干预阶段），情绪阶段再调一次会翻倍延迟

### 为什么关键词保留为 fallback 而非直接废弃

- ONNX 模型文件 ~400MB，需要用户显式下载运行脚本
- `onnxruntime` 是可选项，不是项目强制依赖
- 关键词引擎零依赖、零延迟、始终可用，作为兜底保证服务永不因模型缺失而崩溃

### 为什么策略模式而非 if-else 堆砌

- 未来的情绪引擎可以替换（比如换更大模型、加意图模型），只需实现 `TextEmotionEngine` 协议
- 工厂调用处只根据 `EMOTION_ENGINE` 配置选择引擎，代码路径单一
- 引擎和 `EmotionService` 解耦，各自的单元测试互不干扰

## 架构

```
modules/emotion/
    base.py              TextEmotionEngine 协议 + TextEmotionResult 数据类
    keyword_engine.py    关键词匹配引擎（提取原 _text_emotion 逻辑）
    onnx_engine.py       ONNX Runtime 推理引擎
    stub.py              EmotionService（注入 text_engine，默认 keyword）
```

### 引擎协议

```python
class TextEmotionEngine(Protocol):
    def predict(self, text: str) -> TextEmotionResult: ...
    @property
    def model_name(self) -> str: ...
    @property
    def is_ready(self) -> bool: ...
```

### TextEmotionResult 新增字段

- `all_emotions: Dict[str, float]` — 8 种情绪的概率分布（替代旧的硬编码空 dict）
- `confidence: float` — 引擎对主情绪的置信度（替代旧的硬编码 0.3）
- `intent: str` — 意图标签：information / emotion_expression / casual_chat / unknown

### 意图检测

`onnx_engine._detect_intent` 结合模型输出 + 规则：
- 疑问句式（"什么是"、"怎么"、"？"等）→ information，即使模型输出有情绪词
- 第一人称 + 高置信度负面情绪 → emotion_expression
- 纯中性 → casual_chat

关键词引擎 intent 固定返回 "unknown"。

### 强度计算适配

- 关键词引擎：沿用 `_compute_intensity`（关键词密度 + 语速因子）
- ONNX 引擎：`_compute_intensity_from_model`（模型负面情绪概率 ×0.6 + 强度关键词 ×0.4）

### 降级链

```
EMOTION_ENGINE=onnx
    → ONNXEmotionEngine 加载成功 → 使用
    → 加载失败（缺模型/缺 onnxruntime）→ print warning → KeywordEmotionEngine
```

### 配置

```python
# config/settings.py
EMOTION_ENGINE: str = "keyword"       # "keyword" | "onnx"
EMOTION_ONNX_MODEL_PATH: str = "models/emotion_classifier/model.onnx"
EMOTION_ONNX_TOKENIZER_PATH: str = ""
```

## 影响

- `EmotionTags` 新增 `intent: Optional[str]` 字段，`contract_version` → 1.4
- `EmotionService.__init__` 新增可选参数 `text_engine`，默认 keyword，向后兼容
- 文本信号 `confidence` 从硬编码 0.3 变为引擎实际置信度
- `modality_notes` 新增 `text_engine` 字段标识当前引擎
- 下游 Router/Intervention 的 `emotion` 字段为 `Dict[str, Any]`，新增字段自动透传
