# 视觉情绪标签映射：Ekman 7 类 → 契约 8 类，不做模型重训练

视觉情绪识别使用 HSEmotion 模型，输出 Ekman 7 类基本表情（angry, disgust, fear, happy, sad, surprise, neutral）。管线契约定义的 8 类情绪标签（neutral, anxiety, sadness, anger, fear, stress, happiness, confusion）中包含三个视觉模型无法直接从面部肌肉运动中辨识的类别（anxiety, stress, confusion）。我们决定不重新训练模型，而是通过带信心惩罚的映射表将 Ekman 标签转为契约标签，缺失的标签交由文本和音频信号补充。

## 为什么

- 重新训练一个输出 8 类标签的视觉模型在科学上是困难的：anxiety、stress、confusion 是内在认知/生理状态而非面部表情，即使人类咨询师也需要结合语音和语义来判断。训练会过拟合训练集但在真实推理中乱猜。
- 映射 + 信心惩罚方案无需额外标注数据，无需重新训练，GPU 预算不变。
- 融合层的加权平均机制自然地在视觉不擅长的维度上让文本和音频信号占主导。

## 映射规则

| Ekman 标签 | 契约标签 | 置信度系数 | 说明 |
|------------|----------|-----------|------|
| angry | anger | 1.0 | 直接对应 |
| fear | fear | 1.0 | 直接对应 |
| happy | happiness | 1.0 | 直接对应 |
| sad | sadness | 1.0 | 直接对应 |
| neutral | neutral | 1.0 | 直接对应 |
| disgust | stress | 0.6 | 厌恶常伴随紧张感，但不完全等价 |
| surprise | anxiety | 0.4 | 惊讶可能为恐惧前兆，高不确定性 |
| — | anxiety | — | 视觉不产生，文本/音频补充 |
| — | confusion | — | 视觉不产生，文本/音频补充 |

## 考虑过的方案

- **重新训练 8 类视觉模型**：需要大规模标注数据（百万级面部标 anxiety/stress/confusion），且三类的面部表达没有稳定模式，模型推理质量堪忧。不选。
- **纯映射无惩罚**：将 disgust 和 surprise 以全置信度映射到 stress 和 anxiety。风险是视觉信号在它不擅长的维度上过度影响融合结果。不选。
