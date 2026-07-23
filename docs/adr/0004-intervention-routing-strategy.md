# 干预模块按路由分支差异化生成策略

干预模块是管线第四阶段，输入为前序阶段产出的 route（comfort/knowledge/crisis）、emotion 标签和 safety 结论。我们决定对三种路由使用不同生成策略：crisis 走确定性话术模板、comfort 走 LLM 共情生成、knowledge 走 LLM + RAG 知识检索，而非统一用一种方式生成所有回复。

## 为什么

- **Crisis 不能有 LLM 幻觉**。危机场景下，热线号码、安全引导语一个字都不能错。确定性话术模板（已由 EmergencyPushService 维护）经过审核，零风险。
- **Comfort 是 LLM 的强项**。共情倾听、反映感受、确认体验——这些不需要外部知识，LLM 天然擅长。不需要 RAG 增加延迟。
- **Knowledge 需要内容准确性**。心理学知识科普可能被 LLM 编造成看起来专业但实际错误的建议。RAG 检索约束输出范围是必要保护层。
- **不从 TherapyChain 复用**。TherapyChain 自身包含情绪分析、安全判断和阶段判定，会与管线已算出的结论冲突。干预模块只应调用 LLM 回复生成能力，不重复分析。

## 考虑过的方案

- **全部走 LLM 生成**：crisis 也一样走 LLM。优点是实现统一，缺点是危机场景下 LLM 可能改掉热线号码或说出不当的危机干预语句，不可接受的风险。
- **全部走模板**：失去 comfort 的自然流畅和 knowledge 的知识检索能力，相当于回到 FAQ 机器人时代。不选。
- **复用 TherapyChain.chat()**：会把管线已算好的情绪、安全、路由结论丢掉，让 TherapyChain 重算一遍。两套分析结论可能打架。不选。
