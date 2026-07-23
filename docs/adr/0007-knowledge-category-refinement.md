# 知识库分类从 7 类精简为 9 类（合并+删除+补缺）

原知识库在 ADR 0006 中定义了 7 个公有类别，但实际建设中发现部分类别无存在必要，且存在覆盖缺口。经逐类审视，最终确定为 1 个私有类别 + 8 个公有类别 = 9 类。

## 为什么

### 删除的 2 个类别

- **crisis_intervention（危机干预标准流程）**：crisis 路由走 EmergencyPushService 确定性话术模板，不经 LLM 生成，整个 RAG 链路被短路。此类别在任何路由下都不可能被检索到。
- **scale_interpretation（量表解读）**：每个量表 JSON 文件自身已携带 thresholds 解读规则，量表筛查模块走独立计分而非 RAG。PHQ-9 几分算抑郁是确定性问题，不需要向量检索。

### 合并的 2 个类别

- **therapy_techniques + self_help_strategies → coping_strategies**：两者边界在实际 LLM 分类中无法可靠区分。"焦虑怎么缓解"同时涉及疗法原理和行动技巧，用户不关心知识来源标签。合并为 coping_strategies 涵盖疗法原理 + 可操作应对技巧，消除分类歧义。

### 新增的 4 个类别

- **sleep_health**：失眠是心理健康求助第一主诉，与抑郁/焦虑高度共病。CBT-I 是独立知识体系，无法归入其他类别。
- **trauma_and_stress**：系统已有 PCL-5 筛查，筛查偏高进入 knowledge route 时必须有不偏诊断学的操作层知识支撑（安全稳定化技术、解离地面等）。
- **grief_and_loss**：哀伤是"正常但痛苦"的低风险核心场景。缺此类别时，低风险用户进入 knowledge route 会被导向疾病话语，反而有害。
- **relationships**：私有 clinical 是咨询师视角，公有层缺一般性的人际关系知识（依恋科普、沟通模式、冲突解决）。

### 新增兜底类别

- **psychology_basics**：原 psychology_basics.jsonl 已存在但未纳入分类体系。作为兜底，承接模糊的基础心理学概念查询（情绪调节、认知偏差等），避免分类器在不确定时退回不过滤。

## 考虑过的方案

- **psychosomatic 独立类别**：躯体症状虽是中国特色高发主诉，但其核心知识（躯体形式障碍、疾病焦虑）可归入 disorder_knowledge 的躯体症状章节。用户以"头疼查不出毛病"开头时，分类器应导向 disorder_knowledge + coping_strategies。不单列。
- **addiction 独立类别**：AUDIT 只覆盖酒精，但中国年轻群体问题在网络/游戏成瘾。当前数据缺乏决定暂不单列，未来按需扩展。
- **child_adolescent 独立类别**：儿童青少年发展与教养是独立知识域，但用户画像未确认，暂不单列。
- **retain crisis_intervention**：保留但标记为 disabled。不选，死代码不如删干净。

## 影响

- `data/knowledge/sources.json`：新增 `categories` 字段显式声明 9 类，版本升至 v3
- `core/rag/query_classifier.py`：`CLASSIFIER_SYSTEM_PROMPT` 替换为 8 公有类别的判断指南和示例
- `CONTEXT.md`：知识库索引概念更新为最终 9 类，标注删除/合并原因
- `data/knowledge/public/`：新增 grief_and_loss.jsonl、relationships.jsonl、sleep_health.jsonl、trauma_and_stress.jsonl、coping_strategies.jsonl 空文件（待填充）；删 crisis_intervention.jsonl、scale_interpretation.jsonl、self_help_strategies.jsonl、therapy_techniques.jsonl（如存在空文件则清理）
- 分类器 prompt 变短（7→8 类但有判断指南和更多示例），分类歧义减少，准确率预期提升
