# Cursor 模块开发需求提单模板（四模块通用）

用于给 Cursor 发开发需求时直接复制填写。默认适用于：

- `modules/**`
- `pipeline/**`
- `api/routes/parallel_modules.py`
- `api/routes/pipeline.py`
- `schemas/contracts/**`

提交前请先阅读：`docs/module_development_global_context.md`。

---

## 0. 任务标题

`[模块名] + [动作] + [目标]`

示例：`[Router] 优化危机优先级判定并补齐边界测试`

---

## 1. 背景与目标（必填）

- 业务背景：
- 当前问题：
- 目标结果（可量化）：

示例（可删）：
- 当前 `risk=0.69/0.70` 边界行为不稳定。
- 目标是稳定输出可解释 `reason`，并确保边界用例全通过。

---

## 2. 影响范围（必填）

- 模块：`safety / emotion / router / intervention / pipeline`
- 预计改动文件：
  - `...`
  - `...`
- 不应改动文件（防止越界）：
  - `...`

---

## 3. 契约要求（必填）

- `contract_version`：`1.0`（默认不升级）
- 是否涉及字段变更：`否 / 是（仅新增可选） / 是（破坏性）`
- 变更详情：
  - 输入字段：
  - 输出字段：
  - 默认值与兼容策略：

如为破坏性变更，必须补充：
- 升级版本：
- 迁移方案：
- 受影响端点：

---

## 4. 业务规则（必填）

按“规则编号 + 条件 + 动作 + 优先级”写清楚。

- R1：
- R2：
- R3：

示例（可删）：
- R1：若 `safety.blocked=true` 或 `level>=2`，强制 `route=crisis`，优先级最高。
- R2：`0.3 <= risk < 0.7`，`route=knowledge`。

---

## 5. 实现要求（必填）

- 允许的实现方式：
  - `Mock/Stub/Real` 哪种要改：
  - 是否只改 `modules/factory.py` 装配：
- 错误处理与降级：
  - 异常时最小可用输出：
  - 必须保留的 `reason/meta`：
- 性能目标（可选）：
  - `comfort <= 3s` / `knowledge <= 5s` / `crisis <= 2s`

---

## 6. 测试与验收（必填）

- 必须新增/更新的测试：
  - 模块接口测试：
  - 编排测试：
  - 全链路冒烟（如适用）：
- 关键验收用例：
  1. 正常路径：
  2. 异常路径：
  3. 边界路径：
- 本地运行命令：
  - `python -m pytest ...`

---

## 7. 输出格式要求（建议）

要求 Cursor 回答时包含：

1. 改动了哪些文件。
2. 为什么这样改（对应规则编号）。
3. 如何验证（命令 + 预期结果）。
4. 风险与回滚方案。

---

## 8. 回滚策略（建议）

- 回滚开关（如 `MOCK_*`）：
- 可直接回退的文件：
- 需要人工介入的数据/配置：

---

## 9. 可直接复制的提问模板

```text
请按 docs/module_development_global_context.md 作为全局上下文，完成以下开发任务：

【任务标题】
[在这里填]

【背景与目标】
[在这里填]

【影响范围】
[在这里填]

【契约要求】
[在这里填]

【业务规则】
[在这里填]

【实现要求】
[在这里填]

【测试与验收】
[在这里填]

【输出要求】
请给出：改动文件、改动理由、验证命令、风险与回滚方案。
```

---

## 10. 四模块快捷提单示例（精简版）

### A. Safety

`完善高风险词规则，保证命中后 blocked=true，并补充 matched_terms 与 meta 审计字段。`

### B. Emotion

`优化 risk 归一化与 modality_notes 缺失标注，补齐中性/负向/高压样例测试。`

### C. Router

`实现 crisis > knowledge > comfort 的优先级与边界值测试（0.29/0.30/0.69/0.70）。`

### D. Intervention

`细化 comfort/knowledge/crisis 三分支，其中 crisis 必须触发 emergency_notify()（可先日志模拟）。`
