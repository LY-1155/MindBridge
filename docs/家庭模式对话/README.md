# 家庭模式对话（DOCTOR_MODE）

让 PRISM 像真实心理医生一样，通过自然对话做家庭系统评估与干预，聚焦青少年群体。

## 一、这个功能解决了什么

PRISM 原有的四阶段管线（Safety → Emotion → Router → Intervention）能区分四种路由并生成对应回复，但停留在"心理筛查师 + 知识科普"模式——像客服，不像医生。

本功能让系统**像真实心理医生一样对话**：

- 前台：用家庭系统语言对话（不贴 DSM 标签、说"心情不好"不说"抑郁症"、用循环提问挖掘家庭互动模式）
- 后台：静默追踪临床指标（SCID 标准对照、风险升级、精准检索 query 生成）

## 二、设计依据

基于 **1441 份真实心理医生访谈数据**（周医生门诊语料，3 个数据集）的深度分析：

| 发现 | 结论 |
|---|---|
| 周医生前台从不用 DSM 标签 | 说"心情不好"不说"抑郁症"，说"叛逆"不说"对立违抗障碍" |
| 但后台做系统性临床筛查 | 睡眠、食欲、注意力、自杀、精神病性症状逐项排查 |
| 治疗模型是折中整合 | 家庭系统 + CBT + PM+ + 存在主义，按场景切换 |
| 核心临床信念 | "症状是沟通""问题是建构的""觉察是改变的前提" |
| 家庭系统视角 | 青少年问题的深层病因常是家庭系统功能失调，不是孩子个体"疾病" |

**架构决策：B 前台 + A 后台。** 对话层用家庭系统语言（Model B），后台静默追踪临床 criteria（Model A）。SCID 追踪结果绝不直接出现在对话中，仅影响安全升级和知识库检索方向。

## 三、改了什么

| 文件 | 操作 | 说明 |
|---|---|---|
| `config/settings.py` | 修改 | 新增 `DOCTOR_MODE`（总开关）、`DOCTOR_PERSONA`（persona 选择） |
| `core/memory/session_memory.py` | 修改 | SessionMetadata 新增 `phase` / `family_members` / `working_hypothesis` / `scid_flags` + 便利方法；SessionManager 加内存缓存 fallback |
| `modules/assessment/family_assessor.py` | 新建 | 家庭系统评估器：阶段推进、循环提问引导、安全红线、工作假设 |
| `modules/assessment/scid_tracker.py` | 新建 | 静默 SCID-5 追踪器：MDD/GAD/Panic/PTSD 关键词对照、criteria 累积、风险标记 |
| `modules/intervention/persona.py` | 新建 | 周医生 persona + 三套家庭版 prompt（comfort/knowledge/general） |
| `modules/intervention/generator.py` | 修改 | `DOCTOR_MODE=true` 时自动切换周医生 prompt |
| `modules/intervention/service.py` | 修改 | 集成 Assessor + Tracker（共享方法 `_run_doctor_assessment()`） |
| `.env` | 修改 | `DOCTOR_MODE=true` |
| `tests/test_family_assessor.py` | 新建 | 16 个单元测试 |
| `tests/test_scid_tracker.py` | 新建 | 17 个单元测试 |
| `scripts/verify_doctor_mode.py` | 新建 | 真实 LLM 验证脚本 |

## 四、怎么启用

`.env` 加一行：

```
DOCTOR_MODE=true
```

默认 `false`，不影响原有行为。

启动（必须用 emotion conda 环境）：

```powershell
& "D:\Anaconda\envs\emotion\python.exe" run_server.py
```

打开 http://localhost:8000/chat 验证。

## 五、验证结果

- **单元测试**：33/33 通过
- **真实 LLM 验证**（qwen3.7-max，4 轮家庭场景对话）：周医生 persona 生效、家庭成员识别生效、SCID 追踪生效、跨轮状态累积生效
- 对话样例见 [验证记录.md](验证记录.md)

## 六、数据流

```
用户消息 → Safety → Emotion → Router → InterventionService
                                            │
              ┌─────────────────────────────┘
              │ DOCTOR_MODE=true?
              ├─ FamilySystemAssessor: phase推进 + 探针方向 + 家庭假设
              ├─ SCIDTracker: 静默criteria匹配 + 风险标记(→crisis) + 精准检索query
              └─ Generator: 周医生prompt + assessor上下文 → LLM回复
```

## 七、后续方向

- [ ] Phase 2：危机协议增强（从周医生 crisis protocol 提取三选一框架、系统筛查清单）
- [ ] 更多 session 类型模式（首次访谈 / 家庭 / 个体 / 家长单独）
- [ ] 假设生成时机优化（当前绑定 explore 阶段，可在 check_in 后期提前）
- [ ] `USE_DATABASE=true` 启用持久化，session 状态跨重启保留
