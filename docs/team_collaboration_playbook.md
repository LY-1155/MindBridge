# 四模块并行开发协同手册（会议版）

本文用于团队会议统一开发模式、分工边界、联调方式与交付标准。  
适用当前四模块分工：

1. 输入与安全过滤  
2. 情感分析  
3. 智能路由  
4. 干预闭环

---

## 1. 开发模式

采用 **“契约先行 + 模块并行 + Mock 解耦 + 流水线集成”** 的方式协作。

- 契约先行：先定 JSON 输入输出与版本，再写实现。
- 模块并行：四个小组独立开发，不互相阻塞。
- Mock 解耦：未完成模块先用 Mock/Stub 占位，保证全链路可跑。
- 流水线集成：统一用 `pipeline/run` 做端到端联调。

---

## 2. 当前代码中的落地点

### 核心目录

- 契约模型：`schemas/contracts/v1.py`
- 样例数据：`schemas/contracts/samples/`
- 模块接口（协议）：`modules/ports.py`
- 模块实现（Mock/Stub）：`modules/safety|emotion|router|intervention/`
- 模块装配：`modules/factory.py`
- 运行时实例管理：`modules/runtime.py`
- 统一编排：`pipeline/orchestrator.py`
- 单模块 HTTP：`api/routes/parallel_modules.py`
- 全链路 HTTP：`api/routes/pipeline.py`

### 关键接口

- 安全：`POST /api/v1/modules/safety/check`
- 情感：`POST /api/v1/modules/emotion/analyze`
- 路由：`POST /api/v1/modules/router/route`
- 干预：`POST /api/v1/modules/intervention/run`
- 四段串联：`POST /api/v1/pipeline/run`

---

## 3. 团队分工建议（按模块 Owner）

每个模块至少指定 1 位主 Owner + 1 位备份：

- **A组（安全）**：词库、规则、分级、阻断策略
- **B组（情感）**：多模态融合、标签与风险分值
- **C组（路由）**：规则/模型、可解释路由原因
- **D组（干预）**：安抚链、知识链、危机链输出结构

每组必须负责：

1. 维护本模块契约兼容性（输入/输出不随意改）
2. 维护本模块样例 JSON（`schemas/contracts/samples`）
3. 提供本模块测试（单测 + 必要集成测）
4. 更新文档（至少更新本手册与 IO 样例文档）

---

## 4. 协同规则（防止互相卡住）

### 4.1 契约变更规则（强约束）

- 默认 `contract_version = "1.0"`。
- **非破坏性变更**：可新增可选字段，但不能改现有字段语义。
- **破坏性变更**：必须升级版本并在会议中确认迁移计划。
- 变更契约前必须通知相关模块 Owner。

### 4.2 模块依赖规则（解耦）

- 模块间传输统一使用 JSON（`model_dump()` 后的 dict）。
- 禁止跨模块直接依赖对方内部实现细节。
- 编排层只依赖 `ports.py` 协议，不依赖具体实现类。

### 4.3 Mock/Stub 规则（并行）

- 未完成能力先在 `modules/*/stub.py` 保底。
- 联调默认可用 Mock（`MOCK_* = true`）。
- 模块转真实实现时，仅修改工厂装配点，不改调用链。

---

## 5. 标准开发流程（每个小组都按这个节奏）

1. 明确本模块目标与输入输出字段  
2. 先补/改契约与样例 JSON  
3. 写实现（Mock/Stub/Real）  
4. 写测试（至少 1 个成功路径 + 1 个异常路径）  
5. 跑本地测试后提 PR  
6. 合并后做一次 `pipeline/run` 全链路回归

---

## 6. 测试策略（像 Java 的分层测试）

### 6.1 单模块测试（对应接口级）

- 示例：`tests/test_api_intervention.py`
- 目标：验证本模块 HTTP、契约字段、错误码。

### 6.2 编排测试（对应服务层）

- 示例：`tests/test_pipeline_orchestrator.py`
- 目标：验证编排顺序、安全短路逻辑、四段输出齐全。

### 6.3 全链路接口测试（对应集成测试）

- 示例：`tests/test_api_pipeline_parallel.py`
- 目标：调用 `/api/v1/pipeline/run` 验证四模块联通。

### 6.4 运行命令（Windows 推荐）

```powershell
python -m pytest tests/test_api_intervention.py -v
python -m pytest tests/test_pipeline_orchestrator.py -v
python -m pytest tests/test_api_pipeline_parallel.py -v
```

若要打印用例输入输出：

```powershell
python -m pytest tests/test_api_intervention.py -v -s --print-io
```

---

## 7. 提交与评审清单（PR Checklist）

每个 PR 需自查：

- 是否只改了本模块职责范围？
- 契约字段有无破坏性变更？
- 样例 JSON 是否同步？
- 测试是否覆盖成功与失败路径？
- 文档是否更新（至少 1 处）？
- `pipeline/run` 是否仍可用？

---

## 8. 建议会议节奏

### 每周固定协同会

1. **5 分钟**：四组状态（完成/阻塞/风险）
2. **10 分钟**：契约变更评审（是否版本升级）
3. **10 分钟**：联调问题与责任归属
4. **5 分钟**：下周里程碑确认

### 日常异步同步（建议）

- 每天在群里同步：昨天完成 / 今天计划 / 当前阻塞
- 遇到契约争议优先在文档定稿，不口头漂移

---

## 9. 里程碑建议

### M1（接口打通）

- 四模块 Mock/Stub 均可调用
- `/api/v1/pipeline/run` 稳定返回四段 JSON

### M2（能力替换）

- 至少 2 个模块替换为真实实现
- 契约不破坏，测试继续全绿

### M3（生产前联调）

- 四模块真实实现联通
- 关键场景回归（常规、风险、危机场景）

---

## 10. 对外统一口径

“我们不是按代码文件拆分，而是按模块职责拆分；  
先把接口契约定死，再用 Mock 保障并行；  
每个组只对自己的输入输出负责，最后由 `pipeline/run` 做统一集成验证。”

