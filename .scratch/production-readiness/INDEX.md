# 生产就绪 Gap 跟踪

2026-07-03 grill-with-docs 产出。22 条 gap 按依赖关系排序。

---

## 阻塞链（必须按顺序做）

| # | Gap | 状态 | 依赖 |
|---|-----|------|------|
| 1 | **数据库推倒重做** — 新表（users、credentials、safety_flags、scale_screenings），修复 emotion_records 字段类型（intensity float、risk float），session_id 改完整 UUID，接 Alembic | ✅ done | — |
| 2 | **用户体系** — user_id 与 credential 解耦，先账密上线，预留手机号/微信绑定扩展口 | ✅ done | #1 |
| 3 | **认证中间件** — JWT access(30min) + refresh(30d)，Bearer header，4 个 auth 端点 | ✅ done | #2 |
| 4 | **依赖注入重构** — 所有端点通过 JWT payload 获取当前 user_id，session 必须绑定 user | ✅ done | #3 |

## 安全与隐私

| # | Gap | 状态 |
|---|-----|------|
| 5 | **Emergency Push 真实通路** — 替换 print()，feature flag 区分环境（dry-run / 生产） | ✅ done |
| 6 | **安全标记累积规则** — 同一 user 滑动窗口内 level=1 累计 N 次自动升 crisis，人审接口预留 | ✅ done |
| 7 | **字段级 AES 加密** — messages.content、emotion_records.context、safety_flags.matched_terms，密钥环境变量 | ✅ done |
| 8 | **HTTPS + Nginx 反向代理** — Nginx 做 TLS 终止，转发到 uvicorn 内网 | ✅ done |
| 9 | **`.env` 不入 git + 不入镜像** — 确认 .gitignore 生效，清理现有提交中的凭证 | ✅ done |
| 10 | **Prompt 注入基础防御** — instruction hierarchy，user_text 在 prompt 中明确边界包裹 | ✅ done |
| 11 | **文件上传安全** — 大小限制 + filetype magic bytes 验证真实文件类型 | ✅ done |
| 12 | **知情同意 + AI 标注** — 首次弹窗 + 每条回复底部 "AI 辅助回复，非医疗诊断" | ✅ done |

## 用户权利

| # | Gap | 状态 |
|---|-----|------|
| 13 | **账号软删除 + 30 天后悔期** — user.status=deleted → token 立即吊销 → 30 天宽限期 → 物理删除 | ✅ done |
| 14 | **用户数据导出** — `GET /api/v1/user/export` JSON 全量导出，需密码验证；前端个人信息摘要页面 | ✅ done |

## 基础设施与运维

| # | Gap | 状态 |
|---|-----|------|
| 15 | **Docker 生产化** — `.dockerignore`、多阶段构建、非 root 用户、健康检查 | ✅ done |
| 16 | **gunicorn + 多 uvicorn worker** — 替换 `python run_server.py` 单进程 | ✅ done |
| 17 | **速率限制** — slowapi + Redis，全局 60 req/min/user，crisis 接口不限流 | ✅ done |
| 18 | **数据库备份确认** — 上线前确认阿里云 RDS 自动备份策略已启用 | ✅ done |
| 19 | **SessionManager 重构** — 去除内存 `_sessions` dict 和 `_messages` list，Redis 做热数据、MySQL 做持久化 | ✅ done |

## 可观测性

| # | Gap | 状态 |
|---|-----|------|
| 20 | **结构化日志** — kill 所有 print()，统一 logger + request_id/user_id/session_id，WARNING/ERROR/CRITICAL 三级 | ✅ done |
| 21 | **CRITICAL 告警 webhook** — 钉钉/飞书机器人推送 CRITICAL 级别事件 | ✅ done |
| 22 | **操作手册（Runbook）** — 一页 markdown：重启、数据库排查、依赖服务故障、紧急停服；不设 7×24 值班 | ✅ done |

---

## 不阻塞生产但已记录

| Gap | 备注 |
|-----|------|
| CI/CD（GitHub Actions pytest + lint） | 上线后补，不阻塞首次部署 |
| 前端流式输出 | 体验优化，当前请求-响应可用 |
| 负载测试（locust/k6） | 有真实用户后按流量压测 |
