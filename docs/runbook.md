# 运行手册 (Runbook)

> 最后更新：2026-07-04 | 适用范围：therapy-agent 生产环境
> 不设 7x24 值班。本手册供开发者在收到告警或服务异常时参考。

---

## 系统架构速览

```
用户 → Nginx (:443, TLS) → therapy-agent (:8000, gunicorn + uvicorn workers)
                               ├── Redis (:6379) — 会话缓存 / 速率限制
                               └── MySQL (:3306) — 持久化存储
```

**服务清单：**

| 服务 | 容器 | 端口 | 健康检查 |
|------|------|------|----------|
| therapy-agent | Docker (gunicorn) | 8000 (internal) | `GET /ping` |
| nginx | Docker (nginx:alpine) | 443, 80 | depends_on therapy-agent healthy |
| redis | Docker (redis:7-alpine) | 6379 (internal) | `redis-cli ping` |
| MySQL | 阿里云 RDS | 3306 | 云控制台监控 |

---

## 常用操作

### 重启服务

```bash
# 进入项目目录
cd /path/to/mental-intervene-master

# 重启所有容器
docker compose -f docker/docker-compose.yml --env-file .env restart

# 仅重启应用（不影响 nginx/redis）
docker compose -f docker/docker-compose.yml --env-file .env restart therapy-agent

# 完全重建（代码或依赖变更后）
docker compose -f docker/docker-compose.yml --env-file .env up -d --build
```

### 查看服务状态

```bash
# 容器状态
docker compose -f docker/docker-compose.yml ps

# 应用日志（最近 100 行）
docker compose -f docker/docker-compose.yml logs --tail=100 therapy-agent

# 实时跟踪日志
docker compose -f docker/docker-compose.yml logs -f therapy-agent

# Nginx 日志
docker compose -f docker/docker-compose.yml logs --tail=50 nginx
```

### 查看应用指标

```bash
# 健康检查
curl -k https://localhost/ping

# 速率限制状态（需要 Redis 连接）
docker compose -f docker/docker-compose.yml exec redis redis-cli DBSIZE

# 活跃会话数
docker compose -f docker/docker-compose.yml exec redis redis-cli KEYS "psy:session:*" | wc -l
```

---

## 故障排查

### 症状：服务 502/503 不可用

**原因：** therapy-agent 崩溃或未就绪

**排查：**
```bash
# 1. 检查容器是否运行
docker compose -f docker/docker-compose.yml ps therapy-agent

# 2. 查看应用日志中的错误
docker compose -f docker/docker-compose.yml logs --tail=200 therapy-agent | grep -E "ERROR|CRITICAL"

# 3. 确认数据库连通性
docker compose -f docker/docker-compose.yml exec therapy-agent \
  python -c "from sqlalchemy import create_engine; from config.settings import settings; e=create_engine(settings.DATABASE_URL); e.connect(); print('DB OK')"

# 4. 检查磁盘空间
df -h
```

**常见修复：**
- OOM：检查 `docker stats`，增大容器内存限制
- DB 连接池耗尽：重启 therapy-agent
- LLM API 超时：检查 `OPENAI_API_BASE` 所指向的服务

### 症状：响应慢 / 超时

**原因：** LLM 推理压力大、数据库慢查询、Redis 内存不足

**排查：**
```bash
# 查看 gunicorn worker 数
docker compose -f docker/docker-compose.yml exec therapy-agent ps aux | grep gunicorn

# 查看 Redis 内存
docker compose -f docker/docker-compose.yml exec redis redis-cli INFO memory | grep used_memory_human

# 查看 MySQL 慢查询（需要 RDS 控制台或直接连接）
mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE \
  -e "SHOW PROCESSLIST;"
```

### 症状：数据库连接失败

**原因：** RDS 白名单未放行、密码过期、连接数用尽

**排查：**
```bash
# 1. 测试连通性
mysql -h $MYSQL_HOST -P $MYSQL_PORT -u $MYSQL_USER -p$MYSQL_PASSWORD -e "SELECT 1"

# 2. 检查 .env 中的数据库配置
grep MYSQL_ .env

# 3. 确认 RDS 最大连接数（需在阿里云控制台查看）
```

### 症状：告警 webhook 未收到

**检查：**
```bash
# 确认告警配置
grep ALERT_WEBHOOK .env

# 手动触发测试（进入容器）
docker compose -f docker/docker-compose.yml exec therapy-agent \
  python -c "
import logging
from modules.alert_webhook import AlertWebhookService
from config.settings import settings
svc = AlertWebhookService(
    dingtalk_url=settings.ALERT_WEBHOOK_DINGTALK_URL,
    feishu_url=settings.ALERT_WEBHOOK_FEISHU_URL,
    enabled=True,
)
print(svc.push('测试告警', 'Runbook 手动测试'))
"
```

---

## 依赖服务故障处理

| 依赖 | 不可用时的影响 | 恢复后行为 |
|------|---------------|-----------|
| **LLM API** (qwen/OpenAI) | 聊天回复失败，返回 500 | 自动恢复，无需重启 |
| **MySQL** | 会话/消息无法持久化，部分接口降级 | 自动重连，需确认连接池 |
| **Redis** | 速率限制失效、会话缓存不可用，内存回退 | 自动恢复 |

- LLM API 不可用：therapy-agent 会返回 500 错误给用户，需等待 API 恢复。
- MySQL 不可用：关键操作会失败。恢复后需确认 Alembic 迁移已执行。
- Redis 不可用：`ratelimit` 失效，会话走内存回退（重启丢失），缓存不命中性能下降。

---

## 紧急停服

```bash
# 停止所有服务
docker compose -f docker/docker-compose.yml --env-file .env down

# 需要保留数据库数据时，不要加 -v（不加 -v 则 volumes 保留）
```

**恢复：**
```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
docker compose -f docker/docker-compose.yml ps  # 确认所有容器 healthy
curl -k https://localhost/ping                    # 最终确认
```

---

## 告警参考

| 级别 | 含义 | 示例 |
|------|------|------|
| WARNING | 需关注但不紧急 | 关键词引擎回退、webhook 发送失败 |
| ERROR | 需要响应 | 数据库写入失败、API 调用异常 |
| CRITICAL | 紧急，已推送 webhook | Emergency push 触发、MySQL 连接完全断开 |

CRITICAL 事件会自动推送到配置的钉钉/飞书 webhook（需要 `ALERT_WEBHOOK_ENABLED=true`）。

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `.env` | 所有环境变量（密钥、URL、开关） |
| `docker/docker-compose.yml` | 服务编排 |
| `docker/Dockerfile` | 应用镜像构建 |
| `nginx/nginx.conf` | HTTPS 反向代理配置 |
| `nginx/ssl/dev.crt` / `dev.key` | TLS 证书（生产需更换） |
| `gunicorn.conf.py` | Worker 数量和超时配置 |
| `migrations/` | Alembic 数据库迁移脚本 |
