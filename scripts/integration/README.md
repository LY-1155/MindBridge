# 集成 / 冒烟脚本（非 pytest）

这些脚本从仓库根目录迁移至此，原本是根目录下的 `test_*.py`。它们多数是：

- **需要本地先启动 API**（`uvicorn`，默认 `http://localhost:8000`），或  
- **需要 MySQL / 模型文件 / GPU / 外网 API**，或  
- **控制台打印**，不适合作为 CI 默认 `pytest` 套件。

文件名已**去掉误导性的 `test_` 前缀**，避免被 pytest 当成单元测试收集（pytest 默认只扫描 `tests/`，但若误移入 `tests/` 仍容易混淆）。

自动化、可重复、无额外服务的测试请使用 **`tests/` + `pytest`**（见 `tests/README.md`）。

## 运行方式

在项目根目录执行：

```powershell
python scripts/integration/api_chat_smoke.py
```

或通过环境变量指定 API 地址（HTTP 脚本）：

```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
python scripts/integration/api_multimodal_http.py
```

## 脚本一览

| 脚本 | 说明 |
|------|------|
| `api_chat_smoke.py` | 对话、会话列表等 HTTP 冒烟（httpx） |
| `api_minimal_async.py` | 异步简化 API 检查 |
| `api_chat_session_db.py` | 对话 + 直连数据库校验 |
| `api_multimodal_http.py` | 多模态相关 HTTP（TTS、voices、chat） |
| `multimodal_local_smoke.py` | 本地加载 TTS/情绪/ASR 模块（可能写 `test_output.mp3`） |
| `asr_model_load.py` | 尝试加载 ASR 模型 |
| `mysql_session_storage.py` | MySQL 会话读写 |
| `llm_qwen_smoke.py` | 通义千问在线调用 |
| `tts_list_voices.py` | Edge-TTS 中文声音列表 |
| `edge_tts_debug_voices.py` | Edge-TTS 原始 voice 结构调试 |
| `verify_core_components.py` | 核心组件综合验证（Mock 链路等） |
| `interactive_therapy_cli.py` | 交互式命令行咨询会话 |
