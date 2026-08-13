"""
Application settings loaded from environment variables and `.env`.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_API_BASE: str = "http://localhost:8000/v1"
    MODEL_NAME: str = "qwen2.5-7b-instruct"
    SCORING_MODEL_NAME: str = "qwen-turbo"  # 量表计分专用轻量模型
    REWRITER_MODEL_NAME: str = "qwen-max"   # 查询改写模型（术语映射，建议与主模型同级）
    # ── Embedding ─────────────────────────────────────────────
    EMBEDDING_BACKEND: str = "api"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    EMBEDDING_API_BASE: str = "http://localhost:11434/v1"
    EMBEDDING_API_KEY: str = "ollama"
    EMBEDDING_DIMENSIONS: int = 0  # 0=模型默认(1024); 2048=高精度
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    # ── Chroma HTTP (Docker server) ───────────────────────────
    CHROMA_HTTP_HOST: str = "localhost"
    CHROMA_HTTP_PORT: int = 8001
    # ── Reranker ─────────────────────────────────────────────
    RERANK_MODEL_NAME: str = "qwen3-rerank"  # 百炼重排序模型
    # ── RRF 融合 ─────────────────────────────────────────────
    RRF_DENSE_WEIGHT: float = 1.0  # Chroma 稠密路权重; <1.0 降低噪声影响
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Database
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3307
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "psy_agent"
    USE_DATABASE: bool = True

    # Cache / app
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_TTL: int = 3600  # 会话 Redis 缓存 TTL（秒），每次访问自动刷新
    DEBUG: bool = True
    MAX_HISTORY_TURNS: int = 10

    # 字段级 AES-256-GCM 加密密钥（32 字节，生成：openssl rand -base64 32）
    ENCRYPTION_KEY: str = "change-me-to-a-32-byte-random-key"

    # 四模块并行开发：为 true 时使用 Mock，false 时使用 Stub（可替换为真实实现）
    MOCK_SAFETY: bool = True
    MOCK_EMOTION: bool = True
    MOCK_ROUTER: bool = True
    MOCK_INTERVENTION: bool = True

    # ── 医生模式（周医生 persona + 家庭系统评估 + SCID 后台追踪）────
    DOCTOR_MODE: bool = False       # 总开关；true=启用周医生风格对话
    DOCTOR_PERSONA: str = "zhou"    # persona 选择（zhou / future personas）

    # ── 语义安全评估器（危机判定改造，ADR-0013）────────────────
    SAFETY_JUDGE_MODEL: str = "qwen-turbo"             # 评估器专用模型（flash 级，可在 .env 覆盖）
    SAFETY_JUDGE_ANCHOR_RISK_THRESHOLD: float = 0.4    # 情绪风险锚点低阈值（故意低于 crisis 阈值 0.7，宁可多调不可漏掉）
    SAFETY_JUDGE_EVERY_TURN: bool = False              # 锚点失效后备开关：每轮调用评估器
    SAFETY_JUDGE_TIMEOUT_SECONDS: int = 5              # 评估器 LLM 调用超时（LLMConfig.timeout）
    SAFETY_JUDGE_HISTORY_TURNS: int = 6                # 评估器读最近 N 轮对话
    SAFETY_PROBE_MAX_COUNT: int = 3                    # probe 累积升级阈值（多次探针复现 → crisis）

    # ASR（听写）：sensevoice 偏中文口语场景；无 funasr 或失败时自动回退 whisper
    ASR_BACKEND: str = "sensevoice"
    SENSEVOICE_ASR_MODEL: str = "iic/SenseVoiceSmall"
    SENSEVOICE_DEVICE: str = "cuda:0"

    # Faster-Whisper（ASR_BACKEND=whisper 或作为 sensevoice 的回退）
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_MODEL_PATH: str = (
        "models/models--Systran--faster-whisper-base/snapshots/"
        "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
    )
    WHISPER_DEVICE: str = "auto"

    # Text emotion classification engine
    EMOTION_ENGINE: str = "onnx"       # "keyword" | "onnx"
    EMOTION_ONNX_MODEL_PATH: str = "models/emotion_classifier/model.onnx"
    EMOTION_ONNX_TOKENIZER_PATH: str = ""  # 空 = 同 model_path 目录

    # Multimodal emotion stack
    AUDIO_EMOTION_BACKEND: str = "sensevoice"
    VISUAL_EMOTION_BACKEND: str = "emotiefflib"
    ENABLE_MULTIMODAL_EMOTION_FUSION: bool = True

    # 安全过滤 — 多模态
    SAFETY_MODEL_PATH: str = ""
    SAFETY_NSFW_MODEL_PATH: str = ""  # SigLIP2 NSFW 专用检测模型
    SAFETY_DEVICE: str = "cuda"

    # Tavily search API（知识库 fallback）
    TAVILY_API_KEY: str = ""

    # 紧急推送
    EMERGENCY_PUSH_ENABLED: bool = False  # 是否启用真实救助 API（false=dry-run 仅日志）
    EMERGENCY_RESCUE_API_URL: str = ""     # 救助 API 端点 URL
    EMERGENCY_RESCUE_API_KEY: str = ""     # 救助 API 密钥
    EMERGENCY_PUSH_COOLDOWN_SECONDS: int = 300  # 同一 session 冷却期

    # CRITICAL 告警 webhook（钉钉/飞书机器人）
    ALERT_WEBHOOK_ENABLED: bool = False
    ALERT_WEBHOOK_DINGTALK_URL: str = ""
    ALERT_WEBHOOK_FEISHU_URL: str = ""

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中未定义为类字段的变量


settings = Settings()
