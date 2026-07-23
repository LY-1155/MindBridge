"""
FastAPI应用入口模块
==================

这是整个心理咨询AI应用的入口文件。
FastAPI是一个现代、高性能的Python Web框架，特别适合构建API。

主要功能：
1. 创建FastAPI应用实例
2. 配置CORS（跨域资源共享）
3. 注册路由
4. 配置中间件
5. 定义启动和关闭事件

运行方式：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

访问API文档：
    http://localhost:8000/docs （Swagger UI）
    http://localhost:8000/redoc （ReDoc）
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import asyncio
import logging
import sys
import os
import uuid

# 将项目根目录添加到Python路径
# 这样可以确保所有模块都能正确导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import auth, chat, multimodal, parallel_modules, pipeline, user
from config.settings import settings
from modules.rate_limit import limiter


# ==================== 日志配置 ====================
# Gap #20：结构化日志 — 统一 request_id/user_id/session_id 上下文
from config.logging_config import (
    configure_structured_logging,
    set_request_context,
    get_logger,
    _request_id_ctx,
)
configure_structured_logging(debug=settings.DEBUG)
logger = get_logger(__name__)
if settings.DEBUG:
    for _log_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(_log_name).setLevel(logging.DEBUG)

# ==================== 应用生命周期管理 ====================
# 使用lifespan上下文管理器处理启动和关闭事件
# 这是FastAPI推荐的方式，比on_event装饰器更现代

async def _periodic_account_cleanup():
    """后台定时任务：每 6 小时检查并物理删除超过 30 天后悔期的已注销账号。"""
    while True:
        try:
            await asyncio.sleep(6 * 3600)  # 6 小时
            from modules.user_service import UserService
            purged = UserService.purge_expired_accounts()
            if purged > 0:
                logger.info(f"账号清理：已物理删除 {purged} 个过期账号")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("账号清理任务异常")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    
    这个函数在应用启动前和关闭后执行。
    用于初始化资源（如数据库连接）和清理资源。
    
    工作流程：
    1. 应用启动时：执行yield之前的代码
    2. 应用运行中：处理请求
    3. 应用关闭时：执行yield之后的代码
    """
    # ===== 启动时执行 =====
    logger.info("心理咨询AI服务启动中...")
    logger.info(f"模型配置: {settings.MODEL_NAME}")
    logger.info(f"API基础URL: {settings.OPENAI_API_BASE}")

    # 启动后台定时任务：每 6 小时清理超过 30 天后悔期的已注销账号
    cleanup_task = asyncio.create_task(_periodic_account_cleanup())

    yield  # 应用运行中，处理请求

    # ===== 关闭时执行 =====
    logger.info("心理咨询AI服务关闭中...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# ==================== 创建FastAPI应用实例 ====================
# 这是核心：创建FastAPI应用对象

security_scheme = HTTPBearer()

app = FastAPI(
    title="心理咨询AI API",           # API标题
    description="""
    ## 心理咨询AI助手API

    这是一个基于大语言模型的心理咨询AI助手服务。

    ### 主要功能
    - 智能心理咨询对话
    - 情绪分析与追踪
    - 治疗阶段管理
    - 危机干预机制
    - 思维链分析

    ### 并行开发契约流水线
    - `POST /api/v1/modules/*`：安全 / 情感 / 路由 / 干预 独立接口
    - `POST /api/v1/pipeline/run`：四阶段串联；JSON 见 `docs/parallel_module_io_samples.md`

    ### 治疗方法
    系统整合运用以下治疗方法：
    - 人本主义疗法
    - 认知行为疗法(CBT)
    - 叙事疗法
    - 正念疗法
    - 焦点解决短期治疗(SFBT)
    """,                               # API描述
    version="1.0.0",                   # API版本
    lifespan=lifespan,                 # 生命周期管理器
    docs_url="/docs",                  # Swagger UI文档路径
    redoc_url="/redoc",                # ReDoc文档路径
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
    },
)


class _RequestIdMiddleware(BaseHTTPMiddleware):
    """注入 request_id 到日志上下文 + 响应头。"""

    async def dispatch(self, request: Request, call_next):
        # 生成或复用 request_id
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_context(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class _DebugRequestLoggingMiddleware(BaseHTTPMiddleware):
    """DEBUG 模式下记录每个请求的方法、路径与响应状态。"""

    async def dispatch(self, request: Request, call_next):
        if not settings.DEBUG:
            return await call_next(request)
        req_id = _request_id_ctx.get("-")
        logger.debug("→ %s %s", request.method, request.url.path,
                      extra={"request_id": req_id})
        response = await call_next(request)
        logger.debug("← %s %s status=%s", request.method, request.url.path,
                      response.status_code, extra={"request_id": req_id})
        return response


# ==================== CORS配置 ====================
# CORS（Cross-Origin Resource Sharing）跨域资源共享
# 允许前端应用从不同的域名访问API

app.add_middleware(
    CORSMiddleware,
    # allow_origins: 允许访问的源列表
    # ["*"] 表示允许所有源（开发环境）
    # 生产环境应该指定具体的前端域名
    allow_origins=["*"],
    
    # allow_credentials: 是否允许携带凭证（如Cookie）
    allow_credentials=True,
    
    # allow_methods: 允许的HTTP方法
    allow_methods=["*"],  # 允许所有方法（GET, POST, PUT, DELETE等）
    
    # allow_headers: 允许的请求头
    allow_headers=["*"],  # 允许所有请求头
)

# Request ID 注入（必须在速率限制之前，确保限制日志也有 request_id）
app.add_middleware(_RequestIdMiddleware)

# Rate limiting: per-user 60 req/min (default), crisis 接口不限
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

if settings.DEBUG:
    app.add_middleware(_DebugRequestLoggingMiddleware)


# ==================== 注册路由 ====================
# 将定义好的路由模块注册到应用
# prefix参数为所有路由添加前缀

app.include_router(chat.router)
app.include_router(multimodal.router)
app.include_router(pipeline.router)
app.include_router(auth.router)
app.include_router(parallel_modules.router)
app.include_router(user.router)


STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

if os.path.exists(os.path.join(STATIC_DIR, "css")):
    _static_app = StaticFiles(directory=STATIC_DIR)

    if settings.DEBUG:
        from starlette.types import Scope, Receive, Send
        async def _no_cache_static(scope: Scope, receive: Receive, send: Send):
            async def _send(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"cache-control"] = b"no-cache, no-store, must-revalidate"
                    message["headers"] = list(headers.items())
                await send(message)
            await _static_app(scope, receive, _send)
        app.mount("/static", _no_cache_static, name="static")
    else:
        app.mount("/static", _static_app, name="static")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """
    聊天页面

    返回Web聊天界面。
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        response = HTMLResponse(content=content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    return HTMLResponse(content="<h1>聊天页面未找到</h1>", status_code=404)


# ==================== 异常处理 ====================
# 速率限制专用处理器 — 必须注册在全局 Exception handler 之前，否则被吞掉
@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": "请求过于频繁",
            "message": "请稍后重试",
            "retry_after_seconds": exc.retry_after if hasattr(exc, "retry_after") else 60,
        },
    )


# 全局异常处理器
# 当发生未捕获的异常时，返回友好的错误信息

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    捕获所有未处理的异常，返回统一的错误响应格式。
    这样可以避免向用户暴露敏感的错误信息。
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 统一格式的错误响应
    """
    # 记录错误日志
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    
    # 返回友好的错误信息
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "message": "请稍后重试，如果问题持续存在请联系管理员",
            "detail": str(exc) if settings.DEBUG else None  # 只在调试模式显示详情
        }
    )


# ==================== 根路由 ====================
# 访问根路径时返回欢迎信息

@app.get("/")
async def root():
    """
    根路由
    
    访问API根路径时返回欢迎信息和基本说明。
    """
    return {
        "message": "欢迎使用心理咨询AI API",
        "docs": "/docs",
        "version": "1.0.0"
    }


# ==================== 健康检查路由 ====================
# 用于检查服务是否正常运行

@app.get("/ping")
async def ping():
    """
    简单的健康检查端点
    
    用于负载均衡器和监控系统检查服务状态。
    与/api/v1/health类似，但更简单。
    """
    return {"status": "ok"}


# ==================== 应用启动说明 ====================
# 当直接运行此文件时启动服务

if __name__ == "__main__":
    import uvicorn
    
    # uvicorn是ASGI服务器，用于运行FastAPI应用
    # reload=True 启用热重载，代码修改后自动重启
    # host="0.0.0.0" 允许外部访问
    # port=8000 监听端口
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
