"""
并行开发运行时：按需获取四模块实现（配置驱动 Mock / Stub）。
进程内单例；修改环境变量中的 MOCK_* 后需重启服务或调用 reset。
"""

from __future__ import annotations

from typing import Optional

from config.settings import Settings, settings as default_settings
from modules.factory import PipelineServices, build_pipeline_services


def pipeline_cache_key(settings: Settings) -> str:
    """用于区分 Mock 开关组合的键。"""
    return "|".join(
        [
            str(settings.MOCK_SAFETY),
            str(settings.MOCK_EMOTION),
            str(settings.MOCK_ROUTER),
            str(settings.MOCK_INTERVENTION),
        ]
    )


_services_holder: Optional[PipelineServices] = None
_holder_key: Optional[str] = None


def get_pipeline_services(settings: Optional[Settings] = None) -> PipelineServices:
    """返回当前进程内复用的四模块实例（含 intervention，与 HTTP/pipeline 共用同一套实现）。"""
    global _services_holder, _holder_key
    cfg = settings if settings is not None else default_settings
    key = pipeline_cache_key(cfg)
    if _services_holder is None or _holder_key != key:
        _services_holder = build_pipeline_services(cfg)
        _holder_key = key
    return _services_holder


def reset_pipeline_services() -> None:
    """热重载/单测用：下次请求将按当前配置重新创建实现。"""
    global _services_holder, _holder_key
    _services_holder = None
    _holder_key = None
