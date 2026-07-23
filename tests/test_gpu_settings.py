"""GPU 配置测试：Settings 和 device 参数。"""
from __future__ import annotations

from config.settings import Settings


class TestGpuSettings:
    def test_sensevoice_device_default_cuda(self):
        s = Settings()
        assert s.SENSEVOICE_DEVICE == "cuda:0"

    def test_sensevoice_device_from_kwargs(self):
        s = Settings(SENSEVOICE_DEVICE="cpu")
        assert s.SENSEVOICE_DEVICE == "cpu"
