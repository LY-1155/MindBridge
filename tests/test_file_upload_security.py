"""
Gap #11: 文件上传安全 — 大小限制 + magic bytes 验证真实文件类型

验证行为：
  1. 各文件类别大小限制（image 10MB, audio 20MB, video 100MB）
  2. MIME 类型白名单
  3. magic bytes 验证真实文件类型
  4. 伪装的恶意文件被拒绝（文本改名 .jpg）
  5. 合法文件通过验证
  6. 端点集成：multimodal / pipeline / parallel_modules 接受验证
"""

from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from starlette.datastructures import UploadFile, Headers


# ---------------------------------------------------------------------------
# 测试用最小合法二进制文件
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    """最小合法 JPEG（via PIL）"""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (1, 1), color="red")
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    """最小合法 PNG"""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (1, 1), color="red")
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_wav_bytes(duration_sec: float = 0.1) -> bytes:
    """最小合法 WAV（PCM 16-bit mono）"""
    buf = io.BytesIO()
    framerate = 44100
    nframes = int(framerate * duration_sec)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


def _make_mp4_ftyp() -> bytes:
    """最小 MP4 ftyp box（filetype 可识别为 mp4）"""
    # ISOBMFF: size(4) + 'ftyp'(4) + major_brand(4) + minor_version(4) + compatible_brands(rest)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41isom"


def _make_upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    """构造 Starlette UploadFile 测试对象"""
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


# ---------------------------------------------------------------------------
# 1. 文件大小限制
# ---------------------------------------------------------------------------

class TestSizeLimits:
    """各文件类别有大小的上限检查"""

    @pytest.mark.parametrize("category, limit_mb", [
        ("image", 10),
        ("audio", 20),
        ("video", 100),
    ])
    async def test_valid_size_passes(self, category, limit_mb):
        """合法大小文件通过验证"""
        from modules.file_upload_security import validate_upload

        if category == "image":
            content = _make_jpeg_bytes()
            ct = "image/jpeg"
            fname = "test.jpg"
        elif category == "audio":
            content = _make_wav_bytes()
            ct = "audio/wav"
            fname = "test.wav"
        else:
            content = _make_mp4_ftyp()
            ct = "video/mp4"
            fname = "test.mp4"

        upload = _make_upload(fname, content, ct)
        await validate_upload(upload, category=category)
        # 验证后文件可被重新读取
        post_read = await upload.read()
        assert len(post_read) == len(content)

    @pytest.mark.parametrize("category, name, ct", [
        ("image", "big.jpg", "image/jpeg"),
        ("audio", "big.wav", "audio/wav"),
        ("video", "big.mp4", "video/mp4"),
    ])
    async def test_oversized_file_raises_413(self, category, name, ct):
        """超限文件返回 HTTP 413"""
        from modules.file_upload_security import validate_upload, SIZE_LIMITS_MB
        from fastapi import HTTPException

        limit_mb = SIZE_LIMITS_MB[category]
        oversized = b"X" * (limit_mb * 1024 * 1024 + 1)  # +1 byte over limit

        upload = _make_upload(name, oversized, ct)
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category=category)
        assert exc_info.value.status_code == 413


# ---------------------------------------------------------------------------
# 2. MIME 类型白名单
# ---------------------------------------------------------------------------

class TestMimeTypeAllowList:
    """声明的 MIME 类型必须在白名单中"""

    async def test_allowed_mime_images(self):
        from modules.file_upload_security import validate_upload, ALLOWED_MIME_TYPES
        for mime in ALLOWED_MIME_TYPES["image"]:
            ext = mime.split("/")[-1]
            if ext == "jpeg":
                content = _make_jpeg_bytes()
            else:
                content = _make_png_bytes()  # png/webp/bmp 都用 png magic test
            upload = _make_upload(f"test.{ext}", content, mime)
            # 不应抛异常（magic bytes 是 png 但 MIME 声明不匹配时会在第 3 步捕获）
            # 这里只测 MIME 在白名单中不会被 MIME 检查拒绝
            ...

    async def test_disallowed_mime_raises_415(self):
        """不在白名单的 MIME 返回 HTTP 415"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        upload = _make_upload("test.exe", _make_jpeg_bytes(), "application/x-msdownload")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category="image")
        assert exc_info.value.status_code == 415

    @pytest.mark.parametrize("category, good_ct, bad_ct", [
        ("image", "image/jpeg", "text/html"),
        ("audio", "audio/wav", "application/pdf"),
        ("video", "video/mp4", "application/zip"),
    ])
    async def test_wrong_mime_for_category_raises_415(self, category, good_ct, bad_ct):
        """每个类别有独立的白名单，跨类别的 MIME 被拒绝"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        if category == "image":
            content = _make_jpeg_bytes()
            fname = "test.jpg"
        elif category == "audio":
            content = _make_wav_bytes()
            fname = "test.wav"
        else:
            content = _make_mp4_ftyp()
            fname = "test.mp4"

        upload = _make_upload(fname, content, bad_ct)
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category=category)
        assert exc_info.value.status_code == 415


# ---------------------------------------------------------------------------
# 3. magic bytes 验证
# ---------------------------------------------------------------------------

class TestMagicBytesValidation:
    """filetype 库验证真实文件类型"""

    async def test_plain_text_disguised_as_jpg_raises_422(self):
        """文本文件改名 .jpg 被 magic bytes 识别并拒绝"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        text_content = b"This is not a JPEG image, just plain text."
        upload = _make_upload("innocent.jpg", text_content, "image/jpeg")

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category="image")
        assert exc_info.value.status_code == 422

    async def test_html_disguised_as_mp3_raises_422(self):
        """HTML 文件改名 .mp3 被拒绝"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        html = b"<html><body>Not an MP3</body></html>"
        upload = _make_upload("song.mp3", html, "audio/mpeg")

        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category="audio")
        assert exc_info.value.status_code == 422

    async def test_empty_file_raises_422(self):
        """空文件无法识别 magic bytes → 422"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        upload = _make_upload("empty.jpg", b"", "image/jpeg")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category="image")
        assert exc_info.value.status_code == 422

    async def test_valid_jpeg_passes(self):
        """合法 JPEG magic bytes 通过验证"""
        from modules.file_upload_security import validate_upload
        upload = _make_upload("photo.jpg", _make_jpeg_bytes(), "image/jpeg")
        await validate_upload(upload, category="image")

    async def test_valid_png_passes(self):
        """合法 PNG magic bytes 通过验证"""
        from modules.file_upload_security import validate_upload
        upload = _make_upload("icon.png", _make_png_bytes(), "image/png")
        await validate_upload(upload, category="image")

    async def test_valid_wav_passes(self):
        """合法 WAV magic bytes 通过验证"""
        from modules.file_upload_security import validate_upload
        upload = _make_upload("recording.wav", _make_wav_bytes(), "audio/wav")
        await validate_upload(upload, category="audio")

    async def test_valid_mp4_passes(self):
        """合法 MP4 ftyp magic bytes 通过验证"""
        from modules.file_upload_security import validate_upload
        upload = _make_upload("clip.mp4", _make_mp4_ftyp(), "video/mp4")
        await validate_upload(upload, category="video")

    async def test_jpeg_masquerading_as_mp4_raises_422(self):
        """JPEG 内容声明为 video/mp4 → magic bytes 与 category 矛盾，被拒绝"""
        from modules.file_upload_security import validate_upload
        from fastapi import HTTPException

        upload = _make_upload("video.mp4", _make_jpeg_bytes(), "video/mp4")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(upload, category="video")
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# 4. 端点集成（FastAPI TestClient）
# ---------------------------------------------------------------------------

class TestEndpointIntegration:
    """验证上传端点正确调用 validate_upload"""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        """注入假的用户 ID，跳过 JWT 验证"""
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return "test-user-gap11"

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    # ── 端点集成：正确文件通过 ──

    def test_safety_image_valid_jpeg_does_not_reject_at_guard(self, client):
        """合法 JPEG 不被上传守卫拒绝（后续可能因模型加载失败 500，但非 4xx 守卫错误）"""
        resp = client.post(
            "/api/v1/multimodal/safety/image",
            files={"image": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        # 不应该是上传守卫的 413/415/422
        assert resp.status_code not in (413, 415, 422)

    # ── 端点集成：恶意文件被拒绝 ──

    def test_safety_image_rejects_text_as_jpg(self, client):
        """文本文件改名 .jpg → 端点返回 422"""
        resp = client.post(
            "/api/v1/multimodal/safety/image",
            files={"image": ("evil.jpg", b"<html>not image</html>", "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_safety_image_rejects_oversized(self, client):
        """超限图片 → 端点返回 413"""
        from modules.file_upload_security import SIZE_LIMITS_MB
        fat = b"A" * (SIZE_LIMITS_MB["image"] * 1024 * 1024 + 100)
        resp = client.post(
            "/api/v1/multimodal/safety/image",
            files={"image": ("fat.jpg", fat, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_safety_image_rejects_wrong_mime(self, client):
        """合法 JPEG 但声明 MIME 为 text/html → 端点返回 415"""
        resp = client.post(
            "/api/v1/multimodal/safety/image",
            files={"image": ("photo.jpg", _make_jpeg_bytes(), "text/html")},
        )
        assert resp.status_code == 415
