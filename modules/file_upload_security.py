"""
文件上传安全守卫（gap #11）

三道防线：
  1. 文件大小限制（按类别）
  2. MIME 类型白名单
  3. magic bytes 验证真实文件类型（filetype 库，纯 Python，无系统依赖）

用法：
    from modules.file_upload_security import validate_upload

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        await validate_upload(file, category="image")
        data = await file.read()  # 守卫内部 reset 过指针，可再次读取
"""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

import filetype

# ── 每种类别的大小上限（MB）──────────────────────────────────
SIZE_LIMITS_MB = {
    "image": 10,
    "audio": 20,
    "video": 100,
}

# ── 每种类别允许的 MIME 类型 ─────────────────────────────────
ALLOWED_MIME_TYPES = {
    "image": [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/gif",
        "image/tiff",
    ],
    "audio": [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mp4",
        "audio/x-m4a",
        "audio/flac",
    ],
    "video": [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    ],
}

# ── filetype 库对每种媒体类别的识别扩展名 ──────────────────
EXPECTED_FILETYPE_KINDS = {
    "image": ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff"],
    "audio": ["mp3", "wav", "flac", "ogg", "m4a", "wma", "aac", "opus", "aiff", "au", "amr"],
    "video": ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v", "mpg", "mts", "3gp"],
}


async def validate_upload(upload: UploadFile, *, category: str) -> None:
    """对上传文件执行三道防线检查。

    category: "image" | "audio" | "video"
    通过后 file 指针回到开头，下游可继续读取。

    Raises:
        HTTPException(413): 超出大小限制
        HTTPException(415): MIME 类型不在白名单
        HTTPException(422): magic bytes 不匹配
    """

    if category not in SIZE_LIMITS_MB:
        raise HTTPException(status_code=500, detail=f"内部错误：未知的上传类别 {category}")

    # 读取全部字节（用于大小、类型检测）
    content = await upload.read()
    file_size = len(content)

    # ── 防线 1：大小限制 ──
    limit_mb = SIZE_LIMITS_MB[category]
    limit_bytes = limit_mb * 1024 * 1024
    if file_size > limit_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小 {file_size / 1024 / 1024:.1f}MB 超过上限 {limit_mb}MB",
        )

    # ── 防线 2：MIME 白名单 ──
    declared_mime = upload.content_type or ""
    if declared_mime not in ALLOWED_MIME_TYPES[category]:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的内容类型：{declared_mime}",
        )

    # ── 防线 3：magic bytes 验证 ──
    kind = filetype.guess(content)
    if kind is None:
        raise HTTPException(
            status_code=422,
            detail="无法识别文件类型，请确认上传的是有效的媒体文件",
        )

    if kind.extension not in EXPECTED_FILETYPE_KINDS[category]:
        raise HTTPException(
            status_code=422,
            detail=f"文件内容检测为 {kind.mime}，与所属类别不匹配",
        )

    # 通过全部检查 → 复位文件指针，下游可继续读取
    upload.file.seek(0)
