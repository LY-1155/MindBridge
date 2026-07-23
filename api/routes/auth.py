"""
Auth API 端点
============

POST /api/v1/auth/register   — 注册
POST /api/v1/auth/login      — 登录
POST /api/v1/auth/refresh    — 刷新 access_token
GET  /api/v1/auth/me         — 当前用户信息
POST /api/v1/auth/logout     — 注销（占位）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from modules.auth_service import AuthService
from modules.token_service import TokenService
from modules.auth_deps import get_current_user_id

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ---- Request / Response Models ----

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    user_id: str


class UserInfoResponse(BaseModel):
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    status: Optional[str] = None


# ---- Endpoints ----

@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest) -> Dict[str, Any]:
    """注册并返回 token。"""
    try:
        result = AuthService.register(
            username=req.username,
            password=req.password,
            display_name=req.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    user_id = result["user_id"]
    return {
        "access_token": TokenService.issue_access_token(user_id),
        "refresh_token": TokenService.issue_refresh_token(user_id),
        "user_id": user_id,
    }


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> Dict[str, Any]:
    """登录返回 token。"""
    try:
        result = AuthService.login(username=req.username, password=req.password)
    except ValueError as exc:
        msg = str(exc)
        if "密码错误" in msg:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)

    user_id = result["user_id"]
    return {
        "access_token": TokenService.issue_access_token(user_id),
        "refresh_token": TokenService.issue_refresh_token(user_id),
        "user_id": user_id,
    }


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest) -> Dict[str, Any]:
    """用 refresh_token 换新的 access_token。"""
    try:
        payload = TokenService.verify_token(req.refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="仅支持 refresh_token",
        )

    user_id = payload["user_id"]

    # 检查用户状态：已注销用户不能刷新 token
    from modules.user_service import UserService
    user = UserService.get_user(user_id)
    if user is None or user.get("status") == "deleted":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号已注销或不存在",
        )

    return {
        "access_token": TokenService.issue_access_token(user_id),
        "refresh_token": TokenService.issue_refresh_token(user_id),
        "user_id": user_id,
    }


@router.get("/me", response_model=UserInfoResponse)
def get_me(user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
    """当前用户信息。"""
    from modules.user_service import UserService
    user = UserService.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {
        "user_id": user["user_id"],
        "display_name": user.get("display_name"),
        "status": user.get("status"),
        "consent_at": user.get("consent_at"),
    }


# ── 知情同意 ──────────────────────────────────────────────

class ConsentStatusResponse(BaseModel):
    consented: bool
    consent_at: Optional[str] = None
    consent_version: Optional[str] = None


@router.get("/consent-status", response_model=ConsentStatusResponse)
def get_consent_status(user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
    """查询当前用户的知情同意状态。"""
    from modules.user_service import UserService
    user = UserService.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {
        "consented": user.get("consent_at") is not None,
        "consent_at": user.get("consent_at"),
        "consent_version": user.get("consent_version"),
    }


class ConsentRequest(BaseModel):
    version: str = Field(default="1.0", description="同意的协议版本号")


@router.post("/consent", status_code=status.HTTP_200_OK)
def record_consent(
    body: ConsentRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, str]:
    """记录用户知情同意。"""
    from modules.user_service import UserService
    ok = UserService.record_consent(user_id, version=body.version)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {"status": "ok", "version": body.version}


# ── 账号注销与恢复 ──────────────────────────────────────

class DeleteAccountRequest(BaseModel):
    password: str = Field(..., description="当前密码，用于确认身份")


@router.delete("/account", status_code=status.HTTP_200_OK)
def delete_account(
    body: DeleteAccountRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, str]:
    """注销账号：soft-delete + 所有 token 立即失效。

    - 30 天内可通过 /account/restore 恢复
    - 超过 30 天后数据将被物理删除
    """
    from modules.auth_service import AuthService
    from modules.user_service import UserService

    # 验证密码
    from schemas.database import db_manager
    from schemas.database_v2 import Credential
    with db_manager.get_session_direct() as s:
        cred = s.query(Credential).filter(
            Credential.user_id == user_id,
            Credential.type == "password",
        ).first()
        if not cred or not cred.secret:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="账号未设置密码")
        if not AuthService.verify_password(body.password, cred.secret):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码错误")

    ok = UserService.soft_delete(user_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {"status": "deleted", "message": "账号已注销，30 天内可联系恢复"}


@router.post("/account/restore", status_code=status.HTTP_200_OK)
def restore_account(user_id: str = Depends(get_current_user_id)) -> Dict[str, str]:
    """恢复已注销账号（仅限 30 天后悔期内）。"""
    from modules.user_service import UserService

    in_grace, error = UserService.is_within_grace_period(user_id)
    if not in_grace:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error or "无法恢复")

    ok = UserService.restore(user_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="恢复失败")
    return {"status": "active", "message": "账号已恢复"}
