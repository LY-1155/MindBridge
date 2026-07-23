"""Nginx 测试用的最小 HTTP 回显服务器。

暴露 /ping, /echo, /ws 三个端点用于验证 nginx 代理行为。
"""

from __future__ import annotations

import json
import os
import uuid

from fastapi import FastAPI, Request, WebSocket

app = FastAPI()


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/echo")
async def echo(request: Request):
    """回显请求头，用于验证 X-Forwarded-* 等代理头。"""
    return {
        "headers": dict(request.headers),
        "client_host": request.client.host if request.client else None,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"msg": "connected"})
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_json({"echo": data})
    except Exception:
        pass
