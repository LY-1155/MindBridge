"""
Nginx 反向代理集成测试（gap #8）

TDD 循环：
  1. nginx 配置语法校验
  2. GET /ping 通过代理返回 200
  3. HTTP(80) → HTTPS(443) 重定向
  4. HTTPS 请求正确转发到 uvicorn
  5. X-Forwarded-* 代理头正确设置
  6. /static/ 直由 nginx 返回，不穿透 uvicorn
  7. WebSocket 升级请求正常透传

所有测试依赖 Docker（nginx:alpine 镜像，无需本机安装 nginx）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
import requests
import urllib3

# 自签名证书 — 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGINX_DIR = os.path.join(ROOT, "nginx")
NGINX_CONF = os.path.join(NGINX_DIR, "nginx.conf")
SSL_DIR = os.path.join(NGINX_DIR, "ssl")
CERT_PATH = os.path.join(SSL_DIR, "dev.crt")
KEY_PATH = os.path.join(SSL_DIR, "dev.key")
DOCKER_DIR = os.path.join(ROOT, "docker")
COMPOSE_FILE = os.path.join(DOCKER_DIR, "docker-compose.test.yml")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_docker() -> bool:
    """Docker CLI + daemon 都可用才返回 True。"""
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, check=True, timeout=10
        )
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _has_docker(), reason="Docker 不可用"
)


def _generate_dev_cert():
    """用 Python cryptography 生成自签名证书（跨平台，不依赖 openssl）。"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    os.makedirs(SSL_DIR, exist_ok=True)
    if os.path.isfile(CERT_PATH) and os.path.isfile(KEY_PATH):
        return  # 已有证书，跳过

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    import datetime
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def _compose(args: list[str], **kwargs):
    """docker compose 快捷调用（--project-name 隔离测试环境）。"""
    return subprocess.run(
        ["docker", "compose", "-p", "nginx-test", "-f", COMPOSE_FILE, *args],
        cwd=ROOT, check=True, **kwargs,
    )


# ---------------------------------------------------------------------------
# module-level fixture：起停 docker-compose
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def proxy():
    """启动 docker compose（nginx + therapy-agent），暴露 HTTPS 地址。

    模块级 scope：同一模块内所有集成测试共用同一组容器。
    """
    _generate_dev_cert()

    # 确保干净起点
    _compose(["down", "--remove-orphans", "-t", "5"],
             capture_output=True, text=True, timeout=30)

    # 仅首次构建（缓存命中后秒过）
    _compose(["build"], capture_output=True, text=True, timeout=300)

    # 后台启动
    _compose(["up", "-d", "--wait"],
             capture_output=True, text=True, timeout=60)

    # 留一点时间让 nginx 就绪
    time.sleep(2)

    yield "https://localhost"

    # 清理
    _compose(["down", "--remove-orphans", "-t", "5"],
             capture_output=True, text=True, timeout=30)


# ---------------------------------------------------------------------------
# 1. 配置语法
# ---------------------------------------------------------------------------

class TestNginxConfigSyntax:
    """nginx -t 语法校验"""

    @requires_docker
    def test_nginx_config_syntax_valid(self):
        """nginx.conf 通过 nginx -t 语法检查"""
        assert os.path.isfile(NGINX_CONF), (
            f"nginx.conf 不存在: {NGINX_CONF}"
        )

        # 生成自签名证书（nginx -t 会校验 ssl_certificate 路径存在）
        _generate_dev_cert()

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{NGINX_CONF}:/etc/nginx/nginx.conf:ro",
                "-v", f"{SSL_DIR}:/etc/nginx/ssl:ro",
                "nginx:alpine", "nginx", "-t",
            ],
            capture_output=True, text=True, timeout=30,
        )

        assert result.returncode == 0, (
            f"nginx -t 失败:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 2. ping 透传
# ---------------------------------------------------------------------------

@requires_docker
class TestPingThroughProxy:
    """GET /ping 通过代理返回 200"""

    def test_ping_returns_200_through_proxy(self, proxy):
        resp = requests.get(f"{proxy}/ping", verify=False, timeout=10)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 3. HTTP → HTTPS 重定向
# ---------------------------------------------------------------------------

@requires_docker
class TestHttpRedirect:
    """HTTP(80) → HTTPS(443) 301 永久重定向"""

    def test_http_redirects_to_https(self, proxy):
        resp = requests.get("http://localhost/ping", allow_redirects=False, timeout=10)
        assert resp.status_code == 301, (
            f"期望 301 重定向，实际 {resp.status_code}"
        )
        assert resp.headers["Location"].startswith("https://"), (
            f"重定向目标不是 HTTPS: {resp.headers['Location']}"
        )


# ---------------------------------------------------------------------------
# 4. HTTPS 请求正确转发
# ---------------------------------------------------------------------------

@requires_docker
class TestHttpsProxying:
    """HTTPS 请求内容正确转发到后端，响应体完整返回"""

    def test_https_proxies_to_app(self, proxy):
        resp = requests.get(f"{proxy}/echo", verify=False, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "headers" in data
        assert data["client_host"] is not None


# ---------------------------------------------------------------------------
# 5. 代理头
# ---------------------------------------------------------------------------

@requires_docker
class TestProxyHeaders:
    """X-Forwarded-For / X-Forwarded-Proto / X-Real-IP 正确设置"""

    def test_proxy_headers_set(self, proxy):
        resp = requests.get(f"{proxy}/echo", verify=False, timeout=10)
        assert resp.status_code == 200
        headers = resp.json()["headers"]

        assert "x-forwarded-for" in headers, (
            f"缺少 X-Forwarded-For，收到 headers: {list(headers.keys())}"
        )
        assert "x-forwarded-proto" in headers, (
            f"缺少 X-Forwarded-Proto，收到 headers: {list(headers.keys())}"
        )
        assert "x-real-ip" in headers, (
            f"缺少 X-Real-IP，收到 headers: {list(headers.keys())}"
        )


# ---------------------------------------------------------------------------
# 6. 静态文件
# ---------------------------------------------------------------------------

@requires_docker
class TestStaticFiles:
    """/static/ 直由 nginx 返回，不穿透 uvicorn"""

    def test_static_files_served_by_nginx(self, proxy):
        resp = requests.get(f"{proxy}/static/test.txt", verify=False, timeout=10)
        assert resp.status_code == 200, (
            f"期望 200，实际 {resp.status_code}"
        )
        assert "nginx-test" in resp.text


# ---------------------------------------------------------------------------
# 7. WebSocket
# ---------------------------------------------------------------------------

@requires_docker
class TestWebSocketUpgrade:
    """WebSocket 升级请求正常透传"""

    def test_websocket_upgrade_through_proxy(self, proxy):
        import ssl as _ssl
        import websocket

        ws_url = proxy.replace("https://", "wss://") + "/ws"
        ws = websocket.create_connection(
            ws_url, timeout=10,
            sslopt={"cert_reqs": _ssl.CERT_NONE},
        )

        result = ws.recv()
        import json
        assert json.loads(result) == {"msg": "connected"}

        ws.send("hello")
        echo = ws.recv()
        assert json.loads(echo) == {"echo": "hello"}

        ws.close()
