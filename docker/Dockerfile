# =============================================================================
# 本地开发 Dockerfile — 单阶段构建（Windows 友好，避免跨层拷贝卡死）
# =============================================================================
# 使用示例:
#   docker build -t therapy-agent:latest -f docker/Dockerfile .
#   docker run -p 8000:8000 --env-file .env therapy-agent:latest
# =============================================================================

FROM python:3.11-slim-bookworm

# 系统依赖（torch 需要 libgomp1，opencv 需要 GL 库，音频需要 ffmpeg）
# 先换阿里云 Debian 镜像源加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码（.dockerignore 已排除 .git / tests / docs / models 等）
COPY . .

# 复制入口脚本
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/ping || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["gunicorn", "-c", "gunicorn.conf.py", "api.main:app"]
