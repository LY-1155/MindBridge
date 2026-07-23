#!/bin/sh
# 开发环境自签名证书生成
# 生产环境：替换为阿里云 SSL 证书，将证书文件放入此目录并命名为
#   server.crt（完整证书链）和 server.key（私钥）

openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout dev.key \
    -out dev.crt \
    -subj "/CN=localhost"

echo "自签名证书已生成: dev.crt / dev.key"
echo "生产部署时替换为阿里云签发的 server.crt / server.key"
