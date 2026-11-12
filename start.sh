#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "[1/2] 安装依赖（如已装可跳过）..."
python3 -m pip install -r requirements.txt

echo "[2/2] 启动服务..."
python3 server.py &
SERVER_PID=$!
sleep 3
# 尝试自动打开浏览器
(if command -v open >/dev/null 2>&1; then open http://localhost:8501; \
 elif command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:8501; \
 else echo "请手动打开 http://localhost:8501"; fi)

echo "服务已启动 (pid=$SERVER_PID)。按 Ctrl+C 停止。"
wait $SERVER_PID
