#!/bin/bash
# REFRAG 서버 실행 스크립트

cd /home/keti_spark1/j309

# 가상환경 활성화 (있는 경우)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# REFRAG 서버 실행
echo "Starting REFRAG server on port 8011..."
uvicorn app.refrag_server:app --host 0.0.0.0 --port 8011

