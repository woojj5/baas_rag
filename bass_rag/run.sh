#!/bin/bash

echo "=== RAG Server 실행 스크립트 ==="
echo ""

# 1. 패키지 설치 확인
echo "[1/4] 패키지 확인 중..."
python3 -c "import sentence_transformers, faiss, fastapi, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[!] 패키지가 설치되지 않았습니다. 설치 중..."
    pip install -r requirements.txt
fi
echo "[+] 패키지 확인 완료"
echo ""

# 2. 데이터 확인
echo "[2/4] 데이터 확인 중..."
if [ ! -d "data" ] || [ -z "$(ls -A data/*.txt data/*.md 2>/dev/null)" ]; then
    echo "[!] data/ 디렉토리에 .txt 또는 .md 파일이 없습니다."
    echo "[!] 예시 파일을 생성합니다..."
    mkdir -p data
    echo "EV Battery는 전기차의 핵심 부품입니다. 리튬이온 배터리를 사용합니다." > data/battery.txt
    echo "BMS(Battery Management System)는 배터리를 관리하는 시스템입니다." > data/bms.txt
fi
echo "[+] 데이터 확인 완료"
echo ""

# 3. 인덱스 확인 및 빌드
echo "[3/4] 인덱스 확인 중..."
if [ ! -f "index/faiss.index" ]; then
    echo "[!] 인덱스가 없습니다. 빌드 중..."
    python3 build_index.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] 인덱스 빌드 실패"
        exit 1
    fi
else
    echo "[+] 인덱스가 이미 존재합니다."
fi
echo ""

# 4. 서버 실행
echo "[4/4] 서버 시작 중..."
echo "[+] 서버 주소: http://localhost:8001"
echo "[+] API 문서: http://localhost:8001/docs"
echo ""
python3 rag_server.py
