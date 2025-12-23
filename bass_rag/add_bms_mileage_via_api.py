#!/usr/bin/env python3
"""Add BMS mileage calculation rules to RAG index via API (when server is running)."""
import requests
import sys
from pathlib import Path

def add_bms_mileage_rules_via_api():
    """Add BMS mileage calculation rules via /ingest API."""
    server_url = "http://localhost:8012"
    
    # Check if server is running
    try:
        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code != 200:
            print("❌ 서버가 응답하지 않습니다.")
            return
    except requests.exceptions.RequestException:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return
    
    # Read BMS mileage rules file
    rules_file = Path("data/bms_mileage_calculation_rules.txt")
    if not rules_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {rules_file}")
        return
    
    with open(rules_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 파일 읽기 완료: {len(content)} 문자")
    
    # Add metadata
    metadata = '[메타데이터: 타입: 규칙/필드정의 | 형식: 텍스트 | 요약: true | 원본_문서ID: data/bms_mileage_calculation_rules.txt]\n\n'
    text_with_metadata = metadata + content
    
    # Send to /ingest API
    print("📤 서버에 전송 중...")
    try:
        response = requests.post(
            f"{server_url}/ingest",
            json={"text": text_with_metadata},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ 완료! {data.get('added_chunks', 0)}개 청크가 추가되었습니다.")
    except requests.exceptions.RequestException as e:
        print(f"❌ API 호출 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"   상세: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   상태 코드: {e.response.status_code}")

if __name__ == "__main__":
    add_bms_mileage_rules_via_api()
