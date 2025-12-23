#!/usr/bin/env python3
"""Add BMS mileage calculation rules to RAG index."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag_index import chunk_text, save_index
from app.config import Config
import numpy as np
import faiss

def check_server_running():
    """Check if the server is running on port 8012."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8012))
        sock.close()
        return result == 0
    except:
        return False

def add_bms_mileage_rules():
    """Add BMS mileage calculation rules to the existing index."""
    # Check if server is running
    if check_server_running():
        print("⚠️  경고: 서버가 실행 중입니다!")
        print("   서버가 실행 중일 때는 스크립트로 인덱스를 수정하면")
        print("   서버 종료 시 메모리의 인덱스가 디스크를 덮어쓸 수 있습니다.")
        print()
        print("   해결 방법:")
        print("   1. 서버를 종료한 후 스크립트를 실행하세요")
        print("   2. 또는 서버가 실행 중일 때는 /ingest API를 사용하세요")
        print()
        response = input("   그래도 계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("   취소되었습니다.")
            return
    
    # Load existing index
    index_dir = Path(Config.INDEX_DIR)
    index_file = index_dir / "faiss.index"
    passages_file = index_dir / "passages.json"
    
    if not index_file.exists() or not passages_file.exists():
        print("❌ 인덱스 파일이 없습니다. 먼저 인덱스를 생성하세요.")
        return
    
    # Load existing index and passages
    print("📖 기존 인덱스 로드 중...")
    faiss_index = faiss.read_index(str(index_file))
    
    import json
    with open(passages_file, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    
    print(f"✅ 기존 인덱스 로드 완료: {len(passages)}개 passages")
    
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
    
    # Create chunks
    print("🔨 청크 생성 중...")
    new_chunks = chunk_text(text_with_metadata, metadata='타입: 규칙/필드정의 | 형식: 텍스트')
    
    if not new_chunks:
        print("❌ 청크가 생성되지 않았습니다.")
        return
    
    print(f"✅ {len(new_chunks)}개 청크 생성 완료")
    
    # Load embedding model
    print("🤖 임베딩 모델 로드 중...")
    from sentence_transformers import SentenceTransformer
    model_name = getattr(Config, 'EMBED_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print(f"   모델: {model_name}")
    embedding_model = SentenceTransformer(model_name, device='cpu')
    
    # Create embeddings
    print("🔢 임베딩 생성 중...")
    new_embeddings = embedding_model.encode(new_chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(new_embeddings)
    
    print(f"✅ {len(new_embeddings)}개 임베딩 생성 완료")
    
    # Add to index
    print("📝 인덱스에 추가 중...")
    faiss_index.add(new_embeddings)
    passages.extend(new_chunks)
    
    print(f"✅ 인덱스에 추가 완료: 총 {len(passages)}개 passages")
    
    # Save index
    print("💾 인덱스 저장 중...")
    save_index(faiss_index, passages)
    
    # Verify save
    print("🔍 저장 확인 중...")
    import json
    with open(passages_file, 'r', encoding='utf-8') as f:
        saved_passages = json.load(f)
    print(f"   저장된 passages 수: {len(saved_passages)}")
    bms_mileage_count = sum(1 for p in saved_passages if 'bms_mileage' in p.lower() or '주행거리 산정' in p or 'BMS 기반 주행거리' in p)
    print(f"   BMS 주행거리 관련 passages: {bms_mileage_count}개")
    
    print("✅ 완료! BMS 주행거리 계산 규칙이 인덱스에 추가되었습니다.")
    print(f"   - 추가된 청크 수: {len(new_chunks)}")
    print(f"   - 총 passages 수: {len(passages)}")

if __name__ == "__main__":
    add_bms_mileage_rules()

