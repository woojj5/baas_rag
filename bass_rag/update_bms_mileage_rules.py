#!/usr/bin/env python3
"""Update BMS mileage calculation rules in RAG index (remove old, add new)."""
import sys
from pathlib import Path
import json
import numpy as np
import faiss

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag_index import chunk_text, save_index, load_index
from app.config import Config
from sentence_transformers import SentenceTransformer

def update_bms_mileage_rules():
    """Update BMS mileage calculation rules in the index."""
    # Load existing index
    print("📖 기존 인덱스 로드 중...")
    try:
        faiss_index, passages, model_name, doc_id_to_original, embeddings = load_index()
        print(f"✅ 기존 인덱스 로드 완료: {len(passages)}개 passages")
    except FileNotFoundError:
        print("❌ 인덱스 파일을 찾을 수 없습니다. build_index.py를 먼저 실행해주세요.")
        return
    except Exception as e:
        print(f"❌ 인덱스 로드 중 오류 발생: {e}")
        return
    
    # Find and remove old BMS mileage rules passages
    print("🔍 기존 BMS 주행거리 규칙 패시지 찾는 중...")
    indices_to_remove = []
    for i, passage in enumerate(passages):
        if 'bms_mileage_calculation_rules' in passage.lower() or 'BMS 기반 주행거리 산정 로직' in passage:
            indices_to_remove.append(i)
    
    if indices_to_remove:
        print(f"🗑️  {len(indices_to_remove)}개 기존 패시지 제거 중...")
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            passages.pop(idx)
            # Remove corresponding embedding
            if embeddings is not None and idx < len(embeddings):
                embeddings = np.delete(embeddings, idx, axis=0)
        
        # Rebuild FAISS index
        print("🔨 FAISS 인덱스 재구성 중...")
        dimension = faiss_index.d
        faiss_index = faiss.IndexFlatL2(dimension)
        if embeddings is not None and len(embeddings) > 0:
            faiss_index.add(embeddings)
        print(f"✅ 인덱스 재구성 완료: {len(passages)}개 passages")
    else:
        print("ℹ️  기존 패시지를 찾을 수 없습니다. 새로 추가합니다.")
    
    # Read updated BMS mileage rules file
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
    model_name_attr = getattr(Config, 'EMBED_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print(f"   모델: {model_name_attr}")
    embedding_model = SentenceTransformer(model_name_attr, device='cpu')
    
    # Create embeddings
    print("🔢 임베딩 생성 중...")
    new_embeddings = embedding_model.encode(new_chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(new_embeddings)
    
    print(f"✅ {len(new_embeddings)}개 임베딩 생성 완료")
    
    # Add to index
    print("📝 인덱스에 추가 중...")
    faiss_index.add(new_embeddings)
    passages.extend(new_chunks)
    
    # Update embeddings
    if embeddings is not None:
        updated_embeddings = np.vstack((embeddings, new_embeddings))
    else:
        updated_embeddings = new_embeddings
    
    print(f"✅ 인덱스에 추가 완료: 총 {len(passages)}개 passages")
    
    # Save index
    print("💾 인덱스 저장 중...")
    save_index(faiss_index, passages, doc_id_to_original, updated_embeddings)
    
    # Verify save
    print("🔍 저장 확인 중...")
    _, loaded_passages, _, _, _ = load_index()
    print(f"   저장된 passages 수: {len(loaded_passages)}")
    
    mileage_passages_count = sum(1 for p in loaded_passages if "BMS 기반 주행거리 산정 로직" in p)
    p_kw_count = sum(1 for p in loaded_passages if "p_kw" in p.lower() and "계산된 값" in p)
    print(f"   BMS 주행거리 관련 passages: {mileage_passages_count}개")
    print(f"   p_kw가 '계산된 값'으로 명시된 passages: {p_kw_count}개")
    
    print("✅ 완료! BMS 주행거리 계산 규칙이 인덱스에 업데이트되었습니다.")
    print(f"   - 제거된 기존 패시지: {len(indices_to_remove)}개")
    print(f"   - 추가된 새 청크 수: {len(new_chunks)}개")
    print(f"   - 총 passages 수: {len(passages)}개")

if __name__ == "__main__":
    update_bms_mileage_rules()

