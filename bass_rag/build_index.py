"""CLI script to build RAG index."""
from app.config import Config
from app.rag_index import build_index, save_index


def main():
    """Build and save RAG index with document summaries."""
    print("인덱스 생성 시작...")
    print("문서 요약 생성 중... (Summary 기반 검색)")
    
    try:
        index, passages, model, doc_id_to_original = build_index(use_summaries=True)
        save_index(index, passages, doc_id_to_original)
        
        print(f"완료: {len(passages)}개 패시지 (Summary), {index.ntotal}개 벡터")
        print(f"원본 문서 매핑: {len(doc_id_to_original)}개 문서")
        print(f"인덱스 저장 위치: {Config.INDEX_DIR}")
        print("\n참고: 검색은 Summary로 수행되며, 필요시 원본 문서를 참조합니다.")
    except Exception as e:
        print(f"오류: {e}")
        raise


if __name__ == "__main__":
    main()
