"""CLI script to preprocess data files."""
from pathlib import Path
from app.preprocessor import preprocess_all_files
from app.config import Config
import json


def main():
    """Preprocess all files in before_preprocess directory."""
    print("데이터 전처리 시작...")
    print(f"입력 디렉토리: {Config.ROOT / 'before_preprocess'}")
    print(f"출력 디렉토리: {Config.ROOT / 'data'}")
    
    results = preprocess_all_files()
    
    print(f"\n처리 완료:")
    print(f"- 성공: {len(results['processed'])}개 파일")
    print(f"- 실패: {len(results['errors'])}개 파일")
    
    for item in results["processed"]:
        print(f"\n[{item['type']}] {item['file']}")
        print(f"  행 수: {item['stats']['original_rows']}")
        print(f"  컬럼 수: {item['stats']['columns']}")
        if item['stats']['device_no']:
            print(f"  Device: {item['stats']['device_no']}")
    
    if results["errors"]:
        print(f"\n오류:")
        for err in results["errors"]:
            print(f"  {err['file']}: {err['error']}")
    
    print(f"\n전처리된 파일은 {Config.ROOT / 'data'} 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    main()

