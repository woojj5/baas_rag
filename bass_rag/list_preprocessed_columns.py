#!/usr/bin/env python3
"""List columns from preprocessed CSV files."""
import pandas as pd
from pathlib import Path
from app.config import Config

def list_preprocessed_columns():
    """List columns from all preprocessed BMS and GPS CSV files."""
    data_dir = Config.ROOT / "data"
    
    # Find all preprocessed files
    bms_files = sorted(data_dir.glob("preprocessed_bms*.csv"))
    gps_files = sorted(data_dir.glob("preprocessed_gps*.csv"))
    
    print("=" * 80)
    print("전처리된 CSV 파일들의 칼럼 목록")
    print("=" * 80)
    
    if not bms_files and not gps_files:
        print("전처리된 파일을 찾을 수 없습니다.")
        return
    
    # Process BMS files
    if bms_files:
        print(f"\n[BMS 파일] ({len(bms_files)}개)")
        print("-" * 80)
        for file_path in bms_files:
            try:
                # Read only header
                df = pd.read_csv(file_path, nrows=0)
                columns = list(df.columns)
                print(f"\n파일: {file_path.name}")
                print(f"칼럼 수: {len(columns)}")
                print(f"칼럼 목록:")
                for i, col in enumerate(columns, 1):
                    print(f"  {i:3d}. {col}")
            except Exception as e:
                print(f"\n파일: {file_path.name}")
                print(f"  오류: {e}")
    
    # Process GPS files
    if gps_files:
        print(f"\n[GPS 파일] ({len(gps_files)}개)")
        print("-" * 80)
        for file_path in gps_files:
            try:
                # Read only header
                df = pd.read_csv(file_path, nrows=0)
                columns = list(df.columns)
                print(f"\n파일: {file_path.name}")
                print(f"칼럼 수: {len(columns)}")
                print(f"칼럼 목록:")
                for i, col in enumerate(columns, 1):
                    print(f"  {i:3d}. {col}")
            except Exception as e:
                print(f"\n파일: {file_path.name}")
                print(f"  오류: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("요약")
    print("=" * 80)
    
    all_bms_columns = set()
    all_gps_columns = set()
    
    for file_path in bms_files:
        try:
            df = pd.read_csv(file_path, nrows=0)
            all_bms_columns.update(df.columns)
        except:
            pass
    
    for file_path in gps_files:
        try:
            df = pd.read_csv(file_path, nrows=0)
            all_gps_columns.update(df.columns)
        except:
            pass
    
    if all_bms_columns:
        print(f"\nBMS 파일들의 고유 칼럼 수: {len(all_bms_columns)}")
        print("고유 칼럼 목록:")
        for i, col in enumerate(sorted(all_bms_columns), 1):
            print(f"  {i:3d}. {col}")
    
    if all_gps_columns:
        print(f"\nGPS 파일들의 고유 칼럼 수: {len(all_gps_columns)}")
        print("고유 칼럼 목록:")
        for i, col in enumerate(sorted(all_gps_columns), 1):
            print(f"  {i:3d}. {col}")

if __name__ == "__main__":
    list_preprocessed_columns()

