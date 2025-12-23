#!/usr/bin/env python3
"""Phase 1 테스트: 테이블 필터링 버그 수정 검증"""

import requests
import json
import time

# 서버 URL
REFRAG_URL = "http://localhost:8012/query"

def test_query(url, question, server_name):
    """단일 쿼리 테스트"""
    print(f"\n{'='*80}")
    print(f"[{server_name}] 테스트 질문: {question}")
    print('='*80)
    
    payload = {
        "query": question,
        "top_k": 8
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            
            # 응답에서 LLM 지연시간 정보 추출 (있는 경우)
            llm_latency_ms = result.get("llm_latency_ms", 0)
            llm_latency_sec = llm_latency_ms / 1000.0 if llm_latency_ms else 0
            
            print(f"\n✅ 응답 성공")
            print(f"   총 소요 시간: {elapsed:.2f}초")
            if llm_latency_sec > 0:
                print(f"   LLM 지연시간: {llm_latency_sec:.2f}초 ({llm_latency_ms:.0f}ms)")
                print(f"   네트워크/처리 시간: {elapsed - llm_latency_sec:.2f}초")
            print(f"\n답변:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            # 검증: BMS 변수 확인
            bms_vars = ['seq', 'messageTime', 'msgTime', 'msg_id', 'pack_volt', 'soc', 'cell_v_', 'mod_temp_']
            gps_vars = ['lat', 'lon', 'speed', 'direction', 'hdop', 'fuel_pct']
            
            if 'gps' in question.lower():
                # GPS 질문에 BMS 변수가 포함되어 있으면 오류
                found_bms = [var for var in bms_vars if var.lower() in answer.lower()]
                if found_bms:
                    print(f"\n❌ 오류 발견: GPS 질문에 BMS 변수가 포함됨: {found_bms}")
                    return False, elapsed, llm_latency_sec
                else:
                    print(f"\n✅ 검증 통과: GPS 질문에 BMS 변수 없음")
                    return True, elapsed, llm_latency_sec
            elif 'bms' in question.lower():
                # BMS 질문에 GPS 변수가 포함되어 있으면 오류
                found_gps = [var for var in gps_vars if var.lower() in answer.lower()]
                if found_gps:
                    print(f"\n❌ 오류 발견: BMS 질문에 GPS 변수가 포함됨: {found_gps}")
                    return False, elapsed, llm_latency_sec
                else:
                    print(f"\n✅ 검증 통과: BMS 질문에 GPS 변수 없음")
                    return True, elapsed, llm_latency_sec
        else:
            print(f"\n❌ 오류: HTTP {response.status_code}")
            print(response.text)
            return False, elapsed, 0
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 예외 발생 (소요 시간: {elapsed:.2f}초): {e}")
        return False, elapsed, 0

def main():
    """메인 테스트 함수"""
    print("="*80)
    print("Phase 1 테스트: 테이블 필터링 버그 수정 검증")
    print("="*80)
    
    # 테스트 쿼리 목록
    test_cases = [
        # GPS 테이블 질문 (BMS 변수 제외 확인)
        {
            "question": "GPS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?",
            "expected": "BMS 변수(seq, messageTime 등) 제외"
        },
        {
            "question": "GPS 테이블에 있는 변수 중에서 사용하지 않는 것을 나열해라",
            "expected": "BMS 변수 제외"
        },
        # BMS 테이블 질문 (GPS 변수 제외 확인)
        {
            "question": "BMS 테이블에 존재하는 변수지만, 사용하지 않는 변수는?",
            "expected": "GPS 변수(lat, lon, speed 등) 제외"
        },
        {
            "question": "BMS 테이블에 있는 변수 중에서 사용하지 않는 것을 나열해라",
            "expected": "GPS 변수 제외"
        },
    ]
    
    results = []
    timings = []  # (총 시간, LLM 시간) 튜플 리스트
    
    # REFRAG 서버 테스트만 실행
    print("\n" + "="*80)
    print("REFRAG 서버 테스트 (포트 8012)")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}/{len(test_cases)}]")
        result, total_time, llm_time = test_query(REFRAG_URL, test_case["question"], "REFRAG")
        results.append(("REFRAG", test_case["question"], result))
        timings.append((total_time, llm_time))
        time.sleep(1)  # 서버 부하 방지
    
    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    passed = sum(1 for _, _, result in results if result)
    total = len(results)
    
    for i, (server, question, result) in enumerate(results):
        status = "✅ 통과" if result else "❌ 실패"
        total_time, llm_time = timings[i]
        print(f"{status} [{server}] {question[:50]}... (총: {total_time:.2f}초, LLM: {llm_time:.2f}초)")
    
    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    # 속도 통계
    if timings:
        total_times = [t[0] for t in timings]
        llm_times = [t[1] for t in timings if t[1] > 0]
        
        print("\n" + "="*80)
        print("속도 통계")
        print("="*80)
        print(f"총 소요 시간:")
        print(f"  평균: {sum(total_times)/len(total_times):.2f}초")
        print(f"  최소: {min(total_times):.2f}초")
        print(f"  최대: {max(total_times):.2f}초")
        print(f"  합계: {sum(total_times):.2f}초")
        
        if llm_times:
            print(f"\nLLM 지연시간:")
            print(f"  평균: {sum(llm_times)/len(llm_times):.2f}초 ({sum(llm_times)/len(llm_times)*1000:.0f}ms)")
            print(f"  최소: {min(llm_times):.2f}초 ({min(llm_times)*1000:.0f}ms)")
            print(f"  최대: {max(llm_times):.2f}초 ({max(llm_times)*1000:.0f}ms)")
            print(f"  합계: {sum(llm_times):.2f}초 ({sum(llm_times)*1000:.0f}ms)")
            
            # LLM 비율 계산 (LLM 시간이 있는 경우만)
            if llm_times:
                # LLM 시간이 있는 테스트만 필터링
                total_with_llm = [total_times[i] for i, (_, llm) in enumerate(timings) if llm > 0]
                if total_with_llm:
                    avg_total = sum(total_with_llm)/len(total_with_llm)
                    avg_llm = sum(llm_times)/len(llm_times)
                    llm_ratio = (avg_llm / avg_total * 100) if avg_total > 0 else 0
                    print(f"\nLLM이 전체 시간의 {llm_ratio:.1f}% 차지 (LLM 사용된 테스트 기준)")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! Phase 1 수정이 성공적으로 적용되었습니다.")
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패. 추가 수정이 필요할 수 있습니다.")

if __name__ == "__main__":
    main()

