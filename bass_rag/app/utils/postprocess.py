"""Answer post-processing utilities for RAG/REFRAG servers."""
import re
from typing import Optional

from app.domain_dict import get_domain_dict
from app.data_columns import (
    get_actual_used_columns,
    load_specification_columns,
    COLUMN_NAME_MAPPING,
    normalize_variable_name,
    find_matching_column
)


def postprocess_answer(
    answer: str,
    table_type: str = "bms",
    is_not_used_query: bool = False,
    domain_dict=None,
    question: str = ""  # 질문 내용 추가
) -> str:
    """
    답변 후처리: 서론/사족 제거 및 사용되는 변수 제외 (유사도 기반)
    테이블별 변수 필터링: GPS 질문에 BMS 변수 제외, BMS 질문에 GPS 변수 제외
    
    Args:
        answer: 원본 답변
        table_type: "bms" or "gps" (기본값: "bms")
        is_not_used_query: "사용하지 않는 변수" 질문인지 여부
        domain_dict: DomainDictionary 인스턴스 (테이블 필터링용, None이면 자동 로드)
        
    Returns:
        후처리된 답변
    """
    if not answer:
        return answer
    
    # domain_dict 로드 (테이블 필터링용)
    if domain_dict is None:
        domain_dict = get_domain_dict()
    
    # 원본 답변 저장 (빈 답변 보호용)
    original_answer = answer
    
    # 속도 관련 질문인지 확인 (최우선)
    question_lower_pre = question.lower() if question else ""
    is_speed_query_pre = any(keyword in question_lower_pre for keyword in ['속도', 'speed', 'velocity', 'vel'])
    is_battery_query_pre = any(keyword in question_lower_pre for keyword in ['배터리', 'battery', 'batt', '셀', 'cell', '팩', 'pack', '모듈', 'module', 'soc', 'soh', 'socd'])
    
    # 배터리 관련 질문인 경우, 답변이 너무 짧으면 배터리 관련 변수 강제 추가
    if is_battery_query_pre and not is_not_used_query:
        # 답변에서 변수 추출
        vars_in_answer = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', answer)
        battery_keywords = ['soc', 'socd', 'soh', 'cell', 'volt', 'pack', 'module', 'batt', 'temp', 'insul', 'deter']
        battery_vars_in_answer = [v for v in vars_in_answer if any(kw in v.lower() for kw in battery_keywords)]
        
        # 배터리 관련 변수가 3개 미만이면 강제로 추가
        if len(battery_vars_in_answer) < 3:
            file_columns = get_actual_used_columns(table_type)
            columns = file_columns.get("columns", set()) if isinstance(file_columns, dict) else file_columns
            
            # 배터리 관련 변수 목록 생성 (개별 셀/모듈 변수 제외)
            battery_vars = []
            for col in sorted(columns):
                col_lower = col.lower()
                # 개별 셀 전압 변수 제외 (cell_v_001, cell_v_002 등)
                if col_lower.startswith('cell_v_') and col_lower[7:].isdigit():
                    continue
                # 개별 모듈 온도 변수 제외 (mod_temp_01, mod_temp_02 등)
                if col_lower.startswith('mod_temp_') and (col_lower[9:].isdigit() or len(col_lower[9:]) == 2):
                    continue
                # 배터리 관련 키워드가 있는 변수만 포함
                if any(kw in col_lower for kw in battery_keywords):
                    battery_vars.append(col)
            
            # 이미 답변에 있는 변수 제외
            existing_vars_lower = {v.lower() for v in vars_in_answer}
            new_battery_vars = [v for v in battery_vars if v.lower() not in existing_vars_lower]
            
            # 최대 20개까지 추가 (핵심 변수 우선)
            # 우선순위: soc, soh, socd, pack*, cell_volt_list, mod_temp_list, batt*, moduleMax/Min/Avg, max/min_cell_volt 등
            priority_vars = []
            other_vars = []
            for v in new_battery_vars:
                v_lower = v.lower()
                if any(priority in v_lower for priority in ['soc', 'soh', 'pack_volt', 'pack_current', 'cell_volt_list', 'mod_temp_list', 'batt_internal', 'modulemax', 'modulemin', 'moduleavg', 'max_cell_volt', 'min_cell_volt', 'insul_resistance', 'deter']):
                    priority_vars.append(v)
                else:
                    other_vars.append(v)
            
            # 우선순위 변수 먼저, 나머지는 나중에
            sorted_vars = sorted(priority_vars) + sorted(other_vars)
            additional_vars = sorted_vars[:20]
            
            if additional_vars:
                if answer.strip():
                    # 쉼표로 구분된 형식으로 추가
                    answer = answer.rstrip('.,;')
                    if answer and not answer.endswith(','):
                        answer += ', '
                    answer += ', '.join(additional_vars)
                else:
                    answer = ', '.join(additional_vars)
    
    # 속도 관련 질문인 경우, 답변에서 속도와 무관한 변수 즉시 제거
    if is_speed_query_pre:
        # 속도와 무관한 변수 목록 (명시적 제외)
        speed_unrelated_vars = [
            'fastChargingPortConnected', 'fastchargingportconnected', 'fast_charging_port_connected',
            'cumulativeCurrentCharged', 'cumulativecurrentcharged', 'cumulative_current_charged',
            'cumulativeCurrentDischarged', 'cumulativecurrentdischarged', 'cumulative_current_discharged',
            'cumulativePowerCharged', 'cumulativepowercharged', 'cumulative_power_charged',
            'cumulativePowerDischarged', 'cumulativepowerdischarged', 'cumulative_power_discharged'
        ]
        
        # 각 제외 변수를 답변에서 제거
        for var in speed_unrelated_vars:
            # 변수명만 제거 (다른 단어의 일부가 아닌 경우)
            # 쉼표로 구분된 변수 리스트에서 제거
            answer = re.sub(rf'\b{re.escape(var)}\b,?\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf',\s*\b{re.escape(var)}\b', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf'\b{re.escape(var)}\b', '', answer, flags=re.IGNORECASE)
        
        # cumulative로 시작하는 모든 변수 제거
        answer = re.sub(r'\b[cC]umulative[A-Za-z0-9_]*\b,?\s*', '', answer)
        answer = re.sub(r',\s*\b[cC]umulative[A-Za-z0-9_]*\b', '', answer)
        
        # fastCharging, chargingPort, connected 관련 변수 제거
        answer = re.sub(r'\b[a-zA-Z]*[Ff]ast[Cc]harging[A-Za-z0-9_]*\b,?\s*', '', answer)
        answer = re.sub(r'\b[a-zA-Z]*[Cc]harging[Pp]ort[A-Za-z0-9_]*\b,?\s*', '', answer)
        answer = re.sub(r'\b[a-zA-Z]*[Cc]onnected[A-Za-z0-9_]*\b,?\s*', '', answer)
        
        # 속도 관련 변수만 남기기
        speed_related_patterns = [
            r'\bemobility[_ ]?spd\b',
            r'\bspeed\b',
            r'\bvelocity\b',
            r'\bdrive[_ ]?motor[_ ]?spd[12]\b',
            r'\bdrivemotorspd[12]\b'
        ]
        
        # 답변을 줄 단위로 분리하고 속도 관련 변수만 포함하는 줄만 남기기
        lines = answer.split('\n')
        filtered_lines_pre = []
        for line in lines:
            line_lower = line.lower()
            # 속도와 무관한 키워드가 포함된 줄 제외
            if any(keyword in line_lower for keyword in ['fastchargingportconnected', 'cumulativecurrentcharged', 'cumulativecurrentdischarged', 'cumulative', 'chargingport', 'connected']):
                # 속도 관련 변수가 있는지 확인
                has_speed_var = any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in speed_related_patterns)
                if not has_speed_var:
                    continue  # 속도 관련 변수가 없으면 줄 제외
            filtered_lines_pre.append(line)
        answer = '\n'.join(filtered_lines_pre)
        
        # 빈 줄 정리
        answer = re.sub(r'\n\s*\n+', '\n', answer)
        answer = answer.strip()
    
    # 서론 패턴 제거 (보수적으로 - "사용하지 않는 변수" 질문에서는 덜 제거)
    if not is_not_used_query:
        intro_patterns = [
            r'^제공된 정보에서[^.]*\.',
            r'^제공된 정보를 바탕으로[^.]*\.',
            r'^제공된 정보를 종합하면[^.]*\.',
            r'^제공된 정보에서 명시적으로[^.]*\.',
            r'^제공된 정보에서 사용하지 않는 변수를 정확히 파악하기는 어렵습니다[^.]*\.',
            r'^하지만[^.]*\.',
            r'^다만[^.]*\.',
            r'^그러나[^.]*\.',
            r'^또한[^.]*\.',
            r'^또한, 다른 청크들을 종합적으로 고려했을 때[^.]*\.',
            r'^또한, 다른 청크들을 종합적으로 고려했을 때, 다음과 같은 변수들이[^.]*\.',
        ]
        
        for pattern in intro_patterns:
            answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE)
    
    # "주의:", "참고:" 등으로 시작하는 사족 제거 (보수적으로)
    caution_patterns = [
        r'\n\s*\*\s*주의:.*$',
        r'\n\s*\*\s*참고:.*$',
        r'\n\s*\*\s*주의\*\s*:.*$',
        r'\n\s*주의:.*$',
        r'\n\s*참고:.*$',
        r'\n\s*주의\*\s*:.*$',
        r'\(이는 추정이며.*?\)',
        r'\(실제 사용 여부는.*?\)',
        r'\(추가 정보가 필요.*?\)',
        r'위 목록은.*?추정이며.*?',
        r'실제 사용 여부는.*?프로젝트의.*?요구사항에 따라.*?',
    ]
    
    for pattern in caution_patterns:
        answer = re.sub(pattern, '', answer, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    
    # 질문에서 테이블 타입 추론 (질문 우선, 없으면 답변, 없으면 기본값)
    question_lower_for_table = question.lower() if question else ""
    if 'gps' in question_lower_for_table and 'bms' not in question_lower_for_table:
        inferred_table_type = "gps"
    elif 'bms' in question_lower_for_table:
        inferred_table_type = "bms"
    else:
        # 답변에서 테이블 타입 추론 (BMS 또는 GPS)
        answer_lower = answer.lower()
        if 'gps' in answer_lower and 'bms' not in answer_lower:
            inferred_table_type = "gps"
        else:
            inferred_table_type = table_type  # 기본값 또는 전달받은 값 사용
    
    # 테이블이 명시되지 않았는지 확인
    question_lower_check = question.lower() if question else ""
    table_not_specified = 'bms' not in question_lower_check and 'gps' not in question_lower_check
    
    # 실제 사용되는 컬럼 목록 가져오기
    if table_not_specified:
        # 테이블이 명시되지 않았으면 BMS와 GPS 모두 포함
        bms_columns_dict = get_actual_used_columns("bms")
        gps_columns_dict = get_actual_used_columns("gps")
        actual_columns = bms_columns_dict.get("columns", set()) | gps_columns_dict.get("columns", set())
        
        # BMS와 GPS 규격 파일 모두 가져오기
        bms_spec_columns = load_specification_columns("bms")
        gps_spec_columns = load_specification_columns("gps")
        spec_columns = bms_spec_columns | gps_spec_columns
    else:
        actual_columns_dict = get_actual_used_columns(inferred_table_type)
        actual_columns = actual_columns_dict.get("columns", set())
        spec_columns = load_specification_columns(inferred_table_type)
    
    # "실제 사용하지 않음" 키워드
    not_used_keywords = [
        '실제 사용하지 않음',
        '개발상이유로 존재',
        '개발상의 이유로 존재',
        '개발상 이유로 존재'
    ]
    
    # 답변에서 변수명 추출 및 필터링 (비고 우선, 그 다음 유사도 기반)
    # 불릿 포인트 형식의 여러 줄을 하나로 합치기
    lines = answer.split('\n')
    
    # 불릿 포인트 형식의 연속된 줄들을 하나로 합치기
    merged_lines = []
    current_bullet_block = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # 불릿 포인트 형식인지 확인 (* 또는 - 로 시작)
        is_bullet = bool(re.match(r'^\s*[*-]\s+', line_stripped))
        
        if is_bullet:
            current_bullet_block.append(line)
        else:
            # 불릿 포인트 블록이 있으면 먼저 합치기
            if current_bullet_block:
                merged_lines.append('\n'.join(current_bullet_block))
                current_bullet_block = []
            merged_lines.append(line)
    
    # 마지막 불릿 포인트 블록 처리
    if current_bullet_block:
        merged_lines.append('\n'.join(current_bullet_block))
    
    lines = merged_lines
    filtered_lines = []
    
    # "사용하지 않는 변수" 질문인 경우, 모든 변수를 한 번에 추출하여 필터링
    if is_not_used_query:
        # 모든 변수명 추출 (불릿 포인트, 쉼표 구분, 콜론 형식 모두)
        all_vars = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # 불릿 포인트 형식에서 변수 추출
            bullet_vars = re.findall(r'[*-]\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            all_vars.extend(bullet_vars)
            
            # 쉼표 구분 형식에서 변수 추출 (불릿 포인트가 아닌 경우만)
            if not bullet_vars:
                comma_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\b', line)
                # 콜론 형식 확인
                colon_match = re.search(r':\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)', line)
                if colon_match:
                    colon_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\b', colon_match.group(1))
                    all_vars.extend(colon_vars)
                elif len(comma_vars) >= 2:  # 쉼표로 구분된 변수 리스트
                    all_vars.extend(comma_vars)
        
        # 중복 제거 (순서 유지)
        all_vars = list(dict.fromkeys(all_vars))
        
        # 각 변수를 필터링
        filtered_vars = []
        for var_match in all_vars:
            # 짧은 변수명 제외
            if len(var_match) <= 2:
                continue
            
            # 테이블 필터링
            var_belongs_to_table = False
            if domain_dict and inferred_table_type:
                normalized_var = normalize_variable_name(var_match)
                table_vars = domain_dict.table_to_variables.get(inferred_table_type.upper(), set())
                mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
                
                if (var_match in table_vars or normalized_var in table_vars or 
                    mapped_var in table_vars):
                    var_belongs_to_table = True
                else:
                    var_info = domain_dict.variable_to_info.get(var_match, {})
                    if not var_info:
                        var_info = domain_dict.variable_to_info.get(mapped_var, {})
                    var_table = var_info.get("table", "").upper() if var_info.get("table") else ""
                    if var_table == inferred_table_type.upper():
                        var_belongs_to_table = True
            
            if not var_belongs_to_table and inferred_table_type:
                continue
            
            # 비고 정보 확인
            var_info = domain_dict.variable_to_info.get(var_match, {})
            if not var_info:
                normalized_var = normalize_variable_name(var_match)
                mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
                var_info = domain_dict.variable_to_info.get(mapped_var, {})
            
            note = var_info.get("note", "") if var_info else ""
            note_lower = note.lower() if note else ""
            
            # "순수BMS데이터" 등 제외 비고 확인
            excluded_notes = [
                '순수bms데이터', '순수 bms 데이터',
                '단말 처리상 정의 항목', '단말처리상 정의 항목',
                'db상 정의 항목', 'db 상 정의 항목',
                'gps데이터', 'gps 데이터'
            ]
            if any(excluded in note_lower for excluded in excluded_notes):
                continue  # 제외
            
            # 규격 파일 확인
            normalized_var = normalize_variable_name(var_match)
            mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
            
            is_in_spec = (
                var_match in actual_columns or
                var_match.lower() in {c.lower() for c in actual_columns} or
                normalized_var in actual_columns or
                mapped_var in actual_columns or
                mapped_var.lower() in {c.lower() for c in actual_columns}
            )
            
            if not is_in_spec:
                matched_column = find_matching_column(var_match, actual_columns)
                if matched_column is not None:
                    is_in_spec = True
                if not is_in_spec and mapped_var != normalized_var:
                    matched_column = find_matching_column(mapped_var, actual_columns)
                    if matched_column is not None:
                        is_in_spec = True
            
            if is_in_spec and var_match.lower() != 'seq':
                continue  # 제외
            
            # 포함 (사용되지 않는 변수)
            filtered_vars.append(var_match)
        
        # 필터링된 변수가 있으면 답변 재구성
        if filtered_vars:
            # 원본 답변 형식 유지
            if any('*' in line or '-' in line for line in lines):
                bullet_char = '*' if any('*' in line for line in lines) else '-'
                filtered_lines = [f"{bullet_char}   {var}" for var in filtered_vars]
            else:
                prefix = "BMS 테이블에 존재하지만 사용하지 않는 변수: " if inferred_table_type == "bms" else "사용하지 않는 변수: "
                filtered_lines = [prefix + ', '.join(filtered_vars)]
        else:
            # 필터링된 변수가 없으면 "seq"만 포함
            filtered_lines = ["BMS 테이블에 존재하지만 사용하지 않는 변수: seq"]
        
        answer = '\n'.join(filtered_lines)
        return answer.strip()
    
    # 일반 질문 처리 (기존 로직)
    # 속도 관련 질문인지 확인
    question_lower_global = question.lower() if question else ""
    is_speed_query_global = any(keyword in question_lower_global for keyword in ['속도', 'speed', 'velocity', 'vel'])
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            filtered_lines.append(line)
            continue
        
        line_lower = line.lower()
        
        # 속도 관련 질문인 경우, 속도와 무관한 변수가 포함된 라인 제외 (최우선)
        if is_speed_query_global:
            # 속도와 무관한 키워드가 포함된 변수가 있으면 라인 제외
            speed_unrelated_in_line = any(
                keyword in line_lower for keyword in [
                    'fastchargingportconnected', 'cumulativecurrentcharged', 'cumulativecurrentdischarged',
                    'cumulative', 'chargingport', 'charging_port', 'fast_charging'
                ]
            )
            if speed_unrelated_in_line:
                # 속도 관련 변수만 추출
                speed_related_in_line = any(
                    keyword in line_lower for keyword in [
                        'emobility_spd', 'emobilityspeed', 'speed', 'velocity', 'drive_motor_spd', 'drivemotorspd'
                    ]
                )
                if not speed_related_in_line:
                    # 속도 관련 변수가 없으면 라인 제외
                    continue
        
        # [1단계] 최상단: "실제 사용하지 않음" 비고 확인
        # 비고가 있으면 유사도와 관계없이 항상 포함
        has_not_used_note = any(keyword in line for keyword in not_used_keywords)
        
        if has_not_used_note:
            filtered_lines.append(line)
            continue
        
        # [2단계] 비고가 없으면 유사도 검사
        # 변수명 추출
        var_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
        
        if not var_matches:
            # 변수명이 없으면 그대로 포함
            filtered_lines.append(line)
            continue
        
        # 라인이 변수명만 포함하고 있는지 확인 (변수명 리스트 형태)
        # "BMS 테이블에 존재하지만 사용하지 않는 변수: 변수1, 변수2, ..." 형식도 인식
        # 불릿 포인트 형식 (* 변수명)도 인식
        line_cleaned = re.sub(r'\s+', ' ', line.strip())
        # 쉼표로 구분되고, 변수명 패턴이 여러 개 있으면 변수명 리스트로 간주
        # 또는 "변수: " 뒤에 변수명 리스트가 있는 형식도 인식
        # 또는 불릿 포인트 형식 (* 변수명)도 인식
        has_comma_separated = ',' in line_cleaned
        has_colon_format = bool(re.search(r':\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)', line_cleaned))
        # 불릿 포인트 형식 확인 (여러 줄이 합쳐진 경우도 처리)
        has_bullet_format = bool(re.search(r'[*-]\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_cleaned))
        # 여러 줄이 합쳐진 경우 불릿 포인트가 여러 개 있을 수 있음
        bullet_count = len(re.findall(r'[*-]\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_cleaned))
        is_variable_list_line = (has_comma_separated and len(var_matches) >= 2) or (has_colon_format and len(var_matches) >= 1) or (has_bullet_format and (len(var_matches) >= 1 or bullet_count >= 1))
        
        # 변수명 리스트 형태의 라인 또는 "사용하지 않는 변수" 질문인 경우 유사도 체크
        # "사용하지 않는 변수" 질문에서는 항상 유사도 체크 수행
        # "속도와 관련된 변수" 질문도 필터링 필요
        question_lower_check = question.lower() if question else ""
        is_speed_query_check = any(keyword in question_lower_check for keyword in ['속도', 'speed', 'velocity', 'vel'])
        should_check_similarity = is_variable_list_line or is_not_used_query or is_speed_query_check
        
        if should_check_similarity:
            # 각 변수명을 개별적으로 체크하여 사용되는 변수만 제외
            filtered_vars = []
            for var_match in var_matches:
                # 짧은 변수명 (1-2글자) 제외 (잘못 파싱된 것일 가능성)
                if len(var_match) <= 2:
                    continue
                
                # [테이블 필터링] 변수가 해당 테이블에 속하는지 확인
                var_belongs_to_table = False
                if domain_dict and inferred_table_type:
                    # 변수명 정규화 (camelCase -> snake_case 등)
                    normalized_var = normalize_variable_name(var_match)
                    
                    # 해당 테이블의 변수 목록 확인
                    table_vars = domain_dict.table_to_variables.get(inferred_table_type.upper(), set())
                    
                    # 정확한 변수명 매칭 확인
                    if var_match in table_vars or normalized_var in table_vars:
                        var_belongs_to_table = True
                    else:
                        # 변수명 매핑 확인 (예: messageTime -> msg_time)
                        mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
                        if mapped_var in table_vars:
                            var_belongs_to_table = True
                        else:
                            # variable_to_info에서 직접 확인
                            var_info = domain_dict.variable_to_info.get(var_match, {})
                            var_table = var_info.get("table", "").upper() if var_info.get("table") else ""
                            if var_table == inferred_table_type.upper():
                                var_belongs_to_table = True
                            # 매핑된 변수명으로도 확인
                            if not var_belongs_to_table:
                                mapped_info = domain_dict.variable_to_info.get(mapped_var, {})
                                mapped_table = mapped_info.get("table", "").upper() if mapped_info.get("table") else ""
                                if mapped_table == inferred_table_type.upper():
                                    var_belongs_to_table = True
                
                # 테이블 불일치 시 제외 (GPS 질문에 BMS 변수, BMS 질문에 GPS 변수)
                if not var_belongs_to_table and inferred_table_type:
                    # 해당 테이블에 속하지 않는 변수는 제외
                    continue
                
                # [중요] "속도와 관련된 변수" 질문 필터링 (최우선 적용)
                question_lower_for_speed = question.lower() if question else ""
                is_speed_query = any(keyword in question_lower_for_speed for keyword in ['속도', 'speed', 'velocity', 'vel'])
                
                if is_speed_query:
                    var_lower = var_match.lower()
                    
                    # 속도와 무관한 변수 명시적 제외 (최우선)
                    speed_unrelated_keywords = [
                        'charging', 'charge', '충전', 'port', '포트', 'connected', '연결',
                        'relay', '릴레이', 'cable', '케이블', 'fast', '빠른', 'quick',
                        'temp', '온도', 'voltage', '전압', 'current', '전류', 'power', '전력',
                        'cell', '셀', 'module', '모듈', 'battery', '배터리', 'soc', 'soh',
                        'fastcharging', 'fast_charging', 'chargingport', 'charging_port',
                        'cumulative'  # 누적 전류/전력 변수는 속도와 무관
                    ]
                    
                    # 속도와 무관한 키워드가 포함된 변수 즉시 제외
                    if any(keyword in var_lower for keyword in speed_unrelated_keywords):
                        continue
                    
                    # cumulative* 변수는 누적 전류/전력이므로 속도와 무관 (명시적 제외)
                    if 'cumulative' in var_lower:
                        continue
                    
                    # fastChargingPortConnected 같은 특정 변수 명시적 제외
                    if var_lower in ['fastchargingportconnected', 'fast_charging_port_connected', 'chargingportconnected', 'charging_port_connected']:
                        continue
                    
                    # cumulativeCurrentCharged, cumulativeCurrentDischarged 명시적 제외
                    if var_lower in ['cumulativecurrentcharged', 'cumulative_current_charged', 'cumulativecurrentdischarged', 'cumulative_current_discharged']:
                        continue
                    
                    # 속도와 직접 관련된 변수만 포함
                    speed_related_vars = {
                        'emobility_spd', 'emobilityspeed', 'emobility_speed',  # BMS 속도
                        'speed', 'velocity', 'vel',  # GPS 속도
                        'drive_motor_spd1', 'drive_motor_spd2', 'drivemotorspd1', 'drivemotorspd2'  # 구동 모터 속도
                    }
                    
                    normalized_var_speed = normalize_variable_name(var_match)
                    mapped_var_speed = COLUMN_NAME_MAPPING.get(var_match, normalized_var_speed)
                    
                    # 속도 관련 변수인지 확인
                    is_speed_related = (
                        var_lower in speed_related_vars or
                        normalized_var_speed in speed_related_vars or
                        mapped_var_speed in speed_related_vars or
                        'speed' in var_lower or
                        '속도' in var_match or
                        'spd' in var_lower
                    )
                    
                    # 속도 관련 변수가 아니면 제외
                    if not is_speed_related:
                        continue
                    
                    # 규격 파일에 있는지 확인
                    if spec_columns:
                        normalized_var_speed_check = normalize_variable_name(var_match)
                        mapped_var_speed_check = COLUMN_NAME_MAPPING.get(var_match, normalized_var_speed_check)
                        
                        is_in_spec_speed = (
                            var_match in spec_columns or
                            var_match.lower() in {c.lower() for c in spec_columns} or
                            normalized_var_speed_check in spec_columns or
                            mapped_var_speed_check in spec_columns or
                            mapped_var_speed_check.lower() in {c.lower() for c in spec_columns}
                        )
                        
                        # 유사도 기반 매칭도 시도
                        if not is_in_spec_speed:
                            matched_column = find_matching_column(var_match, spec_columns)
                            if matched_column is not None:
                                is_in_spec_speed = True
                        
                        # 규격 파일에 없으면 제외
                        if not is_in_spec_speed:
                            continue
                
                # [중요] BMS 규격 파일 검증: 규격 파일에 정의되지 않은 변수는 제외
                # 예: p_kw는 BMS 규격 파일에 없으므로 제외해야 함
                if spec_columns and inferred_table_type.lower() == "bms":
                    normalized_var_check = normalize_variable_name(var_match)
                    mapped_var_check = COLUMN_NAME_MAPPING.get(var_match, normalized_var_check)
                    
                    # 규격 파일에 있는지 확인 (정확한 매칭, 대소문자 무시, 정규화된 이름, 매핑된 이름 모두 확인)
                    is_in_spec = (
                        var_match in spec_columns or
                        var_match.lower() in {c.lower() for c in spec_columns} or
                        normalized_var_check in spec_columns or
                        mapped_var_check in spec_columns or
                        mapped_var_check.lower() in {c.lower() for c in spec_columns}
                    )
                    
                    # 유사도 기반 매칭도 시도
                    if not is_in_spec:
                        matched_column = find_matching_column(var_match, spec_columns)
                        if matched_column is not None:
                            is_in_spec = True
                        if not is_in_spec and mapped_var_check != normalized_var_check:
                            matched_column = find_matching_column(mapped_var_check, spec_columns)
                            if matched_column is not None:
                                is_in_spec = True
                    
                    # 특수 케이스: pack_voltage -> pack_volt 매칭
                    if not is_in_spec:
                        if var_match.lower() in ['pack_voltage', 'packvoltage'] or normalized_var_check in ['pack_voltage', 'packvoltage']:
                            if 'pack_volt' in spec_columns:
                                is_in_spec = True
                    
                    # 규격 파일에 없으면 제외 (BMS 테이블에 정의되지 않은 변수)
                    if not is_in_spec:
                        continue
                
                # [중요] "사용하지 않는 변수" 질문에서는 더 엄격한 필터링
                if is_not_used_query:
                    # 1. domain_dict에서 변수의 비고 정보 확인
                    var_info = domain_dict.variable_to_info.get(var_match, {})
                    if not var_info:
                        # 매핑된 변수명으로도 확인
                        normalized_var = normalize_variable_name(var_match)
                        mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
                        var_info = domain_dict.variable_to_info.get(mapped_var, {})
                    
                    note = var_info.get("note", "") if var_info else ""
                    note_lower = note.lower() if note else ""
                    
                    # 2. "순수BMS데이터", "단말 처리상 정의 항목", "DB상 정의 항목" 비고가 있으면 제외 (사용됨)
                    excluded_notes = [
                        '순수bms데이터', '순수 bms 데이터',
                        '단말 처리상 정의 항목', '단말처리상 정의 항목',
                        'db상 정의 항목', 'db 상 정의 항목',
                        'gps데이터', 'gps 데이터'
                    ]
                    has_excluded_note = any(excluded in note_lower for excluded in excluded_notes)
                    
                    if has_excluded_note:
                        # 제외 (사용되는 변수)
                        continue
                    
                    # 3. 규격 파일에 있는 변수는 무조건 제외 (사용됨)
                    # 정규화된 변수명으로 확인
                    normalized_var = normalize_variable_name(var_match)
                    mapped_var = COLUMN_NAME_MAPPING.get(var_match, normalized_var)
                    
                    # actual_columns에 있는지 확인 (정확한 매칭 + 유사도 매칭)
                    # 모든 가능한 변형을 확인
                    is_in_spec = (
                        var_match in actual_columns or
                        var_match.lower() in {c.lower() for c in actual_columns} or
                        normalized_var in actual_columns or
                        mapped_var in actual_columns or
                        mapped_var.lower() in {c.lower() for c in actual_columns}
                    )
                    
                    # 유사도 매칭도 확인 (더 강력하게)
                    if not is_in_spec:
                        matched_column = find_matching_column(var_match, actual_columns)
                        if matched_column is not None:
                            is_in_spec = True
                        # 매핑된 변수명으로도 확인
                        if not is_in_spec and mapped_var != normalized_var:
                            matched_column = find_matching_column(mapped_var, actual_columns)
                            if matched_column is not None:
                                is_in_spec = True
                    
                    if is_in_spec:
                        # 규격 파일에 있으면 제외 (사용됨)
                        # 단, seq는 예외: "실제 사용하지 않음" 비고가 있으므로 포함
                        if var_match.lower() != 'seq':
                            continue
                
                # [중요] 질문 의미 기반 필터링: "거리와 관련된 변수" 질문에서 거리와 무관한 변수 제외
                question_lower = question.lower() if question else ""
                is_distance_query = any(keyword in question_lower for keyword in ['거리', 'distance', 'mileage', '주행거리', 'odometer', '오도미터'])
                is_speed_query = any(keyword in question_lower for keyword in ['속도', 'speed', 'velocity', 'vel'])
                
                if is_distance_query:
                    # 거리와 무관한 변수 제외
                    var_lower = var_match.lower()
                    # p_kw는 BMS 테이블에 실제 컬럼이 아니므로 제외
                    if var_lower in ['p_kw', 'pkw', 'power']:
                        continue
                    # cumulativeCurrentCharged, cumulativeCurrentDischarged는 누적 전류량이므로 거리와 무관
                    if 'cumulative' in var_lower and ('current' in var_lower or '전류' in var_lower or 'chrgd' in var_lower or 'dischrgd' in var_lower):
                        continue
                    # cumulativePowerCharged, cumulativePowerDischarged도 누적 전력량이므로 거리와 무관
                    if 'cumulative' in var_lower and ('power' in var_lower or '전력' in var_lower or 'pw' in var_lower):
                        continue
                    
                    # "거리와 관련된 변수" 질문에서 규격 파일에 있는 변수만 포함
                    # 테이블이 명시되지 않았으면 BMS와 GPS 모두 검색
                    if spec_columns:
                        # 규격 파일(BMS 또는 GPS)에 있는 변수만 포함 (거리 관련 변수 필터링)
                        normalized_var_dist = normalize_variable_name(var_match)
                        mapped_var_dist = COLUMN_NAME_MAPPING.get(var_match, normalized_var_dist)
                        
                        is_in_spec_dist = (
                            var_match in spec_columns or
                            var_match.lower() in {c.lower() for c in spec_columns} or
                            normalized_var_dist in spec_columns or
                            mapped_var_dist in spec_columns or
                            mapped_var_dist.lower() in {c.lower() for c in spec_columns}
                        )
                        
                        # 유사도 기반 매칭도 시도
                        if not is_in_spec_dist:
                            matched_column = find_matching_column(var_match, spec_columns)
                            if matched_column is not None:
                                is_in_spec_dist = True
                        
                        # pack_voltage -> pack_volt 특수 매칭
                        if not is_in_spec_dist:
                            if var_lower in ['pack_voltage', 'packvoltage'] or normalized_var_dist in ['pack_voltage', 'packvoltage']:
                                if 'pack_volt' in spec_columns:
                                    is_in_spec_dist = True
                        
                        # 규격 파일에 없으면 제외
                        if not is_in_spec_dist:
                            continue
                        
                        # 거리와 직접 관련된 변수만 포함 (필터링 완화)
                        # BMS: odometer, pack_volt, pack_current, emobility_spd
                        # GPS: lat, lon, speed 등도 거리 관련일 수 있음
                        # 하지만 너무 제한적으로 필터링하면 모든 변수가 제외될 수 있으므로
                        # 규격 파일에 있고 거리와 무관한 변수만 제외하는 방식으로 변경
                        # (cumulative* 변수는 이미 위에서 제외됨)
                
                # [중요] BMS 규격 파일 검증: 규격 파일에 정의되지 않은 변수는 제외
                # 예: p_kw는 BMS 규격 파일에 없으므로 제외해야 함
                # 단, "p_kw를 구하는 방법" 같은 질문에서는 p_kw 자체를 제외하지 않음
                is_p_kw_calculation_query = any(keyword in question_lower for keyword in ['p_kw', 'pKw', '전력.*계산', '전력.*구하는'])
                
                if spec_columns and inferred_table_type.lower() == "bms" and not is_p_kw_calculation_query:
                    normalized_var_check2 = normalize_variable_name(var_match)
                    mapped_var_check2 = COLUMN_NAME_MAPPING.get(var_match, normalized_var_check2)
                    
                    # 규격 파일에 있는지 확인
                    is_in_spec2 = (
                        var_match in spec_columns or
                        var_match.lower() in {c.lower() for c in spec_columns} or
                        normalized_var_check2 in spec_columns or
                        mapped_var_check2 in spec_columns or
                        mapped_var_check2.lower() in {c.lower() for c in spec_columns}
                    )
                    
                    # 유사도 기반 매칭도 시도
                    if not is_in_spec2:
                        matched_column = find_matching_column(var_match, spec_columns)
                        if matched_column is not None:
                            is_in_spec2 = True
                        if not is_in_spec2 and mapped_var_check2 != normalized_var_check2:
                            matched_column = find_matching_column(mapped_var_check2, spec_columns)
                            if matched_column is not None:
                                is_in_spec2 = True
                    
                    # 특수 케이스: pack_voltage -> pack_volt 매칭
                    if not is_in_spec2:
                        if var_match.lower() in ['pack_voltage', 'packvoltage'] or normalized_var_check2 in ['pack_voltage', 'packvoltage']:
                            if 'pack_volt' in spec_columns:
                                is_in_spec2 = True
                    
                    # 규격 파일에 없으면 제외 (BMS 테이블에 정의되지 않은 변수)
                    if not is_in_spec2:
                        continue
                
                # 유사도 기반 매칭: find_matching_column 사용
                # 이 함수는 70% 이상 단어 겹침이 있으면 매칭됨
                matched_column = find_matching_column(var_match, actual_columns)
                
                if matched_column is None:
                    # 매칭 안됨 = 사용 안되는 변수 = 포함
                    filtered_vars.append(var_match)
                else:
                    # 매칭됨 = 사용되는 변수
                    # 하지만 seq는 특별 처리: "실제 사용하지 않음" 비고가 있으므로 포함
                    if var_match.lower() == 'seq':
                        filtered_vars.append(var_match)
                    # 나머지는 제외 (필터링됨)
            
            # 필터링된 변수명이 있으면 새 라인 생성
            if filtered_vars:
                if is_variable_list_line:
                    # 변수명 리스트 형태면 쉼표로 구분
                    # "BMS 테이블에 존재하지만 사용하지 않는 변수: " 형식 유지
                    if has_colon_format and '변수' in line_cleaned:
                        prefix_match = re.search(r'^([^:]+:\s*)', line)
                        if prefix_match:
                            new_line = prefix_match.group(1) + ', '.join(filtered_vars)
                        else:
                            new_line = ', '.join(filtered_vars)
                    elif has_bullet_format:
                        # 불릿 포인트 형식 유지
                        bullet_char = '*' if '*' in line else '-'
                        new_line = '\n'.join([f"{bullet_char}   {var}" for var in filtered_vars])
                    else:
                        new_line = ', '.join(filtered_vars)
                    filtered_lines.append(new_line)
                else:
                    # 설명이 있는 라인이면 원본 라인 유지 (일부 변수만 제외하는 것은 복잡하므로)
                    filtered_lines.append(line)
            # 모든 변수가 필터링되면 라인 제외
        else:
            # 변수명 리스트가 아니고 설명이 있는 라인도 BMS 규격 파일 검증 수행
            if spec_columns and inferred_table_type.lower() == "bms":
                # 라인에서 변수명 추출하고 규격 파일에 없는 것 제외
                line_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]+)\b', line)
                if len(line_vars) >= 2:  # 변수가 여러 개면 필터링
                    filtered_line_vars = []
                    for var in line_vars:
                        if len(var) <= 2:
                            continue
                        normalized_var = normalize_variable_name(var)
                        mapped_var = COLUMN_NAME_MAPPING.get(var, normalized_var)
                        is_in_spec = (
                            var in spec_columns or
                            var.lower() in {c.lower() for c in spec_columns} or
                            normalized_var in spec_columns or
                            mapped_var in spec_columns or
                            mapped_var.lower() in {c.lower() for c in spec_columns}
                        )
                        if not is_in_spec:
                            matched = find_matching_column(var, spec_columns)
                            if matched is not None:
                                is_in_spec = True
                        # pack_voltage -> pack_volt 특수 매칭
                        if not is_in_spec:
                            if var.lower() in ['pack_voltage', 'packvoltage'] or normalized_var in ['pack_voltage', 'packvoltage']:
                                if 'pack_volt' in spec_columns:
                                    is_in_spec = True
                        if is_in_spec:
                            filtered_line_vars.append(var)
                    # 규격 파일에 있는 변수만 포함된 라인으로 재구성
                    if filtered_line_vars:
                        filtered_lines.append(', '.join(filtered_line_vars))
                    # 모든 변수가 필터링되면 라인 제외
                else:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
    
    answer = '\n'.join(filtered_lines)
    
    # 빈 답변 보호: 필터링 후에도 내용이 있어야 함
    if not answer.strip():
        # 원본 답변에서 "실제 사용하지 않음" 비고가 있는 라인만 추출
        original_lines = original_answer.split('\n')
        not_used_lines = []
        for line in original_lines:
            if any(keyword in line for keyword in not_used_keywords):
                not_used_lines.append(line)
        if not_used_lines:
            answer = '\n'.join(not_used_lines)
        else:
            # "실제 사용하지 않음" 비고가 없으면 원본 답변 반환 (최소한의 필터링만)
            answer = original_answer
    
    # 앞뒤 공백 정리
    answer = answer.strip()
    
    # 빈 줄 정리 (연속된 빈 줄을 하나로)
    answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
    
    return answer

