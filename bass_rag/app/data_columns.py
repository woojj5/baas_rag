"""Utility to get actual column names from preprocessed CSV files (hardcoded)."""
from typing import Set, Dict, Optional, List, Tuple
import re
import pandas as pd
from pathlib import Path
from app.config import Config
from functools import lru_cache

# 변수명 매핑 (다양한 표기법을 표준 형식으로)
COLUMN_NAME_MAPPING: Dict[str, str] = {
    "hvacList1": "hvac_list1",
    "hvacList2": "hvac_list2",
    "moduleAvgTemp": "mod_avg_temp",
    "moduleMinTemp": "mod_min_temp",  # 추가
    "moduleMaxTemp": "mod_max_temp",  # 추가
    "moduleTempList": "mod_temp_list",  # 추가
    "emobilitySpeed": "emobility_spd",
    "maxDeteriorationCellNo": "max_deter_cell_no",
    "maxCellVoltage": "max_cell_volt",  # 추가
    "maxCellVoltageNo": "max_cell_volt_no",  # 추가
    "minCellVoltage": "min_cell_volt",  # 추가
    "minCellVoltageNo": "min_cell_volt_no",  # 추가
    "minDeterioratation": "min_deter",
    "minDeteriorationCellNo": "min_deter_cell_no",
    "cellVoltageList": "cell_volt_list",
    "insulatedResistance": "insul_resistance",
    "cellVoltageDispersion": "cell_volt_dispersion",
    "driveMotorSpd1": "drive_motor_spd1",
    "driveMotorSpd2": "drive_motor_spd2",
    "airbagHWireDuty": "airbag_hwire_duty",
    "airbagHwireDuty": "airbag_hwire_duty",  # 추가 (대소문자 변형)
    "battInternalTemp": "batt_internal_temp",  # 추가
    "subBattVoltage": "sub_batt_volt",  # 추가
    "cumulativeCurrentCharged": "cumul_current_chrgd",  # 추가
    "cumulativeCurrentDischarged": "cumul_current_dischrgd",  # 추가
    "cumulativePowerCharged": "cumul_pw_chrgd",  # 추가
    "cumulativePowerDischarged": "cumul_pw_dischrgd",  # 추가
    "deviceNo": "device_no",  # GPS/BMS 공통
    "DEVICE_NO": "device_no",  # GPS/BMS 공통
    "messageTime": "msg_time",  # BMS
    "msgTime": "msg_time",  # BMS
    # 필요시 추가 매핑 가능
}


# 하드코딩된 BMS 테이블 칼럼 목록 (실제 전처리된 파일에서 추출)
BMS_COLUMNS = {
    "device_no", "measured_month", "msg_time", "time", "acceptable_chrg_pw",
    "acceptable_dischrg_pw", "airbag_hwire_duty", "batt_coolant_inlet_temp",
    "batt_fan_running", "batt_internal_temp", "batt_ltr_rear_temp",
    "batt_pra_busbar_temp", "batt_pw", "bms_running", "cell_volt_dispersion",
    "cell_volt_list", "chrg_cable_conn", "chrg_cnt", "chrg_cnt_q",
    "cumul_current_chrgd", "cumul_current_dischrgd", "cumul_energy_chrgd",
    "cumul_energy_chrgd_q", "cumul_pw_chrgd", "cumul_pw_dischrgd",
    "drive_motor_spd1", "drive_motor_spd2", "emobility_spd", "est_chrg_time",
    "ext_temp", "fast_chrg_port_conn", "fast_chrg_relay_on", "hvac_list1",
    "hvac_list2", "insul_resistance", "int_temp", "inverter_capacity_volt",
    "main_relay_conn", "max_cell_volt", "max_cell_volt_no", "max_deter_cell_no",
    "min_cell_volt", "min_cell_volt_no", "min_deter", "min_deter_cell_no",
    "mod_avg_temp", "mod_max_temp", "mod_min_temp", "mod_temp_list", "msg_id",
    "odometer", "op_time", "pack_current", "pack_volt", "seq",
    "slow_chrg_port_conn", "soc", "socd", "soh", "start_time", "sub_batt_volt",
    "trip_chrg_pw", "trip_dischrg_pw", "v2l", "timestamp",
    # cell_v_001 ~ cell_v_192 (최대 192개)
    "cell_v_001", "cell_v_002", "cell_v_003", "cell_v_004", "cell_v_005",
    "cell_v_006", "cell_v_007", "cell_v_008", "cell_v_009", "cell_v_010",
    "cell_v_011", "cell_v_012", "cell_v_013", "cell_v_014", "cell_v_015",
    "cell_v_016", "cell_v_017", "cell_v_018", "cell_v_019", "cell_v_020",
    "cell_v_021", "cell_v_022", "cell_v_023", "cell_v_024", "cell_v_025",
    "cell_v_026", "cell_v_027", "cell_v_028", "cell_v_029", "cell_v_030",
    "cell_v_031", "cell_v_032", "cell_v_033", "cell_v_034", "cell_v_035",
    "cell_v_036", "cell_v_037", "cell_v_038", "cell_v_039", "cell_v_040",
    "cell_v_041", "cell_v_042", "cell_v_043", "cell_v_044", "cell_v_045",
    "cell_v_046", "cell_v_047", "cell_v_048", "cell_v_049", "cell_v_050",
    "cell_v_051", "cell_v_052", "cell_v_053", "cell_v_054", "cell_v_055",
    "cell_v_056", "cell_v_057", "cell_v_058", "cell_v_059", "cell_v_060",
    "cell_v_061", "cell_v_062", "cell_v_063", "cell_v_064", "cell_v_065",
    "cell_v_066", "cell_v_067", "cell_v_068", "cell_v_069", "cell_v_070",
    "cell_v_071", "cell_v_072", "cell_v_073", "cell_v_074", "cell_v_075",
    "cell_v_076", "cell_v_077", "cell_v_078", "cell_v_079", "cell_v_080",
    "cell_v_081", "cell_v_082", "cell_v_083", "cell_v_084", "cell_v_085",
    "cell_v_086", "cell_v_087", "cell_v_088", "cell_v_089", "cell_v_090",
    "cell_v_091", "cell_v_092", "cell_v_093", "cell_v_094", "cell_v_095",
    "cell_v_096", "cell_v_097", "cell_v_098", "cell_v_099", "cell_v_100",
    "cell_v_101", "cell_v_102", "cell_v_103", "cell_v_104", "cell_v_105",
    "cell_v_106", "cell_v_107", "cell_v_108", "cell_v_109", "cell_v_110",
    "cell_v_111", "cell_v_112", "cell_v_113", "cell_v_114", "cell_v_115",
    "cell_v_116", "cell_v_117", "cell_v_118", "cell_v_119", "cell_v_120",
    "cell_v_121", "cell_v_122", "cell_v_123", "cell_v_124", "cell_v_125",
    "cell_v_126", "cell_v_127", "cell_v_128", "cell_v_129", "cell_v_130",
    "cell_v_131", "cell_v_132", "cell_v_133", "cell_v_134", "cell_v_135",
    "cell_v_136", "cell_v_137", "cell_v_138", "cell_v_139", "cell_v_140",
    "cell_v_141", "cell_v_142", "cell_v_143", "cell_v_144", "cell_v_145",
    "cell_v_146", "cell_v_147", "cell_v_148", "cell_v_149", "cell_v_150",
    "cell_v_151", "cell_v_152", "cell_v_153", "cell_v_154", "cell_v_155",
    "cell_v_156", "cell_v_157", "cell_v_158", "cell_v_159", "cell_v_160",
    "cell_v_161", "cell_v_162", "cell_v_163", "cell_v_164", "cell_v_165",
    "cell_v_166", "cell_v_167", "cell_v_168", "cell_v_169", "cell_v_170",
    "cell_v_171", "cell_v_172", "cell_v_173", "cell_v_174", "cell_v_175",
    "cell_v_176", "cell_v_177", "cell_v_178", "cell_v_179", "cell_v_180",
    "cell_v_181", "cell_v_182", "cell_v_183", "cell_v_184", "cell_v_185",
    "cell_v_186", "cell_v_187", "cell_v_188", "cell_v_189", "cell_v_190",
    "cell_v_191", "cell_v_192",
    # mod_temp_01 ~ mod_temp_18 (최대 18개)
    "mod_temp_01", "mod_temp_02", "mod_temp_03", "mod_temp_04", "mod_temp_05",
    "mod_temp_06", "mod_temp_07", "mod_temp_08", "mod_temp_09", "mod_temp_10",
    "mod_temp_11", "mod_temp_12", "mod_temp_13", "mod_temp_14", "mod_temp_15",
    "mod_temp_16", "mod_temp_17", "mod_temp_18",
    "car_type"
}

# 하드코딩된 GPS 테이블 칼럼 목록 (실제 전처리된 파일에서 추출)
GPS_COLUMNS = {
    "device_no", "time", "direction", "fuel_pct", "hdop", "lat", "lng",
    "mode", "source", "speed", "state", "measured_month", "timestamp",
    "lon", "car_type"
}


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.
    Example: cellVoltageList -> cell_voltage_list
    """
    # Insert underscore before uppercase letters (except first)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters that follow lowercase
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def normalize_variable_name(var_name: str) -> str:
    """
    Normalize variable name to a canonical form for comparison.
    - Convert camelCase to snake_case
    - Lowercase everything
    - Remove underscores at start/end
    - Handle uppercase rules Excel variables (DEVICE_NO -> device_no)
    
    Example: cellVoltageList -> cell_voltage_list
    Example: DEVICE_NO -> device_no
    """
    # First check explicit mapping
    if var_name in COLUMN_NAME_MAPPING:
        return COLUMN_NAME_MAPPING[var_name]
    
    # Handle uppercase variables from rules Excel (e.g., DEVICE_NO, SPEED, LAT, LNG)
    if var_name.isupper() and '_' in var_name:
        # Already in snake_case format, just lowercase
        return var_name.lower()
    
    # Convert camelCase to snake_case
    normalized = camel_to_snake(var_name)
    return normalized.strip('_')


def find_matching_column(var_name: str, available_columns: Set[str]) -> Optional[str]:
    """
    Find matching column in available_columns even if names don't match exactly.
    
    Matching rules:
    1. Exact match (case-insensitive)
    2. Normalized match (camelCase -> snake_case)
    3. Partial word match (if core words match)
    
    Args:
        var_name: Variable name to find (e.g., "cellVoltageList")
        available_columns: Set of available column names (e.g., {"cell_volt_list", ...})
        
    Returns:
        Matching column name if found, None otherwise
    """
    var_lower = var_name.lower()
    var_normalized = normalize_variable_name(var_name)
    
    # 1. Exact match (case-insensitive)
    for col in available_columns:
        if col.lower() == var_lower:
            return col
    
    # 2. Normalized match
    for col in available_columns:
        col_normalized = normalize_variable_name(col)
        if col_normalized == var_normalized:
            return col
    
    # 3. Extract core words and match
    # Remove common suffixes/prefixes and numbers
    var_words = set(re.findall(r'[a-z]+', var_normalized))
    var_words.discard('list')  # Remove common words
    var_words.discard('no')
    var_words.discard('num')
    
    if len(var_words) >= 2:  # Need at least 2 meaningful words
        best_match = None
        best_score = 0
        
        for col in available_columns:
            col_normalized = normalize_variable_name(col)
            col_words = set(re.findall(r'[a-z]+', col_normalized))
            col_words.discard('list')
            col_words.discard('no')
            col_words.discard('num')
            
            # Calculate word overlap
            overlap = len(var_words & col_words)
            total_unique = len(var_words | col_words)
            if total_unique > 0:
                score = overlap / total_unique
                # Require at least 70% word overlap
                if score >= 0.7 and score > best_score:
                    best_score = score
                    best_match = col
        
        if best_match:
            return best_match
    
    return None


def normalize_column_name(column_name: str) -> str:
    """
    Normalize column name using mapping (e.g., hvacList1 -> hvac_list1).
    
    Args:
        column_name: Original column name
        
    Returns:
        Normalized column name
    """
    return COLUMN_NAME_MAPPING.get(column_name, column_name)


def get_actual_used_columns(table_type: str = "bms") -> dict:
    """
    Get actual column names (hardcoded + specification files).
    Specification files (bms_specification.csv, gps_specification.csv) are the definitive source.
    If a variable exists in specification files, it's considered "used" even if not in rules Excel.
    
    Args:
        table_type: "bms" or "gps"
        
    Returns:
        Dictionary with a single key "columns" containing the column set
    """
    # Start with specification file columns (definitive source)
    spec_columns = load_specification_columns(table_type)
    
    if table_type.lower() == "bms":
        # Include both hardcoded and specification columns
        columns = BMS_COLUMNS.copy() | spec_columns
        # Add mapped variants (e.g., hvacList1 -> hvac_list1 is already in BMS_COLUMNS)
        # But also add reverse mapping for lookup
        for mapped_name, standard_name in COLUMN_NAME_MAPPING.items():
            if standard_name in columns:
                columns.add(mapped_name)  # Add both forms for lookup
        return {"columns": columns}
    elif table_type.lower() == "gps":
        # Include both hardcoded and specification columns
        columns = GPS_COLUMNS.copy() | spec_columns
        return {"columns": columns}
    else:
        return {"columns": set()}


def get_all_used_columns(table_type: str = "bms") -> Set[str]:
    """
    Get all column names for given table type (hardcoded).
    Includes both standard and mapped variants.
    
    Args:
        table_type: "bms" or "gps"
        
    Returns:
        Set of all column names (including mapped variants)
    """
    if table_type.lower() == "bms":
        columns = BMS_COLUMNS.copy()
        # Add mapped variants for lookup
        for mapped_name, standard_name in COLUMN_NAME_MAPPING.items():
            if standard_name in columns:
                columns.add(mapped_name)
        return columns
    elif table_type.lower() == "gps":
        return GPS_COLUMNS
    else:
        return set()


def format_columns_for_prompt(file_columns: dict) -> str:
    """
    Format column information for inclusion in RAG prompt (optimized - minimal info only).
    
    Args:
        file_columns: Dictionary with "columns" key containing column set
        
    Returns:
        Formatted string for prompt (simplified to reduce token count)
    """
    if not file_columns or "columns" not in file_columns:
        return ""
    
    columns = file_columns["columns"]
    if not columns:
        return ""
    
    # Extract core columns (exclude cell_v_* and mod_temp_* for brevity)
    core_columns = [c for c in sorted(columns) 
                    if not c.startswith("cell_v_") and not c.startswith("mod_temp_")]
    
    # Count cell and mod temp columns
    cell_volt_count = sum(1 for c in columns if c.startswith("cell_v_"))
    mod_temp_count = sum(1 for c in columns if c.startswith("mod_temp_"))
    
    # Get mapped column names that are in the list (for explicit display)
    mapped_names_in_list = []
    for mapped_name, standard_name in COLUMN_NAME_MAPPING.items():
        if standard_name in columns:
            mapped_names_in_list.append(mapped_name)
    
    # Determine table type from columns (check if GPS or BMS)
    is_gps = any(col in columns for col in ['device_no', 'direction', 'fuel_pct', 'hdop', 'lat', 'lon', 'speed', 'state', 'time', 'timestamp'])
    is_bms = any(col in columns for col in ['cell_v_001', 'mod_temp_01', 'pack_volt', 'soc', 'soh'])
    
    table_type_str = "GPS" if is_gps and not is_bms else "BMS" if is_bms else "BMS/GPS"
    
    lines = [f"=== {table_type_str} 칼럼 ==="]
    # Further reduced: 15 → 10 core columns
    lines.append(f"핵심: {', '.join(core_columns[:10])}")
    if len(core_columns) > 10:
        lines.append(f"... 외 {len(core_columns) - 10}개")
    
    # Simplified cell/mod temp info (one line)
    if cell_volt_count > 0 and mod_temp_count > 0:
        lines.append(f"셀전압: {cell_volt_count}개, 모듈온도: {mod_temp_count}개")
    elif cell_volt_count > 0:
        lines.append(f"셀전압: {cell_volt_count}개")
    elif mod_temp_count > 0:
        lines.append(f"모듈온도: {mod_temp_count}개")
    
    # Simplified list info (one line)
    has_cell_volt_list = "cell_volt_list" in columns
    has_mod_temp_list = "mod_temp_list" in columns
    if has_cell_volt_list and has_mod_temp_list:
        lines.append("cell_volt_list, mod_temp_list 있음 → 모두 사용됨")
    elif has_cell_volt_list:
        lines.append("cell_volt_list 있음 → cell_v_* 사용됨")
    elif has_mod_temp_list:
        lines.append("mod_temp_list 있음 → mod_temp_* 사용됨")
    
    # Simplified mapped names (reduced from 20 to 10)
    if mapped_names_in_list:
        mapped_display = sorted(mapped_names_in_list)[:10]
        lines.append(f"매핑: {', '.join(mapped_display)}")
        if len(mapped_names_in_list) > 10:
            lines.append(f"... 외 {len(mapped_names_in_list) - 10}개")
    
    # Simplified matching rules (one line)
    lines.append("매칭: camelCase↔snake_case 동일, 대소문자 무시")
    
    return "\n".join(lines)


@lru_cache(maxsize=2)
def load_specification_columns(table_type: str = "bms") -> Set[str]:
    """
    Load column names from specification CSV files (cached).
    
    Args:
        table_type: "bms" or "gps"
        
    Returns:
        Set of column names from specification file
    """
    spec_file = Config.ROOT / "rules" / f"{table_type}_specification.csv"
    
    if not spec_file.exists():
        return set()
    
    try:
        df = pd.read_csv(spec_file)
        if "column_name" in df.columns:
            return set(df["column_name"].astype(str).str.strip().unique())
        return set()
    except Exception as e:
        print(f"Warning: Could not load specification file {spec_file}: {e}")
        return set()


# Cache for rules variables (expensive to compute)
# Clear cache when domain_dict is rebuilt
_rules_variables_cache: Dict[str, Set[str]] = {}

def get_rules_variables(table_type: str = "bms", domain_dict=None) -> Set[str]:
    """
    Get variable names from rules Excel files (cached).
    
    Args:
        table_type: "bms" or "gps"
        domain_dict: Optional pre-built DomainDictionary (to avoid rebuilding)
        
    Returns:
        Set of variable names from rules documents
    """
    # Use cache if available
    cache_key = table_type.lower()
    if cache_key in _rules_variables_cache:
        return _rules_variables_cache[cache_key]
    
    # Use provided domain_dict or get from global
    if domain_dict is None:
        from app.domain_dict import get_domain_dict
        domain_dict = get_domain_dict()
    
    # Get variables for the specified table
    table_vars = domain_dict.table_to_variables.get(table_type.upper(), set())
    
    # Also get all variables (case-insensitive match, including GPS_텔레매틱스)
    all_vars = set()
    for var_name in domain_dict.variable_to_info.keys():
        var_table = domain_dict.variable_to_info[var_name].get("table", "")
        var_table_upper = var_table.upper() if var_table else ""
        # Match GPS (including GPS_텔레매틱스)
        if table_type.upper() == "GPS":
            # Check if it's GPS or contains GPS (handles GPS_텔레매틱스)
            if var_table_upper == "GPS" or (var_table and "GPS" in var_table_upper):
                all_vars.add(var_name)
        elif var_table_upper == table_type.upper():
            all_vars.add(var_name)
    
    result = table_vars | all_vars
    _rules_variables_cache[cache_key] = result
    return result


# Cache for missing variables check (expensive to compute)
_missing_variables_cache: Dict[str, Dict[str, Set[str]]] = {}

def check_missing_variables(table_type: str = "bms", domain_dict=None) -> Dict[str, Set[str]]:
    """
    Check which specification variables are missing from rules or actual CSV files (cached).
    
    Args:
        table_type: "bms" or "gps"
        domain_dict: Optional pre-built DomainDictionary (to avoid rebuilding)
        
    Returns:
        Dictionary with:
        - "specification": set of variables in specification
        - "missing_from_rules": set of variables in spec but not in rules
        - "missing_from_csv": set of variables in spec but not in actual CSV
        - "missing_from_both": set of variables in spec but not in rules or CSV
    """
    # Use cache if available
    cache_key = table_type.lower()
    if cache_key in _missing_variables_cache:
        return _missing_variables_cache[cache_key]
    
    spec_vars = load_specification_columns(table_type)
    rules_vars = get_rules_variables(table_type, domain_dict=domain_dict)
    csv_vars = get_actual_used_columns(table_type).get("columns", set())
    
    # Normalize variable names for comparison
    def normalize_for_comparison(var: str) -> str:
        # Convert to lowercase and handle common variations
        normalized = normalize_variable_name(var).lower()
        # Handle common mappings
        # lng <-> lon (treat as same)
        if normalized == "lng" or normalized == "lon":
            return "lon"
        # Handle uppercase rules Excel variables (DEVICE_NO -> device_no)
        # Already handled by normalize_variable_name, but ensure lowercase
        return normalized
    
    spec_normalized = {normalize_for_comparison(v): v for v in spec_vars}
    rules_normalized = {normalize_for_comparison(v): v for v in rules_vars}
    csv_normalized = {normalize_for_comparison(v): v for v in csv_vars}
    
    missing_from_rules = set()
    missing_from_csv = set()
    missing_from_both = set()
    
    for spec_norm, spec_orig in spec_normalized.items():
        in_rules = any(
            normalize_for_comparison(rv) == spec_norm or 
            find_matching_column(spec_orig, {rv}) is not None
            for rv in rules_vars
        )
        in_csv = any(
            normalize_for_comparison(cv) == spec_norm or 
            find_matching_column(spec_orig, {cv}) is not None
            for cv in csv_vars
        )
        
        if not in_rules:
            missing_from_rules.add(spec_orig)
        if not in_csv:
            missing_from_csv.add(spec_orig)
        if not in_rules and not in_csv:
            missing_from_both.add(spec_orig)
    
    result = {
        "specification": spec_vars,
        "missing_from_rules": missing_from_rules,
        "missing_from_csv": missing_from_csv,
        "missing_from_both": missing_from_both
    }
    
    # Cache the result
    _missing_variables_cache[cache_key] = result
    return result


def check_unused_variables(table_type: str = "bms", domain_dict=None) -> Dict[str, Set[str]]:
    """
    Check which rules Excel variables are not in specification files (cached).
    These are considered "unused" variables.
    
    Args:
        table_type: "bms" or "gps"
        domain_dict: Optional pre-built DomainDictionary (to avoid rebuilding)
        
    Returns:
        Dictionary with:
        - "rules_variables": set of variables in rules Excel
        - "spec_variables": set of variables in specification files
        - "unused_variables": set of variables in rules but not in spec (not similar)
    """
    rules_vars = get_rules_variables(table_type, domain_dict=domain_dict)
    spec_vars = load_specification_columns(table_type)
    
    # Normalize variable names for comparison
    def normalize_for_comparison(var: str) -> str:
        normalized = normalize_variable_name(var).lower()
        if normalized == "lng" or normalized == "lon":
            return "lon"
        return normalized
    
    spec_normalized = {normalize_for_comparison(v): v for v in spec_vars}
    rules_normalized = {normalize_for_comparison(v): v for v in rules_vars}
    
    unused_variables = set()
    
    for rules_norm, rules_orig in rules_normalized.items():
        # Check if similar variable exists in spec
        is_similar = False
        
        # Direct match
        if rules_norm in spec_normalized:
            is_similar = True
        else:
            # Check using find_matching_column for fuzzy matching
            matching = find_matching_column(rules_orig, spec_vars)
            if matching is not None:
                is_similar = True
        
        # If not similar, it's unused
        if not is_similar:
            unused_variables.add(rules_orig)
    
    return {
        "rules_variables": rules_vars,
        "spec_variables": spec_vars,
        "unused_variables": unused_variables
    }


def format_unused_variables_info(table_type: str = "bms", domain_dict=None) -> str:
    """
    Format information about unused variables (in rules Excel but not in specification files).
    
    Args:
        table_type: "bms" or "gps"
        domain_dict: Optional pre-built DomainDictionary (to avoid rebuilding)
        
    Returns:
        Formatted string about unused variables
    """
    unused_info = check_unused_variables(table_type, domain_dict=domain_dict)
    
    if not unused_info["unused_variables"]:
        return ""
    
    lines = [f"\n⚠️⚠️⚠️ {table_type.upper()} 테이블 - rules Excel에 있지만 규격 파일에는 없는 변수들 (사용하지 않는 변수) ⚠️⚠️⚠️:"]
    lines.append("다음 변수들은 rules 문서(20230822 아이카 데이터 필드)에는 있지만,")
    lines.append(f"규격 파일({table_type}_specification.csv)에는 없거나 비슷한 변수도 없습니다:")
    lines.append("")
    
    unused_vars = sorted(unused_info["unused_variables"])
    for var in unused_vars[:30]:  # Limit to first 30
        lines.append(f"  - {var}")
    
    if len(unused_vars) > 30:
        lines.append(f"  ... 외 {len(unused_vars) - 30}개")
    
    lines.append("")
    lines.append("⚠️ 중요: 위 변수들은 rules Excel에는 있지만 규격 파일에는 없는 변수입니다.")
    lines.append("이 변수들은 '사용하지 않는 변수'로 분류할 수 있습니다.")
    
    return "\n".join(lines)


def format_missing_variables_info(table_type: str = "bms", domain_dict=None) -> str:
    """
    Format information about missing variables for prompt (cached).
    
    Args:
        table_type: "bms" or "gps"
        domain_dict: Optional pre-built DomainDictionary (to avoid rebuilding)
        
    Returns:
        Formatted string about missing variables
    """
    missing_info = check_missing_variables(table_type, domain_dict=domain_dict)
    
    if not missing_info["missing_from_both"]:
        return ""
    
    lines = ["\n⚠️⚠️⚠️ 규격 파일에 있지만 실제로는 없는 변수들 ⚠️⚠️⚠️:"]
    lines.append("다음 변수들은 규격 파일(bms_specification.csv 또는 gps_specification.csv)에 정의되어 있지만,")
    lines.append("rules 문서(20230822 아이카 데이터 필드)와 실제 전처리된 CSV 파일 모두에 없습니다:")
    lines.append("")
    
    missing_vars = sorted(missing_info["missing_from_both"])
    for var in missing_vars[:20]:  # Limit to first 20
        lines.append(f"  - {var}")
    
    if len(missing_vars) > 20:
        lines.append(f"  ... 외 {len(missing_vars) - 20}개")
    
    lines.append("")
    lines.append("⚠️ 중요: 위 변수들은 규격에는 있지만 실제로는 사용되지 않는 변수입니다.")
    lines.append("질문에 답할 때 이 변수들이 없다고 명시하거나, '사용 여부가 확실하지 않은' 변수로 분류하세요.")
    
    return "\n".join(lines)

