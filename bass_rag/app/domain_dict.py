"""Domain dictionary for EV Battery/BAAS domain-specific terms and mappings."""
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import re

from app.config import Config


class DomainDictionary:
    """Domain-specific dictionary for EV Battery/BAAS terminology."""
    
    def __init__(self):
        self.variable_to_info: Dict[str, Dict[str, str]] = {}  # variable_name -> {table, description, unit, etc.}
        self.table_to_variables: Dict[str, Set[str]] = {}  # table -> set of variables
        self.description_to_variables: Dict[str, Set[str]] = {}  # description keywords -> variables
        self.domain_terms: Dict[str, List[str]] = {}  # term -> synonyms/expansions
        self.variable_aliases: Dict[str, Set[str]] = {}  # variable -> aliases (lowercase)
        
    def build_from_rules(self):
        """Build domain dictionary from rules files."""
        rules_dir = Config.ROOT / "rules"
        
        if not rules_dir.exists():
            return
        
        # Load Excel files
        for file_path in rules_dir.glob("*.xlsx"):
            try:
                all_sheets = pd.read_excel(file_path, sheet_name=None)
                
                for sheet_name, sheet_df in all_sheets.items():
                    if sheet_df.empty:
                        continue
                    
                    # Detect header
                    first_row = sheet_df.iloc[0]
                    header_keywords = ['순서', '변수명', '설명', '단위', '범위', '호출주기', '비고', '테이블구분']
                    first_row_str = ' '.join([str(v) for v in first_row.values if pd.notna(v)]).lower()
                    
                    if any(keyword in first_row_str for keyword in header_keywords):
                        sheet_df.columns = [str(col).strip() if pd.notna(col) else f'Column_{i}' for i, col in enumerate(first_row.values)]
                        sheet_df = sheet_df.iloc[1:].reset_index(drop=True)
                    
                    # Extract variable information
                    for idx, row in sheet_df.iterrows():
                        if row.isna().all():
                            continue
                        
                        var_name = None
                        table = None
                        description = None
                        unit = None
                        note = None
                        
                        # Find columns
                        for col_name in row.index:
                            col_clean = str(col_name).strip().lower()
                            value = row[col_name]
                            
                            if pd.isna(value):
                                continue
                            
                            value_str = str(value).strip()
                            
                            if '변수명' in col_clean or 'variable' in col_clean or 'field' in col_clean:
                                var_name = value_str
                            elif '테이블구분' in col_clean or 'table' in col_clean:
                                table = value_str
                            elif '설명' in col_clean or 'description' in col_clean:
                                description = value_str
                            elif '단위' in col_clean or 'unit' in col_clean:
                                unit = value_str
                            elif '비고' in col_clean or 'note' in col_clean or 'remark' in col_clean:
                                note = value_str
                        
                        if var_name:
                            # Normalize table name (GPS_텔레매틱스 -> GPS)
                            # Handle both original and uppercase versions
                            table_upper = table.upper() if table else ""
                            table_lower = table.lower() if table else ""
                            
                            # Check if it's GPS_텔레매틱스 (case-insensitive)
                            if table and ("GPS" in table_upper or "GPS" in table_lower) and ("텔레매틱스" in table or "텔레매틱스" in table_upper or "텔레매틱스" in table_lower):
                                normalized_table = "GPS"
                            elif table:
                                # Keep original table name but uppercase for consistency
                                normalized_table = table_upper
                            else:
                                normalized_table = ""
                            
                            # Store variable info
                            self.variable_to_info[var_name] = {
                                "table": normalized_table or (table or ""),
                                "description": description or "",
                                "unit": unit or "",
                                "note": note or ""
                            }
                            
                            # Build table -> variables mapping
                            if normalized_table:
                                if normalized_table not in self.table_to_variables:
                                    self.table_to_variables[normalized_table] = set()
                                self.table_to_variables[normalized_table].add(var_name)
                            # Also add to original table if different (for backward compatibility)
                            if table and table != normalized_table and table.upper() != normalized_table:
                                if table not in self.table_to_variables:
                                    self.table_to_variables[table] = set()
                                self.table_to_variables[table].add(var_name)
                            
                            # Build description -> variables mapping (extract keywords)
                            if description:
                                keywords = self._extract_keywords(description)
                                for keyword in keywords:
                                    if keyword not in self.description_to_variables:
                                        self.description_to_variables[keyword] = set()
                                    self.description_to_variables[keyword].add(var_name)
                            
                            # Build aliases (lowercase variations)
                            var_lower = var_name.lower()
                            if var_name not in self.variable_aliases:
                                self.variable_aliases[var_name] = set()
                            self.variable_aliases[var_name].add(var_lower)
                            # Add variations (remove underscores, spaces, etc.)
                            var_clean = re.sub(r'[_\s-]', '', var_lower)
                            if var_clean != var_lower:
                                self.variable_aliases[var_name].add(var_clean)
            
            except Exception as e:
                print(f"Warning: Could not process Excel file {file_path} for domain dict: {e}")
                continue
        
        # Build domain term expansions
        self._build_domain_terms()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from description text."""
        # Remove common stop words (Korean)
        stop_words = {'의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', '와', '과', '및', '등', '또한', '또는'}
        
        # Split by common delimiters
        words = re.split(r'[,\s\.\-\/\(\)]+', text.lower())
        
        # Filter meaningful words (length >= 2, not stop words)
        keywords = [w for w in words if len(w) >= 2 and w not in stop_words]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _build_domain_terms(self):
        """Build domain-specific term expansions."""
        # BAAS domain terms
        self.domain_terms = {
            "soc": ["state of charge", "배터리 충전율", "충전 상태", "soc", "충전량"],
            "soh": ["state of health", "배터리 건강도", "건강 상태", "soh", "수명"],
            "pack_volt": ["pack voltage", "팩 전압", "전압", "voltage", "volt"],
            "cell_volt": ["cell voltage", "셀 전압", "전압"],
            "mod_temp": ["module temperature", "모듈 온도", "온도", "temperature", "temp"],
            "odo_km": ["odometer", "주행거리", "누적 주행거리", "mileage", "km"],
            "bms": ["battery management system", "배터리 관리 시스템", "bms"],
            "gps": ["global positioning system", "위치 정보", "gps", "위치"]
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with domain-specific terms and variable aliases."""
        expansions = [query]  # Original query
        
        query_lower = query.lower()
        
        # Domain term expansion
        for term, synonyms in self.domain_terms.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym != term:
                        expanded = query.replace(term, synonym)
                        if expanded not in expansions:
                            expansions.append(expanded)
        
        # Variable name expansion (if query contains variable-like terms)
        for var_name, aliases in self.variable_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    # Add full variable name
                    if var_name not in query:
                        expanded = query.replace(alias, var_name)
                        if expanded not in expansions:
                            expansions.append(expanded)
                    # Add description if available
                    var_info = self.variable_to_info.get(var_name, {})
                    if var_info.get("description"):
                        desc_keywords = self._extract_keywords(var_info["description"])
                        for keyword in desc_keywords[:2]:  # Top 2 keywords
                            if keyword not in query_lower:
                                expanded = f"{query} {keyword}"
                                if expanded not in expansions:
                                    expansions.append(expanded)
        
        return expansions[:5]  # Limit to 5 expansions
    
    def find_exact_matches(self, query: str) -> List[Tuple[str, str]]:
        """Find exact variable name or table matches in query.
        Returns: List of (variable_name, match_type) tuples.
        """
        matches = []
        query_lower = query.lower()
        
        # Check variable names (exact match or alias)
        for var_name, aliases in self.variable_aliases.items():
            if var_name.lower() in query_lower:
                matches.append((var_name, "variable_exact"))
            else:
                for alias in aliases:
                    if alias in query_lower:
                        matches.append((var_name, "variable_alias"))
                        break
        
        # Check table names
        for table in self.table_to_variables.keys():
            if table.lower() in query_lower:
                matches.append((table, "table"))
        
        return matches
    
    def filter_by_table(self, passages: List[str], table_filter: str) -> List[str]:
        """Filter passages that contain variables from specific table."""
        if not table_filter or table_filter not in self.table_to_variables:
            return passages
        
        table_vars = self.table_to_variables[table_filter]
        filtered = []
        
        for passage in passages:
            # Check if passage contains any variable from the table
            for var_name in table_vars:
                if var_name in passage or var_name.lower() in passage.lower():
                    filtered.append(passage)
                    break
        
        return filtered
    
    def prioritize_rules_passages(self, passages: List[str], passage_indices: List[int]) -> List[int]:
        """Prioritize passages from rules directory.
        Returns: Reordered list of indices (rules passages first).
        """
        rules_indices = []
        other_indices = []
        
        for idx in passage_indices:
            if 0 <= idx < len(passages):
                passage = passages[idx]
                # Check if passage is from rules (contains rules metadata or rules/ path)
                if "rules/" in passage or "규칙/필드정의" in passage or "[메타데이터: 타입: 규칙" in passage:
                    rules_indices.append(idx)
                else:
                    other_indices.append(idx)
        
        # Rules passages first, then others
        return rules_indices + other_indices


# Global domain dictionary instance
_domain_dict: DomainDictionary | None = None


def get_domain_dict() -> DomainDictionary:
    """Get or create global domain dictionary instance."""
    global _domain_dict
    if _domain_dict is None:
        _domain_dict = DomainDictionary()
        _domain_dict.build_from_rules()
        # Clear caches when domain_dict is rebuilt
        try:
            from app.data_columns import clear_rules_variables_cache
            clear_rules_variables_cache()
        except ImportError:
            pass
    return _domain_dict

