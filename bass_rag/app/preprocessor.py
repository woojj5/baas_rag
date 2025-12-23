"""Data preprocessing for BMS and GPS CSV files."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from app.config import Config


def load_cartype_mapping() -> Dict[str, str]:
    """Load device_no to car_type mapping from rules."""
    cartype_path = Config.ROOT / "rules" / "aicar_cartype_list.csv"
    
    if not cartype_path.exists():
        return {}
    
    try:
        df = pd.read_csv(cartype_path)
        return dict(zip(df["device_no"].astype(str), df["car_type"]))
    except Exception:
        return {}


def parse_timestamp(time_str: str, msg_time_str: Optional[str] = None) -> Optional[datetime]:
    """Parse timestamp from various formats."""
    if pd.isna(time_str) or time_str == "":
        return None
    
    time_str = str(time_str).strip()
    
    # Try different formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except:
            continue
    
    # If msg_time is provided, try that
    if msg_time_str:
        for fmt in formats:
            try:
                return datetime.strptime(str(msg_time_str).strip(), fmt)
            except:
                continue
    
    return None


def preprocess_bms_file(file_path: Path, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Preprocess BMS CSV file."""
    # Read pipe-delimited CSV
    df = pd.read_csv(
        file_path,
        sep="|",
        skipinitialspace=True,
        low_memory=False
    )
    
    # Remove metadata rows (rows with "(N rows)" pattern or separator lines starting with "-")
    def is_metadata_row(row):
        """Check if row is a metadata row."""
        # Convert row to string and check for patterns
        row_str = ' '.join(str(val) for val in row.values if pd.notna(val))
        # Check for "(N rows)" pattern
        if re.search(r'\(\d+\s+rows?\)', row_str, re.IGNORECASE):
            return True
        # Check for separator lines (mostly dashes and plus signs)
        if re.match(r'^[\s\-+]+$', row_str):
            return True
        return False
    
    # Filter out metadata rows
    mask = ~df.apply(is_metadata_row, axis=1)
    df = df[mask].reset_index(drop=True)
    
    # Remove empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract device_no and measured_month from filename if not in data
    filename = file_path.stem
    match = re.search(r"bms\.(\d+)\.(\d{4}-\d{2})", filename)
    if match:
        device_no_from_file = match.group(1)
        measured_month_from_file = match.group(2)
        
        if "device_no" not in df.columns or df["device_no"].isna().all():
            df["device_no"] = device_no_from_file
        
        if "measured_month" not in df.columns:
            df["measured_month"] = measured_month_from_file
    
    # Normalize timestamp
    if "time" in df.columns and "msg_time" in df.columns:
        df["timestamp"] = df.apply(
            lambda row: parse_timestamp(row.get("time"), row.get("msg_time")),
            axis=1
        )
    elif "time" in df.columns:
        df["timestamp"] = df["time"].apply(lambda x: parse_timestamp(x))
    elif "msg_time" in df.columns:
        df["timestamp"] = df["msg_time"].apply(lambda x: parse_timestamp(x))
    
    # Convert odometer to numeric
    if "odometer" in df.columns:
        df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    
    # Parse cell_volt_list if exists
    if "cell_volt_list" in df.columns:
        def parse_cell_volts(cell_str):
            if pd.isna(cell_str):
                return None
            try:
                cells = [float(x.strip()) for x in str(cell_str).split(",")]
                return cells
            except:
                return None
        
        cell_volts = df["cell_volt_list"].apply(parse_cell_volts)
        max_cells = max((len(c) for c in cell_volts.dropna() if isinstance(c, list)), default=0)
        
        if max_cells > 0:
            # Build all cell columns at once to avoid fragmentation
            cell_cols = {}
            for i in range(1, min(max_cells + 1, 193)):
                col_name = f"cell_v_{i:03d}"
                if col_name not in df.columns:
                    cell_cols[col_name] = cell_volts.apply(
                        lambda x: x[i-1] if x and len(x) >= i else None
                    )
            if cell_cols:
                df = pd.concat([df, pd.DataFrame(cell_cols)], axis=1)
    
    # Parse mod_temp_list if exists
    if "mod_temp_list" in df.columns:
        def parse_mod_temps(temp_str):
            if pd.isna(temp_str):
                return None
            try:
                temps = [float(x.strip()) for x in str(temp_str).split(",")]
                return temps
            except:
                return None
        
        mod_temps = df["mod_temp_list"].apply(parse_mod_temps)
        max_mods = max((len(t) for t in mod_temps.dropna() if isinstance(t, list)), default=0)
        
        if max_mods > 0:
            # Build all mod temp columns at once to avoid fragmentation
            mod_cols = {}
            for i in range(1, min(max_mods + 1, 19)):
                col_name = f"mod_temp_{i:02d}"
                if col_name not in df.columns:
                    mod_cols[col_name] = mod_temps.apply(
                        lambda x: x[i-1] if x and len(x) >= i else None
                    )
            if mod_cols:
                df = pd.concat([df, pd.DataFrame(mod_cols)], axis=1)
    
    # Add car_type from mapping
    cartype_map = load_cartype_mapping()
    if "device_no" in df.columns:
        df["car_type"] = df["device_no"].astype(str).map(cartype_map)
    
    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Save preprocessed file
    if output_dir is None:
        output_dir = Config.ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"preprocessed_{file_path.name}"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    stats = {
        "original_rows": len(df),
        "columns": len(df.columns),
        "device_no": df["device_no"].iloc[0] if "device_no" in df.columns else None,
        "date_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None
        }
    }
    
    return df, stats


def preprocess_gps_file(file_path: Path, output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Preprocess GPS CSV file."""
    # Read pipe-delimited CSV
    df = pd.read_csv(
        file_path,
        sep="|",
        skipinitialspace=True,
        low_memory=False
    )
    
    # Remove metadata rows (rows with "(N rows)" pattern or separator lines starting with "-")
    def is_metadata_row(row):
        """Check if row is a metadata row."""
        # Convert row to string and check for patterns
        row_str = ' '.join(str(val) for val in row.values if pd.notna(val))
        # Check for "(N rows)" pattern
        if re.search(r'\(\d+\s+rows?\)', row_str, re.IGNORECASE):
            return True
        # Check for separator lines (mostly dashes and plus signs)
        if re.match(r'^[\s\-+]+$', row_str):
            return True
        return False
    
    # Filter out metadata rows
    mask = ~df.apply(is_metadata_row, axis=1)
    df = df[mask].reset_index(drop=True)
    
    # Remove empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Extract device_no from filename if not in data
    filename = file_path.stem
    match = re.search(r"gps\.(\d+)\.(\d{4}-\d{2})", filename)
    if match:
        device_no_from_file = match.group(1)
        measured_month_from_file = match.group(2)
        
        if "device_no" not in df.columns or df["device_no"].isna().all():
            df["device_no"] = device_no_from_file
        
        if "measured_month" not in df.columns:
            df["measured_month"] = measured_month_from_file
    
    # Normalize timestamp
    if "time" in df.columns:
        df["timestamp"] = df["time"].apply(lambda x: parse_timestamp(x))
    
    # Rename GPS columns to standard names
    if "lng" in df.columns:
        df["lon"] = df["lng"]
    
    # Convert numeric columns
    numeric_cols = ["lat", "lon", "speed", "hdop", "fuel_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add car_type from mapping
    cartype_map = load_cartype_mapping()
    if "device_no" in df.columns:
        df["car_type"] = df["device_no"].astype(str).map(cartype_map)
    
    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Save preprocessed file
    if output_dir is None:
        output_dir = Config.ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"preprocessed_{file_path.name}"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    stats = {
        "original_rows": len(df),
        "columns": len(df.columns),
        "device_no": df["device_no"].iloc[0] if "device_no" in df.columns else None,
        "date_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None
        }
    }
    
    return df, stats


def preprocess_all_files(input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, any]:
    """Preprocess all files in before_preprocess directory."""
    if input_dir is None:
        input_dir = Config.ROOT / "before_preprocess"
    
    if output_dir is None:
        output_dir = Config.ROOT / "data"
    
    results = {
        "processed": [],
        "errors": []
    }
    
    for file_path in input_dir.glob("*.csv"):
        try:
            if file_path.name.startswith("bms."):
                df, stats = preprocess_bms_file(file_path, output_dir)
                results["processed"].append({
                    "file": file_path.name,
                    "type": "BMS",
                    "stats": stats
                })
            elif file_path.name.startswith("gps."):
                df, stats = preprocess_gps_file(file_path, output_dir)
                results["processed"].append({
                    "file": file_path.name,
                    "type": "GPS",
                    "stats": stats
                })
        except Exception as e:
            results["errors"].append({
                "file": file_path.name,
                "error": str(e)
            })
    
    return results

