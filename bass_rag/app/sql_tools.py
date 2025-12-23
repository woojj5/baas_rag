"""SQL generation tools for CSV and database analysis."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from app.config import Config

# InfluxDB imports
try:
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False


def csv_preview(path: str, n_rows: int = 5) -> Dict[str, Any]:
    """Preview CSV file and return schema information."""
    file_path = Path(path)
    
    if not file_path.exists():
        return {"error": f"File not found: {path}"}
    
    try:
        df = pd.read_csv(file_path, nrows=n_rows)
        
        schema = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            # Convert numpy types to Python native types for JSON serialization
            sample_values = [float(v) if isinstance(v, (int, float)) else str(v) for v in sample_values]
            
            schema.append({
                "column": col,
                "dtype": dtype,
                "sample_values": sample_values,
                "null_count": int(df[col].isna().sum())
            })
        
        # Get total rows efficiently
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        
        # Convert preview data to JSON-serializable format
        preview_data = []
        for _, row in df.head(n_rows).iterrows():
            preview_row = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    preview_row[col] = None
                elif isinstance(val, (int, float)):
                    preview_row[col] = float(val)
                else:
                    preview_row[col] = str(val)
            preview_data.append(preview_row)
        
        return {
            "file_path": str(file_path),
            "total_rows": total_rows,
            "columns": df.columns.tolist(),
            "schema": schema,
            "preview_data": preview_data
        }
    except Exception as e:
        return {"error": f"Error reading CSV: {str(e)}"}


def infer_baas_schema(schema_info: Dict[str, Any]) -> Dict[str, Any]:
    """Infer BAAS domain-specific schema from CSV preview."""
    columns = schema_info.get("columns", [])
    schema = schema_info.get("schema", [])
    
    inferred = {
        "key_columns": [],
        "time_columns": [],
        "baas_fields": {
            "soc": None,
            "soh": None,
            "odometer": None,
            "voltage": None,
            "temperature": None,
            "gps": []
        }
    }
    
    col_lower = {col.lower(): col for col in columns}
    
    # Key columns
    for key in ["device_no", "device_id", "vehicle_id", "vehicle_no", "id"]:
        if key in col_lower:
            inferred["key_columns"].append(col_lower[key])
    
    # Time columns
    for time_key in ["timestamp", "time", "date", "datetime", "created_at"]:
        if time_key in col_lower:
            inferred["time_columns"].append(col_lower[time_key])
    
    # BAAS fields
    for field, patterns in [
        ("soc", ["soc", "state_of_charge", "charge"]),
        ("soh", ["soh", "state_of_health", "health"]),
        ("odometer", ["odometer", "odo", "mileage", "distance"]),
        ("voltage", ["voltage", "volt", "pack_volt"]),
        ("temperature", ["temp", "temperature", "mod_temp"])
    ]:
        for pattern in patterns:
            # Check exact match first
            if pattern in col_lower:
                inferred["baas_fields"][field] = col_lower[pattern]
                break
            # Then check if pattern is contained in any column name
            for col in columns:
                if pattern in col.lower():
                    inferred["baas_fields"][field] = col
                    break
            if inferred["baas_fields"][field]:
                break
    
    # GPS fields
    for col in columns:
        col_l = col.lower()
        if any(gps_term in col_l for gps_term in ["gps", "lat", "lon", "longitude", "latitude", "location"]):
            inferred["baas_fields"]["gps"].append(col)
    
    return inferred


def generate_basic_exploration_sql(table_name: str, columns: List[str], limit: int = 10) -> str:
    """Generate basic exploration SQL queries."""
    sql_queries = []
    
    # SELECT * LIMIT
    sql_queries.append(f"-- 전체 데이터 샘플\nSELECT * FROM {table_name} LIMIT {limit};")
    
    # Column listing
    sql_queries.append(f"\n-- 컬럼 목록\nSELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
    
    # Row count
    sql_queries.append(f"\n-- 전체 행 수\nSELECT COUNT(*) as total_rows FROM {table_name};")
    
    return "\n".join(sql_queries)


def generate_basic_stats_sql(table_name: str, columns: List[str], schema: List[Dict]) -> str:
    """Generate basic statistics SQL queries."""
    sql_queries = []
    
    numeric_cols = []
    text_cols = []
    
    for col_info in schema:
        col = col_info["column"]
        dtype = col_info.get("dtype", "")
        if "int" in dtype or "float" in dtype or "numeric" in dtype:
            numeric_cols.append(col)
        else:
            text_cols.append(col)
    
    if numeric_cols:
        stats_list = []
        for col in numeric_cols[:5]:  # Limit to 5 columns
            stats_list.extend([
                f"COUNT({col}) as {col}_count",
                f"COUNT(DISTINCT {col}) as {col}_distinct",
                f"MIN({col}) as {col}_min",
                f"MAX({col}) as {col}_max",
                f"AVG({col}) as {col}_avg",
                f"STDDEV({col}) as {col}_stddev",
                f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as {col}_null_count"
            ])
        stats_cols = ", ".join(stats_list)
        sql_queries.append(f"-- 숫자형 컬럼 통계\nSELECT {stats_cols} FROM {table_name};")
    
    if text_cols:
        text_stats_list = []
        for col in text_cols[:5]:
            text_stats_list.extend([
                f"COUNT(DISTINCT {col}) as {col}_distinct",
                f"COUNT({col}) as {col}_count",
                f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as {col}_null_count"
            ])
        text_stats = ", ".join(text_stats_list)
        sql_queries.append(f"\n-- 텍스트형 컬럼 통계\nSELECT {text_stats} FROM {table_name};")
    
    return "\n".join(sql_queries) if sql_queries else "-- 통계 생성 불가 (스키마 정보 부족)"


def generate_baas_domain_sql(table_name: str, inferred: Dict[str, Any], time_col: Optional[str] = None) -> str:
    """Generate BAAS domain-specific SQL queries."""
    sql_queries = []
    baas = inferred["baas_fields"]
    
    # Monthly mileage (odometer delta)
    if baas["odometer"] and time_col:
        sql_queries.append(f"""-- 월별 주행거리 (Odometer 변화량)
SELECT 
    DATE_TRUNC('month', {time_col}) as month,
    device_no,
    MAX({baas["odometer"]}) - MIN({baas["odometer"]}) as monthly_mileage
FROM {table_name}
WHERE {baas["odometer"]} IS NOT NULL
GROUP BY DATE_TRUNC('month', {time_col}), device_no
ORDER BY month DESC, device_no;""")
    
    # SOC/SOH summary
    if baas["soc"]:
        sql_queries.append(f"""
-- SOC 요약 통계
SELECT 
    device_no,
    AVG({baas["soc"]}) as avg_soc,
    MIN({baas["soc"]}) as min_soc,
    MAX({baas["soc"]}) as max_soc,
    STDDEV({baas["soc"]}) as soc_stddev
FROM {table_name}
WHERE {baas["soc"]} IS NOT NULL
GROUP BY device_no;""")
    
    if baas["soh"]:
        sql_queries.append(f"""
-- SOH 요약 통계
SELECT 
    device_no,
    AVG({baas["soh"]}) as avg_soh,
    MIN({baas["soh"]}) as min_soh,
    MAX({baas["soh"]}) as max_soh
FROM {table_name}
WHERE {baas["soh"]} IS NOT NULL
GROUP BY device_no;""")
    
    # Voltage imbalance
    if baas["voltage"]:
        sql_queries.append(f"""
-- 전압 불균형 메트릭
SELECT 
    device_no,
    AVG({baas["voltage"]}) as avg_voltage,
    STDDEV({baas["voltage"]}) as voltage_stddev,
    MAX({baas["voltage"]}) - MIN({baas["voltage"]}) as voltage_range
FROM {table_name}
WHERE {baas["voltage"]} IS NOT NULL
GROUP BY device_no
HAVING STDDEV({baas["voltage"]}) > 0.1;""")
    
    # Temperature anomalies
    if baas["temperature"]:
        sql_queries.append(f"""
-- 온도 이상치 탐지
SELECT 
    device_no,
    {time_col or "timestamp"},
    {baas["temperature"]},
    CASE 
        WHEN {baas["temperature"]} > 45 THEN 'High'
        WHEN {baas["temperature"]} < 0 THEN 'Low'
        ELSE 'Normal'
    END as anomaly_type
FROM {table_name}
WHERE {baas["temperature"]} IS NOT NULL
    AND ({baas["temperature"]} > 45 OR {baas["temperature"]} < 0)
ORDER BY {time_col or "timestamp"} DESC;""")
    
    # GPS distance-based stats
    if len(baas["gps"]) >= 2:
        lat_col = baas["gps"][0]
        lon_col = baas["gps"][1] if len(baas["gps"]) > 1 else None
        
        if lon_col:
            sql_queries.append(f"""
-- GPS 기반 거리 통계 (예시: 첫 번째와 마지막 위치 간 거리)
WITH ranked AS (
    SELECT 
        device_no,
        {lat_col},
        {lon_col},
        {time_col or "timestamp"},
        ROW_NUMBER() OVER (PARTITION BY device_no ORDER BY {time_col or "timestamp"}) as rn_first,
        ROW_NUMBER() OVER (PARTITION BY device_no ORDER BY {time_col or "timestamp"} DESC) as rn_last
    FROM {table_name}
    WHERE {lat_col} IS NOT NULL AND {lon_col} IS NOT NULL
)
SELECT 
    r1.device_no,
    -- Haversine distance (km) - 간단한 근사치
    6371 * acos(
        cos(radians(r1.{lat_col})) * 
        cos(radians(r2.{lat_col})) * 
        cos(radians(r2.{lon_col}) - radians(r1.{lon_col})) + 
        sin(radians(r1.{lat_col})) * 
        sin(radians(r2.{lat_col}))
    ) as distance_km
FROM ranked r1
JOIN ranked r2 ON r1.device_no = r2.device_no AND r2.rn_last = 1
WHERE r1.rn_first = 1;""")
    
    return "\n".join(sql_queries) if sql_queries else "-- BAAS 도메인 SQL 생성 불가 (필수 필드 없음)"


def generate_sql_from_csv_preview(csv_path: str, table_name: str = "data_table") -> Dict[str, Any]:
    """Generate complete SQL analysis from CSV preview."""
    preview = csv_preview(csv_path, n_rows=10)
    
    if "error" in preview:
        return {"error": preview["error"]}
    
    columns = preview["columns"]
    schema = preview["schema"]
    
    inferred = infer_baas_schema(preview)
    time_col = inferred["time_columns"][0] if inferred["time_columns"] else None
    
    # Generate SQL
    exploration_sql = generate_basic_exploration_sql(table_name, columns)
    stats_sql = generate_basic_stats_sql(table_name, columns, schema)
    baas_sql = generate_baas_domain_sql(table_name, inferred, time_col)
    
    return {
        "schema_summary": {
            "columns": columns,
            "inferred": inferred,
            "schema_details": schema
        },
        "exploration_sql": exploration_sql,
        "stats_sql": stats_sql,
        "baas_sql": baas_sql
    }


def db_basic_stats(
    db_url: str,
    table_name: str,
    schema: Optional[str] = None
) -> Dict[str, Any]:
    """Get basic statistics from database table."""
    try:
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Get table columns
            inspector = inspect(engine)
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            columns = inspector.get_columns(table_name, schema=schema)
            
            column_info = []
            for col in columns:
                column_info.append({
                    "column": col["name"],
                    "dtype": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": str(col.get("default", ""))
                })
            
            # Get row count
            count_query = text(f'SELECT COUNT(*) as cnt FROM {full_table_name}')
            row_count = conn.execute(count_query).scalar()
            
            # Get sample data
            sample_query = text(f'SELECT * FROM {full_table_name} LIMIT 5')
            sample_df = pd.read_sql(sample_query, conn)
            
            # Get column names
            column_names = [col["name"] for col in columns]
            
            return {
                "table_name": table_name,
                "schema": schema,
                "columns": column_names,
                "column_info": column_info,
                "row_count": int(row_count),
                "sample_data": sample_df.to_dict(orient="records")
            }
    
    except SQLAlchemyError as e:
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


def generate_sql_from_db(
    db_url: str,
    table_name: str,
    schema: Optional[str] = None
) -> Dict[str, Any]:
    """Generate SQL from database table."""
    stats = db_basic_stats(db_url, table_name, schema)
    
    if "error" in stats:
        return {"error": stats["error"]}
    
    columns = stats["columns"]
    column_info = stats["column_info"]
    
    # Convert column_info to schema format for inference
    schema_list = []
    for col in column_info:
        schema_list.append({
            "column": col["column"],
            "dtype": col["dtype"],
            "sample_values": []
        })
    
    # Create preview-like structure for inference
    preview_like = {
        "columns": columns,
        "schema": schema_list
    }
    
    inferred = infer_baas_schema(preview_like)
    time_col = inferred["time_columns"][0] if inferred["time_columns"] else None
    
    full_table_name = f"{schema}.{table_name}" if schema else table_name
    
    # Generate SQL
    exploration_sql = generate_basic_exploration_sql(full_table_name, columns)
    stats_sql = generate_basic_stats_sql(full_table_name, columns, schema_list)
    baas_sql = generate_baas_domain_sql(full_table_name, inferred, time_col)
    
    return {
        "schema_summary": {
            "columns": columns,
            "inferred": inferred,
            "schema_details": schema_list,
            "row_count": stats["row_count"]
        },
        "exploration_sql": exploration_sql,
        "stats_sql": stats_sql,
        "baas_sql": baas_sql
    }


def parse_influxdb_url(url: str) -> Dict[str, str]:
    """Parse InfluxDB connection URL.
    
    Format: influxdb://[token@]host:port[/bucket]?org=org_name
    Example: influxdb://mytoken@localhost:8086/my_bucket?org=myorg
    """
    if not url.startswith('influxdb://'):
        raise ValueError("InfluxDB URL must start with 'influxdb://'")
    
    # Remove protocol
    url = url[11:]
    
    # Parse token, host, port, bucket, org
    parts = url.split('/')
    auth_host = parts[0]
    bucket = parts[1] if len(parts) > 1 else None
    
    # Parse org from query string
    org = None
    if bucket and '?' in bucket:
        bucket, query = bucket.split('?', 1)
        for param in query.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                if key == 'org':
                    org = value
    
    # Parse token and host:port
    if '@' in auth_host:
        token, host_port = auth_host.split('@', 1)
    else:
        token = None
        host_port = auth_host
    
    if ':' in host_port:
        host, port = host_port.split(':', 1)
    else:
        host = host_port
        port = "8086"
    
    return {
        "url": f"http://{host}:{port}",
        "token": token or "",
        "org": org or "",
        "bucket": bucket or ""
    }


def influxdb_basic_stats(
    url: str,
    bucket: str,
    measurement: Optional[str] = None,
    org: Optional[str] = None,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """Get basic statistics from InfluxDB bucket/measurement."""
    if not INFLUXDB_AVAILABLE:
        return {"error": "influxdb-client package not installed. Install with: pip install influxdb-client"}
    
    try:
        # Parse URL if it's a full connection string
        if url.startswith('influxdb://'):
            parsed = parse_influxdb_url(url)
            url = parsed["url"]
            token = token or parsed["token"]
            org = org or parsed["org"]
            bucket = bucket or parsed["bucket"]
        
        if not url or not bucket:
            return {"error": "InfluxDB URL and bucket are required"}
        
        # Create client
        client = InfluxDBClient(url=url, token=token, org=org)
        query_api = client.query_api()
        
        # Build Flux query
        if measurement:
            # Query specific measurement
            flux_query = f'''
            from(bucket: "{bucket}")
              |> range(start: -30d)
              |> filter(fn: (r) => r["_measurement"] == "{measurement}")
              |> limit(n: 1000)
            '''
        else:
            # Query all measurements in bucket
            flux_query = f'''
            from(bucket: "{bucket}")
              |> range(start: -30d)
              |> limit(n: 1000)
            '''
        
        # Execute query
        result = query_api.query(flux_query)
        
        # Process results
        measurements = set()
        fields = set()
        tags = set()
        sample_data = []
        row_count = 0
        
        for table in result:
            for record in table.records:
                row_count += 1
                measurements.add(record.get_measurement())
                
                # Collect fields and tags
                for key, value in record.values.items():
                    if key.startswith('_field'):
                        fields.add(record.get_field())
                    elif key not in ['_time', '_value', '_measurement', '_field', '_start', '_stop']:
                        tags.add(key)
                
                # Collect sample data (first 5 records)
                if len(sample_data) < 5:
                    sample_record = {
                        "_time": str(record.get_time()),
                        "_measurement": record.get_measurement(),
                        "_field": record.get_field(),
                        "_value": record.get_value()
                    }
                    # Add tags
                    for key, value in record.values.items():
                        if key not in ['_time', '_value', '_measurement', '_field', '_start', '_stop']:
                            sample_record[key] = value
                    sample_data.append(sample_record)
        
        # Get measurement list
        measurement_list = list(measurements)
        
        # If specific measurement requested, get its fields
        if measurement and measurement in measurement_list:
            measurement_fields_query = f'''
            from(bucket: "{bucket}")
              |> range(start: -30d)
              |> filter(fn: (r) => r["_measurement"] == "{measurement}")
              |> group(columns: ["_field"])
              |> distinct(column: "_field")
            '''
            fields_result = query_api.query(measurement_fields_query)
            field_list = []
            for table in fields_result:
                for record in table.records:
                    field_list.append(record.get_value())
        else:
            field_list = list(fields)
        
        client.close()
        
        return {
            "bucket": bucket,
            "measurement": measurement,
            "measurements": measurement_list,
            "fields": field_list,
            "tags": list(tags),
            "row_count": row_count,
            "sample_data": sample_data[:5]  # Limit to 5 samples
        }
    
    except Exception as e:
        return {"error": f"InfluxDB error: {str(e)}"}


def generate_flux_from_influxdb(
    url: str,
    bucket: str,
    measurement: Optional[str] = None,
    org: Optional[str] = None,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """Generate Flux queries from InfluxDB bucket/measurement."""
    stats = influxdb_basic_stats(url, bucket, measurement, org, token)
    
    if "error" in stats:
        return {"error": stats["error"]}
    
    measurement_name = measurement or (stats["measurements"][0] if stats["measurements"] else "measurement")
    fields = stats["fields"]
    tags = stats["tags"]
    
    # Generate Flux queries
    exploration_flux = f'''-- 전체 데이터 샘플 (최근 100개)
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> limit(n: 100)

-- 측정값(Measurement) 목록
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> group(columns: ["_measurement"])
  |> distinct(column: "_measurement")

-- 필드(Field) 목록
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> group(columns: ["_field"])
  |> distinct(column: "_field")

-- 데이터 포인트 수
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> count()'''
    
    stats_flux = ""
    if fields:
        stats_flux = f'''-- 필드별 통계 (평균, 최소, 최대)
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> group(columns: ["_field"])
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> group()
  |> aggregateWindow(every: 1h, fn: min, createEmpty: false)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)'''
    
    baas_flux = ""
    # Check if this looks like BAAS data
    baas_fields = {
        "soc": None,
        "soh": None,
        "odometer": None,
        "voltage": None,
        "temperature": None
    }
    
    field_lower = {f.lower(): f for f in fields}
    for field_key, patterns in [
        ("soc", ["soc", "state_of_charge", "charge"]),
        ("soh", ["soh", "state_of_health", "health"]),
        ("odometer", ["odometer", "odo", "mileage", "distance"]),
        ("voltage", ["voltage", "volt", "pack_volt"]),
        ("temperature", ["temp", "temperature", "mod_temp"])
    ]:
        for pattern in patterns:
            if pattern in field_lower:
                baas_fields[field_key] = field_lower[pattern]
                break
    
    if any(baas_fields.values()):
        baas_flux = f'''-- BAAS 도메인 특화 Flux 쿼리

-- SOC 평균 (시간별)
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> filter(fn: (r) => r["_field"] == "{baas_fields.get('soc', 'soc')}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)

-- SOH 추적
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> filter(fn: (r) => r["_field"] == "{baas_fields.get('soh', 'soh')}")
  |> aggregateWindow(every: 1d, fn: mean, createEmpty: false)

-- 전압 통계
from(bucket: "{bucket}")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "{measurement_name}")
  |> filter(fn: (r) => r["_field"] == "{baas_fields.get('voltage', 'voltage')}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> group()
  |> aggregateWindow(every: 1h, fn: min, createEmpty: false)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)'''
    
    return {
        "schema_summary": {
            "bucket": bucket,
            "measurement": measurement_name,
            "measurements": stats["measurements"],
            "fields": fields,
            "tags": tags,
            "row_count": stats["row_count"]
        },
        "exploration_flux": exploration_flux,
        "stats_flux": stats_flux,
        "baas_flux": baas_flux
    }

