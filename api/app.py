import os
import sys
import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import time
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_engine import QueryEngine, QueryResponse
from parser import parse
from classify import make_sampling_plan, SamplingPlan, Strategy
from features import extract, Phase1Features, Phase2Features

# --- Simple CatalogClient for DuckDB ---
class TableStats:
    def __init__(self, con: duckdb.DuckDBPyConnection, table_name: str, row_count: int):
        self.con = con
        self.table_name = table_name
        self.row_count = row_count

    def histogram_frequency(self, col: str, val: Any) -> float:
        try:
            res = self.con.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE {col} = ?", (val,)).fetchone()
            count = res[0] if res else 0
            return count / max(self.row_count, 1)
        except: return 0.1

    def histogram_range_fraction(self, col: str, val: Any) -> float:
        # Simplified range estimate fallback
        return 0.3

    def column_variance(self, cols: List[str]) -> float:
        # Simplified: check for variance in the first group-by column if any
        if not cols: return 0.0
        try:
            col = cols[0]
            res = self.con.execute(f"SELECT VAR_SAMP({col}) FROM {self.table_name}").fetchone()
            return float(res[0]) if res and res[0] is not None else 0.0
        except: return 1.0

    def ndistinct(self, cols: List[str]) -> int:
        if not cols: return 1
        try:
            col_str = ", ".join(cols)
            res = self.con.execute(f"SELECT COUNT(DISTINCT {col_str}) FROM {self.table_name}").fetchone()
            return res[0] if res else 1
        except: return 100

class CatalogClient:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def get_stats(self, table_name: str) -> TableStats:
        # Quote paths if necessary so duckdb can parse them correctly
        safe_table = f"'{table_name}'" if '.parquet' in table_name.lower() or '/' in table_name else table_name
        try:
            res = self.con.execute(f"SELECT COUNT(*) FROM {safe_table}").fetchone()
            row_count = res[0] if res else 1000000
            return TableStats(self.con, safe_table, row_count)
        except:
            return TableStats(self.con, safe_table, 1000000)

app = FastAPI(title="Quail AQE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup database and engine
PARQUET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DB", "data.parquet")
con = duckdb.connect(database=":memory:")
catalog = CatalogClient(con)
engine = QueryEngine(con)

@app.on_event("startup")
async def startup_event():
    # Create a VIEW so queries against 'data' run directly against the Parquet file (no memory load)
    if os.path.exists(PARQUET_PATH):
        try:
            con.execute(f"CREATE VIEW data AS SELECT * FROM '{PARQUET_PATH}'")
            print(f"✅ Created VIEW 'data' mapped natively to {PARQUET_PATH}")
        except Exception as e:
            print(f"❌ Failed to create view for data: {e}")
    else:
        # Dummy table for testing if file missing
        con.execute("CREATE TABLE data (id INTEGER, val DOUBLE)")
        print(f"⚠️ Parquet not found at {PARQUET_PATH}, created empty 'data' table.")

class QueryRequest(BaseModel):
    query: str
    confidence_level: Optional[float] = 0.95

@app.get("/")
def root():
    return {
        "message": "Quail Approximate Query Engine (AQE) Backend",
        "status": "ready"
    }

@app.post("/query/direct")
def query_direct(request: QueryRequest):
    """Executes a query directly without approximation."""
    try:
        start_time = time.time()
        resp = engine.execute_exact(request.query)
        duration = time.time() - start_time
        return {"response": resp, "duration_s": duration}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query/approximation")
def query_with_approximation(request: QueryRequest):
    """Executes a query with AQE (Approximate Query Execution)."""
    try:
        start_time = time.time()
        # 1. Parse and extract features
        ir = parse(request.query)
        p1, p2 = extract(ir, catalog)
        
        # 2. Derive plan
        plan = make_sampling_plan(p1, p2, ir, hash_key="InvoiceNo")
        
        # 3. Execute
        resp = engine.execute_approx(
            request.query, 
            plan=plan, 
            confidence_level=request.confidence_level
        )
        duration = time.time() - start_time
        return {"response": resp, "duration_s": duration}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

