import os, sys, time, duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_engine import QueryEngine
from parser import parse
from classify import make_sampling_plan
from features import extract

PARQUET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DB", "data.parquet")

# --- Catalog ---

class TableStats:
    def __init__(self, con, table_name: str, row_count: int):
        self.con, self.table_name, self.row_count = con, table_name, row_count

    def histogram_frequency(self, col: str, val: Any) -> float:
        try:
            res = self.con.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE {col} = ?", (val,)).fetchone()
            return (res[0] if res else 0) / max(self.row_count, 1)
        except:
            return 0.1

    def histogram_range_fraction(self, col: str, val: Any) -> float:
        return 0.3

    def column_variance(self, cols: List[str]) -> float:
        if not cols: return 0.0
        try:
            res = self.con.execute(f"SELECT VAR_SAMP({cols[0]}) FROM {self.table_name}").fetchone()
            return float(res[0]) if res and res[0] is not None else 0.0
        except:
            return 1.0

    def ndistinct(self, cols: List[str]) -> int:
        if not cols: return 1
        try:
            res = self.con.execute(f"SELECT COUNT(DISTINCT {', '.join(cols)}) FROM {self.table_name}").fetchone()
            return res[0] if res else 1
        except:
            return 100

class CatalogClient:
    def __init__(self, con):
        self.con = con

    def get_stats(self, table_name: str) -> TableStats:
        safe = f"'{table_name}'" if ".parquet" in table_name.lower() or "/" in table_name else table_name
        try:
            res = self.con.execute(f"SELECT COUNT(*) FROM {safe}").fetchone()
            return TableStats(self.con, safe, res[0] if res else 1_000_000)
        except:
            return TableStats(self.con, safe, 1_000_000)

# --- App setup ---

app = FastAPI(title="Quail AQE API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

con = duckdb.connect(database=":memory:")
catalog = CatalogClient(con)
engine = QueryEngine(con)

@app.on_event("startup")
async def startup_event():
    if os.path.exists(PARQUET_PATH):
        con.execute(f"CREATE VIEW data AS SELECT * FROM '{PARQUET_PATH}'")
        print(f"VIEW 'data' -> {PARQUET_PATH}")
    else:
        con.execute("CREATE TABLE data (id INTEGER, val DOUBLE)")
        print(f"Parquet not found, created empty 'data' table.")

# --- Models ---

class QueryRequest(BaseModel):
    query: str
    confidence_level: Optional[float] = 0.95

# --- Routes ---

@app.get("/")
def root():
    return {"message": "Quail AQE API", "status": "ready"}

@app.post("/query/direct")
def query_direct(req: QueryRequest):
    try:
        t = time.time()
        return {"response": engine.execute_exact(req.query), "duration_s": time.time() - t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query/approximation")
def query_approximation(req: QueryRequest):
    try:
        t = time.time()
        ir = parse(req.query)
        plan = make_sampling_plan(*extract(ir, catalog), ir, hash_key="InvoiceNo")
        print(f"[AQE] Strategy: {plan.strategy} | Reason: {plan.reason} | SampleRate: {plan.sample_rate}")
        resp = engine.execute_approx(req.query, plan=plan, confidence_level=req.confidence_level)
        print(f"[AQE] executed_sql: {resp.executed_sql}")
        print(f"[AQE] warnings: {resp.warnings}")
        return {"response": resp, "duration_s": time.time() - t}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
