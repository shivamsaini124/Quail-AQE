import duckdb

def query_duckdb(query: str, parquet_path: str):
    query = query.replace("data", f"'{parquet_path}'")
    
    try:
        return duckdb.connect().execute(query).fetchdf()
    except Exception as e:
        return f"Error: {e}"