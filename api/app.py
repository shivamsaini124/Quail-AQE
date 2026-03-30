from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI backend is running"}

@app.post("/query/direct")
def query_direct(query: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/query/approximation")
def query_with_approximation(query: str):
    raise HTTPException(status_code=501, detail="Not implemented")
