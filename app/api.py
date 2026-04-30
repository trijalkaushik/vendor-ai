from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from app.core.loader import load_data
from app.core.search import search, search_raw, extract_intent, generate_answer, MODEL

app = FastAPI(title="Vendor AI")

# Load data once at startup
df = load_data()

# Session state (simple in-memory, single user)
last_candidates = {"data": None}

class QueryRequest(BaseModel):
    query: str
    is_followup: bool = False

FOLLOW_UP_TRIGGERS = ["same", "that", "their", "its", "those", "these", "them"]

def _is_followup(query: str) -> bool:
    words = query.lower().split()
    return any(t in words for t in FOLLOW_UP_TRIGGERS)

@app.post("/api/chat")
def chat(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return {"answer": "", "rows": []}

    followup = _is_followup(query) and last_candidates["data"] is not None

    if followup:
        candidates = last_candidates["data"]
        intent = extract_intent(query)
        answer = generate_answer(query, candidates, intent)
    else:
        candidates = search_raw(df, query)
        last_candidates["data"] = candidates
        answer = search(df, query)

    # Convert candidates to JSON-safe list
    rows = []
    if candidates is not None and not candidates.empty:
        display_cols = [c for c in candidates.columns if not c.startswith("_")]
        rows = candidates[display_cols].head(10).fillna("").to_dict(orient="records")

    return {"answer": answer, "rows": rows}

@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL, "rows_loaded": len(df)}

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)