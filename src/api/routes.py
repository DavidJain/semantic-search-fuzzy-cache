from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

engine = None
cache = None

class QueryRequest(BaseModel):
    query: str

def initialize(query_engine, semantic_cache):
    global engine, cache
    engine = query_engine
    cache = semantic_cache

@router.post("/query")
def query_api(request: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="System initializing, please wait.")
    return engine.process_query(request.query)

@router.get("/cache/stats")
def cache_stats():
    if cache is None:
        return {"message": "Cache not initialized"}
    return cache.stats()

@router.delete("/cache")
def clear_cache():
    if cache is not None:
        cache.clear()
    return {"message": "Cache cleared"}