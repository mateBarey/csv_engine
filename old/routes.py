from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List , Optional , Dict, Any , TypedDict, Union, Tuple
import asyncio
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Allow all CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryReq(BaseModel):

    fields: str
    limit: int
@app.get("/search")
def search(req: QueryReq):
    pass
    # FastAPI mini-service: GET /search?q=...&fields=title,body&limit=10 returning JSON; add /healthz.

    # pytest: unit tests for parser (malformed rows, multiline), diacritics folding (żółć → zolc), and query DSL.

    # performance: add a --profile flag that prints timings for parse vs. search; show top 3 hot functions (use cProfile, pstats).