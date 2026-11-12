"""HTTP layer. Thin on purpose: all logic lives in backend/pipeline.py."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import generator
from .pipeline import RAGPipeline

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

app = FastAPI(title="Energy Operations Document RAG Assistant")
pipeline = RAGPipeline()  # builds chunks + TF-IDF index once at start-up (milliseconds)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=6)


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(pipeline.chunks), "llm_enabled": generator.llm_enabled()}


@app.get("/api/demo")
def demo():
    return {"case_study": "Energy Operations Document RAG Assistant", "status": "running",
            "chunks": len(pipeline.chunks), "llm_enabled": generator.llm_enabled(),
            "try": "POST /api/ask {\"question\": \"What should be checked for inverter derating?\"}"}


@app.get("/api/sources")
def sources():
    return [{"chunk_id": c.id, "source": c.source, "title": c.title, "text": c.text} for c in pipeline.chunks]


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return pipeline.ask(req.question, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
