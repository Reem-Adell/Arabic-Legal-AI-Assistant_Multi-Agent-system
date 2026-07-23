"""
main.py — FastAPI backend for the final project.

Endpoints
---------
GET  /api/health          liveness + which RAG mode is active
POST /api/chat             main multi-agent conversation endpoint
GET  /api/lawyers          list of lawyers (for the frontend directory panel)
GET  /api/appointments/{session_id}   appointments booked in this session

Run locally (mock mode, no GPU needed):
    LEGAL_RAG_MOCK=1 uvicorn main:app --reload --port 8000

Run in Colab with the real models + ngrok: see run_in_colab.md
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-ai")

import database as db
from seed_data import DEFAULT_LAWYERS
from rag_pipeline import pipeline, MOCK_MODE
import agents
from schemas import ChatRequest, ChatResponse, Lawyer, AppointmentOut

app = FastAPI(title="Arabic Legal AI Assistant — Multi-Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()
    db.seed_lawyers_if_empty(DEFAULT_LAWYERS)
    pipeline.load()


@app.get("/api/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE, "rag_ready": pipeline.ready}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    try:
        result = agents.handle_message(req.session_id, req.message, req.client_name)
    except Exception:
        logger.exception("chat handling failed for session=%s message=%r", req.session_id, req.message)
        raise HTTPException(500, "internal error — check server.log for the traceback")
    return result


@app.get("/api/lawyers", response_model=list[Lawyer])
def list_lawyers():
    return db.get_all_lawyers()


@app.get("/api/appointments/{session_id}", response_model=list[AppointmentOut])
def get_appointments(session_id: str):
    return db.get_appointments_for_session(session_id)


# ---------------------------------------------------------------------
# Serve the static HTML/CSS/JS frontend from the same FastAPI process
# so a single ngrok tunnel exposes both the API and the GUI.
# ---------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
