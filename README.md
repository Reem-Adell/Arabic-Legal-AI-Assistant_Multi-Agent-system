# Arabic Legal AI Assistant — Multi-Agent Final Project

Builds on the midterm "Arabic Legal AI Assistant" (hybrid BM25 + FAISS
RAG over Egyptian Family Law & Labor Law) and turns it into a full
multi-agent system with a web GUI, a FastAPI backend, appointment
booking, and an external-API integration — per the Final Project
(Week 6) brief.

## What's new vs. the midterm project

| Requirement (brief) | Implementation |
|---|---|
| Data retrieval from PDFs | Reused midterm hybrid-RAG pipeline (`backend/rag_pipeline.py`, `backend/midterm_parsing.py`) |
| LLM processing & summarization | Qwen2.5-7B-Instruct, unchanged prompt/format from the midterm project |
| Automated actions | Appointment booking is written to a SQLite database and shown back to the user |
| Workflow orchestration / multi-agent | `backend/agents.py` — `LegalQAAgent`, `SpecializationAgent`, `SchedulingAgent`, coordinated by a conversation-state-machine `Orchestrator` |
| GUI | `frontend/` — plain HTML/CSS/JS chat interface + appointment/lawyer panels |
| Backend | FastAPI (`backend/main.py`) |
| Deployment for demo | ngrok (see `run_in_colab.md`) |
| External API integration | `backend/external_api.py` — free Nager.Date public-holidays API, used so the booking agent never proposes an appointment on an Egyptian public holiday |
| Testing & evaluation | `backend/agents.py` logic was exercised end-to-end locally in mock mode (see below) before wiring in the real models |

## How the appointment flow works

1. User asks a legal question in the chat.
2. `LegalQAAgent` answers it via the same hybrid RAG pipeline as the
   midterm project, and the Legal-Index routing result tells the
   orchestrator which law was matched.
3. If a law was matched, `SpecializationAgent` maps it to a lawyer
   specialization (`قانون الأحوال الشخصية` → `أحوال شخصية`,
   `قانون العمل` → `قانون العمل`) and offers to book an appointment.
4. If the user agrees, they're asked for a preferred date/time.
5. `SchedulingAgent`:
   - checks the SQLite `appointments` table for a free lawyer with
     that specialization at that exact slot,
   - if none is free, searches forward (skipping weekends and, when
     reachable, Egyptian public holidays via the external API) for
     the nearest open slot across all lawyers with that
     specialization, and proposes it,
   - on confirmation, books the appointment (`UNIQUE(lawyer_id, date,
     time)` at the DB level prevents double-booking even under
     concurrent requests).
6. The booked appointment (lawyer name, specialization, date, time)
   is returned to the frontend and rendered in a dedicated
   "تفاصيل الموعد" panel.

## Project layout

```
backend/
  main.py            FastAPI app, routes, static frontend mount
  agents.py           Multi-agent orchestration + conversation state machine
  rag_pipeline.py      Midterm RAG pipeline, repackaged as a class (+ mock mode)
  midterm_parsing.py    Article-level PDF chunking (unchanged from midterm)
  database.py          SQLite: lawyers, appointments
  external_api.py       Public-holidays API integration
  schemas.py            Pydantic request/response models
  seed_data.py            Default lawyer roster
  requirements.txt
frontend/
  index.html / style.css / script.js   Chat GUI + side panels
run_in_colab.md      Step-by-step Colab + ngrok deployment
```

## Quick start (no GPU — frontend/backend/agents only)

```bash
cd backend
pip install fastapi uvicorn pydantic requests pyngrok
LEGAL_RAG_MOCK=1 uvicorn main:app --reload --port 8000
```
Open http://127.0.0.1:8000 — the RAG answers are placeholders labelled
`[MOCK MODE]`, but routing, the booking agent, the database, and the
whole GUI are fully live. This was used to test the booking/alt-slot
logic (double-booking prevention, nearest-slot search) before running
the real models.

## Full run with real models

See `run_in_colab.md` — needs a Colab GPU runtime, the midterm PDFs +
`Indexsheet.csv`, and a free ngrok authtoken.

## Notes / design decisions

- Sessions are kept in memory (`agents.SESSIONS`) for simplicity; a
  production version would persist conversation state per-user.
- The RAG pipeline's `answer_query()` keeps the midterm project's rule
  of returning the fixed refusal sentence when retrieved passages
  don't support an answer, and never invents legal information.
- Explicit failure over silent fallback is kept from the midterm
  project's principles: a missing `Indexsheet.csv` raises
  `FileNotFoundError` rather than silently using hardcoded data, and
  MOCK mode is an explicit, logged switch rather than an implicit one.
