from pydantic import BaseModel
from typing import Optional, Dict, List


class ChatRequest(BaseModel):
    session_id: str
    message: str
    client_name: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    stage: str
    appointment: Optional[Dict] = None


class Lawyer(BaseModel):
    id: int
    name: str
    specialization: str


class AppointmentOut(BaseModel):
    id: int
    lawyer_name: str
    lawyer_specialization: str
    appointment_date: str
    appointment_time: str
    client_name: str
    status: str
