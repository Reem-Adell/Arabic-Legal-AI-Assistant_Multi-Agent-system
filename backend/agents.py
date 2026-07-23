"""
agents.py

Implements the multi-agent orchestration required by the final-project
brief. Each agent has a single responsibility; the Orchestrator drives
a small conversation state machine per session and calls agents in turn.

Agents
------
1. LegalQAAgent        — wraps the midterm RAG pipeline (rag_pipeline.py)
2. SpecializationAgent — maps a routed law -> the lawyer specialization
                         the office needs to offer
3. SchedulingAgent      — availability checks, alternative-slot search,
                         booking, using database.py + external_api.py
                         (external_api.py is the required external-API
                         integration: Egyptian public holidays)
4. Orchestrator         — the conversation state machine tying it together
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, List

import database as db
from external_api import is_office_closed
from rag_pipeline import pipeline, RagAnswer

WORK_START = time(9, 0)
WORK_END = time(17, 0)
SLOT_MINUTES = 60
MAX_SEARCH_DAYS = 21

YES_WORDS = {"نعم", "ايوه", "أيوه", "اه", "آه", "تمام", "موافق", "اوك", "ok", "yes", "أكيد", "اكيد", "yep"}
NO_WORDS = {"لا", "لأ", "مش عايز", "no", "مش موافق", "الغاء", "إلغاء"}


def _norm(text: str) -> str:
    return text.strip().lower()


def is_affirmative(text: str) -> bool:
    t = _norm(text)
    return any(w in t for w in YES_WORDS)


def is_negative(text: str) -> bool:
    t = _norm(text)
    return any(w in t for w in NO_WORDS)


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def parse_datetime_from_text(text: str) -> Optional[datetime]:
    """Accepts formats like '2025-08-20 14:00', '20/8/2025 2pm', or
    Arabic-numeral variants. Returns None if nothing parseable is found
    (explicit failure -> caller re-prompts, no silent guessing)."""
    cleaned = text.translate(ARABIC_DIGITS)
    date_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{4})", cleaned)
    time_match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm|ص|م)?|(\d{1,2})\s*(am|pm|ص|م)", cleaned)
    if not date_match:
        return None
    d_str = date_match.group(1)
    d_fmt = "%Y-%m-%d" if "-" in d_str else "%d/%m/%Y"
    try:
        d = datetime.strptime(d_str, d_fmt).date()
    except ValueError:
        return None

    hour, minute = 10, 0  # default appointment time if only a date was given
    if time_match:
        if time_match.group(1):
            hour, minute = int(time_match.group(1)), int(time_match.group(2))
            suffix = time_match.group(3)
        else:
            hour, minute = int(time_match.group(4)), 0
            suffix = time_match.group(5)
        if suffix in ("pm", "م") and hour < 12:
            hour += 12
        if suffix in ("am", "ص") and hour == 12:
            hour = 0
    return datetime.combine(d, time(hour, minute))


@dataclass
class SessionState:
    stage: str = "idle"  # idle | await_booking_confirm | await_datetime | await_alt_confirm
    specialization: Optional[str] = None
    routed_law: Optional[str] = None
    last_question: Optional[str] = None
    client_name: str = "عميل"
    proposed_slot: Optional[datetime] = None
    proposed_lawyer_id: Optional[int] = None
    history: List[Dict[str, str]] = field(default_factory=list)


SESSIONS: Dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState()
    return SESSIONS[session_id]


# ---------------------------------------------------------------------
# Agent 1: Legal QA (wraps the RAG pipeline)
# ---------------------------------------------------------------------
class LegalQAAgent:
    def answer(self, query: str, history_text: str) -> RagAnswer:
        return pipeline.answer_query(query, history_text)


# ---------------------------------------------------------------------
# Agent 2: Specialization mapping is already embedded in RagAnswer via
# rag_pipeline.LAW_TO_SPECIALIZATION; this agent just formats the ask.
# ---------------------------------------------------------------------
class SpecializationAgent:
    def offer_text(self, specialization: str) -> str:
        return (f"يبدو أن استشارتك تحتاج محاميًا متخصصًا في \"{specialization}\". "
                f"هل تريد حجز موعد مع أحد محامينا المتخصصين؟ (نعم / لا)")


# ---------------------------------------------------------------------
# Agent 3: Scheduling / booking, with the external public-holidays API
# ---------------------------------------------------------------------
class SchedulingAgent:
    def _slot_ok(self, lawyer_id: int, dt: datetime) -> bool:
        if is_office_closed(dt.date()):
            return False
        if not (WORK_START <= dt.time() < WORK_END):
            return False
        return not db.is_slot_taken(lawyer_id, dt.date().isoformat(), dt.strftime("%H:%M"))

    def check_and_maybe_find_alternative(self, specialization: str, requested: datetime):
        """Returns (lawyer_dict_or_None, exact_match: bool, alt_dt_or_None)."""
        lawyers = db.get_lawyers_by_specialization(specialization)
        if not lawyers:
            return None, False, None

        for lw in lawyers:
            if self._slot_ok(lw["id"], requested):
                return lw, True, None

        # search forward for the nearest slot across all matching lawyers
        cursor = requested
        end_search = requested + timedelta(days=MAX_SEARCH_DAYS)
        while cursor <= end_search:
            if WORK_START <= cursor.time() < WORK_END and not is_office_closed(cursor.date()):
                for lw in lawyers:
                    if self._slot_ok(lw["id"], cursor):
                        return lw, False, cursor
            cursor += timedelta(minutes=SLOT_MINUTES)
            if cursor.time() >= WORK_END:
                nxt_day = cursor.date() + timedelta(days=1)
                cursor = datetime.combine(nxt_day, WORK_START)
        return None, False, None

    def book(self, lawyer_id: int, client_name: str, dt: datetime, session_id: str) -> Dict:
        return db.book_appointment(lawyer_id, client_name, dt.date().isoformat(),
                                    dt.strftime("%H:%M"), session_id)


qa_agent = LegalQAAgent()
spec_agent = SpecializationAgent()
sched_agent = SchedulingAgent()


# ---------------------------------------------------------------------
# Orchestrator: conversation state machine
# ---------------------------------------------------------------------
def handle_message(session_id: str, message: str, client_name: Optional[str] = None) -> Dict:
    s = get_session(session_id)
    if client_name:
        s.client_name = client_name
    s.history.append({"role": "user", "content": message})
    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in s.history[-6:])

    reply = ""
    appointment = None

    if s.stage == "idle":
        result = qa_agent.answer(message, history_text)
        reply = result.answer_text
        s.last_question = message
        if result.specialization:
            s.specialization = result.specialization
            s.routed_law = result.routed_law
            reply += "\n\n" + spec_agent.offer_text(result.specialization)
            s.stage = "await_booking_confirm"

    elif s.stage == "await_booking_confirm":
        if is_affirmative(message):
            reply = "تمام، من فضلك اكتب التاريخ والوقت المفضّل للموعد (مثال: 2025-08-20 14:00)."
            s.stage = "await_datetime"
        elif is_negative(message):
            reply = "تمام، لا مشكلة. تفضل بسؤالك القانوني التالي متى شئت."
            s.stage = "idle"
        else:
            # treat as a new legal question instead of stalling
            s.stage = "idle"
            return handle_message(session_id, message, client_name)

    elif s.stage == "await_datetime":
        dt = parse_datetime_from_text(message)
        if not dt:
            reply = "لم أفهم التاريخ/الوقت. من فضلك استخدم الصيغة: YYYY-MM-DD HH:MM (مثال: 2025-08-20 14:00)."
        else:
            lawyer, exact, alt_dt = sched_agent.check_and_maybe_find_alternative(s.specialization, dt)
            if lawyer is None:
                reply = "عذرًا، لا يوجد محامٍ متاح لهذا التخصص خلال الفترة القادمة. حاول لاحقًا من فضلك."
                s.stage = "idle"
            elif exact:
                appt = sched_agent.book(lawyer["id"], s.client_name, dt, session_id)
                appointment = appt
                reply = "تم حجز الموعد بنجاح! التفاصيل موضحة في قسم المواعيد."
                s.stage = "idle"
            else:
                s.proposed_slot = alt_dt
                s.proposed_lawyer_id = lawyer["id"]
                reply = (f"الموعد المطلوب غير متاح. أقرب موعد متاح مع الأستاذ/ة {lawyer['name']} "
                         f"هو {alt_dt.strftime('%Y-%m-%d')} الساعة {alt_dt.strftime('%H:%M')}. "
                         f"هل توافق على هذا الموعد؟ (نعم / لا)")
                s.stage = "await_alt_confirm"

    elif s.stage == "await_alt_confirm":
        if is_affirmative(message) and s.proposed_slot and s.proposed_lawyer_id:
            appt = sched_agent.book(s.proposed_lawyer_id, s.client_name, s.proposed_slot, session_id)
            appointment = appt
            reply = "تم حجز الموعد بنجاح! التفاصيل موضحة في قسم المواعيد."
            s.stage = "idle"
            s.proposed_slot, s.proposed_lawyer_id = None, None
        elif is_negative(message):
            reply = "تمام، من فضلك اكتب تاريخًا ووقتًا آخر تفضّله (مثال: 2025-08-20 14:00)."
            s.stage = "await_datetime"
        else:
            reply = "من فضلك أجب بـ (نعم) للموافقة على الموعد المقترح أو (لا) لاقتراح وقت آخر."

    s.history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "stage": s.stage, "appointment": appointment}
