"""
database.py
SQLite persistence layer for lawyers and appointments.

Design principle (kept consistent with the mid-term project):
explicit failure over silent fallback. If the DB file cannot be
created/opened, this raises instead of silently degrading.
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "legal_office.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS lawyers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialization TEXT NOT NULL   -- e.g. 'الأحوال الشخصية' or 'قانون العمل'
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lawyer_id INTEGER NOT NULL,
    client_name TEXT NOT NULL,
    appointment_date TEXT NOT NULL,   -- 'YYYY-MM-DD'
    appointment_time TEXT NOT NULL,   -- 'HH:MM' (24h)
    status TEXT NOT NULL DEFAULT 'confirmed',
    session_id TEXT,
    FOREIGN KEY (lawyer_id) REFERENCES lawyers(id),
    UNIQUE (lawyer_id, appointment_date, appointment_time)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def seed_lawyers_if_empty(lawyers: List[Dict[str, str]]):
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM lawyers").fetchone()["c"]
        if count == 0:
            conn.executemany(
                "INSERT INTO lawyers (name, specialization) VALUES (:name, :specialization)",
                lawyers,
            )


def get_lawyers_by_specialization(specialization: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM lawyers WHERE specialization = ?", (specialization,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_lawyers() -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM lawyers").fetchall()
        return [dict(r) for r in rows]


def is_slot_taken(lawyer_id: int, date: str, time: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM appointments
               WHERE lawyer_id = ? AND appointment_date = ? AND appointment_time = ?
                     AND status = 'confirmed'""",
            (lawyer_id, date, time),
        ).fetchone()
        return row is not None


def get_booked_times_for_day(lawyer_id: int, date: str) -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT appointment_time FROM appointments
               WHERE lawyer_id = ? AND appointment_date = ? AND status = 'confirmed'""",
            (lawyer_id, date),
        ).fetchall()
        return [r["appointment_time"] for r in rows]


def book_appointment(lawyer_id: int, client_name: str, date: str, time: str,
                      session_id: Optional[str] = None) -> Dict:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO appointments
               (lawyer_id, client_name, appointment_date, appointment_time, session_id)
               VALUES (?, ?, ?, ?, ?)""",
            (lawyer_id, client_name, date, time, session_id),
        )
        appt_id = cur.lastrowid
        row = conn.execute(
            """SELECT a.*, l.name AS lawyer_name, l.specialization AS lawyer_specialization
               FROM appointments a JOIN lawyers l ON a.lawyer_id = l.id
               WHERE a.id = ?""",
            (appt_id,),
        ).fetchone()
        return dict(row)


def get_appointments_for_session(session_id: str) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT a.*, l.name AS lawyer_name, l.specialization AS lawyer_specialization
               FROM appointments a JOIN lawyers l ON a.lawyer_id = l.id
               WHERE a.session_id = ? ORDER BY a.id DESC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
