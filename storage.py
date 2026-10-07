"""SQLite persistence for chat sessions, messages and long-term memory facts."""
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("CHATBOT_DB_PATH") or Path(__file__).with_name("chatbot.db"))
DEFAULT_TITLE = "New chat"
TITLE_MAX_LEN = 40


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                fact_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )


# --- sessions -------------------------------------------------------------

def create_session() -> str:
    session_id = uuid.uuid4().hex
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, DEFAULT_TITLE, now, now),
        )
    return session_id


def list_sessions(limit: int = 50) -> list[sqlite3.Row]:
    with closing(_connect()) as conn:
        return conn.execute(
            # The message-id tie-break keeps the order deterministic when the clock returns the same
            # timestamp for two updates (coarse clock resolution, e.g. on Windows).
            "SELECT id, title, updated_at FROM sessions "
            "ORDER BY updated_at DESC, "
            "(SELECT MAX(m.id) FROM messages m WHERE m.session_id = sessions.id) DESC, "
            "sessions.rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()


def delete_session(session_id: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# --- messages -------------------------------------------------------------

def add_message(session_id: str, role: str, content: str) -> None:
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        if role == "user":
            title = " ".join(content.split())
            if len(title) > TITLE_MAX_LEN:
                title = title[: TITLE_MAX_LEN - 1].rstrip() + "…"
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND title = ?",
                (title, session_id, DEFAULT_TITLE),
            )


def get_messages(session_id: str) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# --- long-term memory -----------------------------------------------------

def add_memories(facts: list[str]) -> int:
    """Insert new facts, skipping duplicates (case/whitespace-insensitive). Returns count added."""
    added = 0
    with closing(_connect()) as conn, conn:
        for fact in facts:
            fact = " ".join(fact.split())
            if not fact:
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO memories (fact, fact_key, created_at) VALUES (?, ?, ?)",
                (fact, fact.lower(), _now()),
            )
            added += cur.rowcount
    return added


def list_memories() -> list[sqlite3.Row]:
    with closing(_connect()) as conn:
        return conn.execute("SELECT id, fact FROM memories ORDER BY id").fetchall()


def delete_memory(memory_id: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


def clear_memories() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM memories")
