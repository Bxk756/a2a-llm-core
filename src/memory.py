import sqlite3
from typing import List, Dict, Optional


class AgentMemory:
    """
    Persistent agent memory using SQLite.
    Stores conversation history per session_id.
    Includes audit helpers for export.
    """

    def __init__(self, db_path: str = "memory.db", max_turns: int = 10):
        self.db_path = db_path
        self.max_turns = max_turns
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Helpful index for audit queries
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_session_id
            ON memory(session_id)
            """
        )
        conn.commit()
        conn.close()

    def add(self, session_id: str, role: str, content: str):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO memory (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )

        # Enforce max_turns (each turn has AGENT A + AGENT B -> roughly 2 messages per turn)
        # Keep the most recent max_turns*2 rows per session.
        cur.execute(
            """
            DELETE FROM memory
            WHERE id NOT IN (
                SELECT id FROM memory
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            AND session_id = ?
            """,
            (session_id, self.max_turns * 2, session_id),
        )

        conn.commit()
        conn.close()

    def get_context(self, session_id: str) -> str:
        rows = self.fetch_messages(session_id=session_id, limit=None, ascending=True)

        lines: List[str] = []
        for r in rows:
            lines.append(f"{r['role']} : {r['content']}")

        return "\n".join(lines)

    def reset(self, session_id: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM memory WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    # -----------------------------
    # Audit helpers
    # -----------------------------
    def list_sessions(self, limit: int = 100) -> List[Dict]:
        """
        Returns recent sessions with their last timestamp and message count.
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT session_id,
                   COUNT(*) AS message_count,
                   MAX(ts) AS last_ts
            FROM memory
            GROUP BY session_id
            ORDER BY last_ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        out: List[Dict] = []
        for session_id, message_count, last_ts in rows:
            out.append(
                {
                    "session_id": session_id,
                    "message_count": int(message_count),
                    "last_ts": last_ts,
                }
            )
        return out

    def fetch_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> List[Dict]:
        """
        Fetch messages for a session_id.
        If limit is provided, returns only that many messages (most recent if ascending=False).
        """
        conn = self._connect()
        cur = conn.cursor()

        order = "ASC" if ascending else "DESC"
        if limit is None:
            cur.execute(
                f"""
                SELECT id, session_id, role, content, ts
                FROM memory
                WHERE session_id = ?
                ORDER BY id {order}
                """,
                (session_id,),
            )
        else:
            cur.execute(
                f"""
                SELECT id, session_id, role, content, ts
                FROM memory
                WHERE session_id = ?
                ORDER BY id {order}
                LIMIT ?
                """,
                (session_id, limit),
            )

        rows = cur.fetchall()
        conn.close()

        out: List[Dict] = []
        for _id, sid, role, content, ts in rows:
            out.append(
                {
                    "id": int(_id),
                    "session_id": sid,
                    "role": role,
                    "content": content,
                    "ts": ts,
                }
            )
        return out
