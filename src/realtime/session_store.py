"""
realtime/session_store.py

In-memory store for active RealtimeSession objects.
Handles session TTL and cleanup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from src.realtime.engine import RealtimeSession

# Sessions older than this are auto-expired
_SESSION_TTL_MINUTES = 120


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, RealtimeSession]   = {}
        self._created:  dict[str, datetime]          = {}

    def put(self, session: RealtimeSession) -> None:
        self._sessions[session.session_id] = session
        self._created[session.session_id]  = datetime.now()

    def get(self, session_id: str) -> Optional[RealtimeSession]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None)
        self._created.pop(session_id, None)
        return removed is not None

    def list_sessions(self) -> list[dict]:
        now = datetime.now()
        result = []
        for sid, sess in self._sessions.items():
            created = self._created.get(sid, now)
            result.append({
                "session_id":    sid,
                "date_str":      sess.date_str,
                "day_type":      sess.day_type,
                "model":         sess.model,
                "station_codes": sess.station_codes,
                "has_real_data": sess.has_real_data,
                "created_at":    created.isoformat(),
                "age_minutes":   (now - created).seconds // 60,
            })
        return result

    def purge_expired(self) -> int:
        cutoff = datetime.now() - timedelta(minutes=_SESSION_TTL_MINUTES)
        expired = [
            sid for sid, ts in self._created.items() if ts < cutoff
        ]
        for sid in expired:
            self.delete(sid)
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)


# Singleton instance — imported by the router
store = SessionStore()
