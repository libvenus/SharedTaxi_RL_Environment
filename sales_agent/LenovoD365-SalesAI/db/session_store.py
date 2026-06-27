"""Session persistence for the orchestrator.

In-process cache backed by a JSON file. Not safe for multiple replicas —
replace with Postgres JSONB for horizontal scaling (see
docs/orchestrator-architecture.md).
"""

import json
from pathlib import Path

from agent_framework import AgentSession


def _in_memory(session) -> dict | None:
    state = getattr(session, "state", None)
    if not isinstance(state, dict):
        return None
    in_memory = state.get("in_memory")
    return in_memory if isinstance(in_memory, dict) else None


class SessionStore:
    def __init__(self, path: str | Path, max_messages: int = 20) -> None:
        self._path = Path(path)
        self._max = max_messages
        self._sessions: dict[str, AgentSession] = {}
        self._load()

    def get(self, conversation_id: str):
        return self._sessions.get(conversation_id)

    def add(self, conversation_id: str, session: AgentSession) -> None:
        self._sessions[conversation_id] = session

    def delete(self, conversation_id: str) -> bool:
        """Drop a conversation's session and persist. Returns True if one existed."""
        existed = conversation_id in self._sessions
        self._sessions.pop(conversation_id, None)
        if existed:
            self.save()
        return existed

    def snapshot(self, conversation_id: str) -> dict:
        session = self._sessions.get(conversation_id)
        return session.to_dict() if session is not None else {}

    def messages(self, conversation_id: str) -> list:
        native = self.snapshot(conversation_id)
        state = native.get("state") if isinstance(native, dict) else {}
        in_memory = state.get("in_memory") if isinstance(state, dict) else {}
        msgs = in_memory.get("messages") if isinstance(in_memory, dict) else []
        return msgs if isinstance(msgs, list) else []

    def trim(self, session) -> None:
        """Sliding window: keep only the most recent N messages so history (and
        the LLM context) does not grow unbounded."""
        in_memory = _in_memory(session)
        if in_memory is None:
            return
        messages = in_memory.get("messages")
        if isinstance(messages, list) and len(messages) > self._max:
            in_memory["messages"] = messages[-self._max:]

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {cid: s.to_dict() for cid, s in self._sessions.items()}
            self._path.write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception:
            pass

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for cid, sdict in raw.items():
                self._sessions[cid] = AgentSession.from_dict(sdict)
        except Exception:
            pass
