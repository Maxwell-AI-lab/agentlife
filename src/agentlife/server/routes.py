"""REST API routes for the AgentLife dashboard."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from agentlife.store import Store

router = APIRouter()

_store: Store | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


# ── Sessions ──


@router.get("/sessions")
async def list_sessions(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    sessions = await _get_store().list_sessions(limit=limit, offset=offset)
    return {"sessions": [s.model_dump() for s in sessions]}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await _get_store().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await _get_store().delete_session(session_id)
    return {"ok": True}


@router.delete("/sessions")
async def clear_all():
    await _get_store().clear_all()
    return {"ok": True}


# ── Spans ──


@router.get("/sessions/{session_id}/spans")
async def get_spans(session_id: str):
    session = await _get_store().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    spans = await _get_store().get_spans(session_id)
    return {"spans": [s.model_dump() for s in spans]}


# ── Health ──


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
