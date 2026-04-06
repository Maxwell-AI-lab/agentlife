"""FastAPI application — serves REST API and embedded Web UI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agentlife.server.routes import router

UI_DIR = Path(__file__).parent.parent / "ui" / "dist"

app = FastAPI(title="AgentLife", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if UI_DIR.exists() and (UI_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
