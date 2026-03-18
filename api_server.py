"""
FastAPI server for Galadrial: POST /chat with a message, get the assistant reply.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Then open http://<this-pc-ip>:8000/ in a browser from this PC or another device on the same LAN.
"""

import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from assistant_engine import handle_message
from dnd_loader import (
    build_context_from_llm_selection,
    get_selection_catalog,
    load_images_as_data_urls,
    _parse_llm_selection_response,
)
from llm import ask_lmstudio, ask_lmstudio_with_images
from reminder_engine import get_reminder_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_event(msg: str) -> None:
    logger.info(msg)


def _lan_ips() -> list[str]:
    """Return this machine's LAN IPv4 addresses (excluding localhost)."""
    ips = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips or ["<get your PC IP from ipconfig>"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Galadrial API server started. POST /chat with {\"message\": \"...\"}")
    port = 8000
    for ip in _lan_ips():
        logger.info("LAN URL: http://%s:%s/ (use this from phone/other devices on same WiFi)", ip, port)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Galadrial API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class DndImprovRequest(BaseModel):
    transcript: str


class CancelReminderRequest(BaseModel):
    reminder_id: int


@app.post("/dnd-improv", response_model=ChatResponse)
async def dnd_improv(body: DndImprovRequest):
    """D&D improv endpoint."""
    transcript = (body.transcript or "").strip()
    try:
        catalog = get_selection_catalog()
        if catalog is None or (not catalog.chunks_with_sources and not catalog.image_paths):
            ctx = build_context_from_llm_selection([], [], [], [])
        else:
            selection_prompt = (
                "You are helping a D&D DM. Below is the recent table conversation and a catalog of "
                "available campaign notes (chunk number and preview) and map filenames. "
                "Reply with ONLY two lines: which chunk numbers are relevant (CHUNKS: 0, 1, 3), "
                "and which map filenames are relevant (MAPS: file.png, or MAPS: none). "
                "Pick only what is needed for the current conversation.\n\n"
                "--- Recent conversation ---\n"
                f"{transcript or '(no transcript)'}\n\n"
                "--- Catalog ---\n"
                f"{catalog.catalog_text}"
            )
            sel_response = ask_lmstudio(selection_prompt)
            sel_text = (sel_response.get("output") or [{}])[0].get("content", "") or ""
            indices, map_filenames = _parse_llm_selection_response(sel_text)
            ctx = build_context_from_llm_selection(
                catalog.chunks_with_sources,
                indices,
                catalog.image_paths,
                map_filenames,
            )
        prompt = (
            "You are an improv assistant for a D&D DM. Use the following campaign notes and character context "
            "(and the attached maps if any). Given the recent conversation at the table, suggest short dialogue "
            "or description the DM can read aloud. Reply with only the suggested text (1–3 sentences unless more is needed).\n\n"
            "--- Campaign notes ---\n"
            f"{ctx.notes_text}\n\n"
            "--- Recent conversation ---\n"
            f"{transcript or '(no transcript)'}"
        )
        image_data_urls = load_images_as_data_urls(ctx.image_paths) if ctx.image_paths else []
        response = ask_lmstudio_with_images(prompt, image_data_urls)
        reply = (response.get("output") or [{}])[0].get("content", "").strip() or "No response."
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.exception("D&D improv failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web client."""
    return get_client_html()


@app.get("/dnd", response_class=HTMLResponse)
@app.get("/dnd/", response_class=HTMLResponse)
async def dnd_page():
    """Serve the D&D improv page."""
    return get_dnd_html()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Send a message, get the assistant's reply."""
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must be non-empty")
    try:
        reply = handle_message(message, log_fn=_log_event)
        return ChatResponse(reply=reply or "No response.")
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reminders", response_class=HTMLResponse)
@app.get("/reminders/", response_class=HTMLResponse)
async def reminders_page():
    """Serve the reminders dashboard."""
    return get_reminders_html()


@app.get("/reminders/data")
async def reminders_data():
    """Return active and recently fired reminders as JSON."""
    engine = get_reminder_engine(log_fn=_log_event)
    return {
        "active": engine.list_active(),
        "fired": engine.list_fired(),
    }


@app.post("/reminders/cancel")
async def reminders_cancel(body: CancelReminderRequest):
    """Cancel a pending reminder by ID."""
    engine = get_reminder_engine(log_fn=_log_event)
    ok = engine.cancel(body.reminder_id)
    if ok:
        return {"success": True, "message": f"Reminder #{body.reminder_id} cancelled."}
    raise HTTPException(status_code=404, detail=f"Reminder #{body.reminder_id} not found or already fired.")


@app.post("/reminders/set")
async def reminders_set(body: dict):
    """Set a new reminder via the web UI."""
    message = str(body.get("message", "reminder")).strip()
    try:
        minutes = float(body.get("minutes", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="minutes must be a number")
    if minutes <= 0:
        raise HTTPException(status_code=400, detail="minutes must be greater than 0")
    engine = get_reminder_engine(log_fn=_log_event)
    reminder = engine.add(message, minutes)
    return {
        "success": True,
        "reminder": reminder.to_dict(),
    }


def get_client_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Galadrial</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0b0d10;
      color: #f7f7f7;
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    h1 { margin: 1rem; font-size: 1.25rem; color: #f0b34a; }
    #log {
      flex: 1;
      overflow: auto;
      padding: 1rem;
      margin: 0 1rem 1rem;
      background: #11141a;
      border-radius: 8px;
      font-size: 0.95rem;
      line-height: 1.5;
    }
    #log .you { color: #9a9fb2; margin-bottom: 0.5rem; }
    #log .assistant { color: #f7f7f7; margin-bottom: 1rem; }
    #form {
      display: flex;
      gap: 0.5rem;
      padding: 1rem;
      background: #11141a;
    }
    #input {
      flex: 1;
      padding: 0.75rem 1rem;
      border: 1px solid #222733;
      border-radius: 8px;
      background: #151922;
      color: #f7f7f7;
      font-size: 1rem;
    }
    #input:focus { outline: none; border-color: #f0b34a; }
    button {
      padding: 0.75rem 1.25rem;
      background: #f0b34a;
      color: #050608;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #e0a33a; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .error { color: #e55; }
  </style>
</head>
<body>
  <h1>Galadrial</h1>
  <div id="log"></div>
  <form id="form">
    <input type="text" id="input" placeholder="Message..." autocomplete="off" />
    <button type="submit" id="send">Send</button>
  </form>
</body>
</html>"""


def get_dnd_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D&D Improv - Galadrial</title>
</head>
<body>
  <h1>D&D Improv</h1>
  <p>D&D improv page content</p>
</body>
</html>"""


def get_reminders_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reminders - Galadrial</title>
</head>
<body>
  <h1>Reminders Dashboard</h1>
  <p>Reminders management interface</p>
</body>
</html>"""
