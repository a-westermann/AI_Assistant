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
        # Connect to an external address to see which interface is used; doesn't send data
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
    """
    D&D improv: send transcript of table conversation, get suggested dialogue to read aloud.
    The LLM first chooses which note chunks and maps are relevant, then we send only those
    and get the dialogue suggestion (two LLM calls).
    """
    transcript = (body.transcript or "").strip()
    try:
        catalog = get_selection_catalog()
        if catalog is None or (not catalog.chunks_with_sources and not catalog.image_paths):
            # No campaign content: still run dialogue with empty context
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
            "or description the DM can read aloud. Reply with only the suggested text (1\u20133 sentences unless more is needed).\n\n"
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
    """Serve the web client so you can chat from a browser."""
    return get_client_html()


@app.get("/dnd", response_class=HTMLResponse)
@app.get("/dnd/", response_class=HTMLResponse)
async def dnd_page():
    """Serve the D&D improv page: record mic, transcribe, get suggested dialogue."""
    return get_dnd_html()


@app.get("/health")
async def health():
    """Simple health check for connectivity."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Send a message, get the assistant's reply. Runs routing + tools + LLM on this PC."""
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must be non-empty")
    try:
        reply = handle_message(message, log_fn=_log_event)
        return ChatResponse(reply=reply or "No response.")
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Reminders endpoints ----------------------------------------------------


@app.get("/reminders", response_class=HTMLResponse)
@app.get("/reminders/", response_class=HTMLResponse)
async def reminders_page():
    """Serve the reminders dashboard page."""
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


# ---- HTML pages --------------------------------------------------------------


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
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function append(sender, text, isError) {
      const div = document.createElement('div');
      div.className = sender + (isError ? ' error' : '');
      div.textContent = (sender === 'you' ? 'You: ' : 'Galadrial: ') + text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = '';
      append('you', message);
      sendBtn.disabled = true;
      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          append('assistant', data.detail || r.statusText || 'Request failed', true);
          return;
        }
        append('assistant', data.reply || 'No response.');
      } catch (err) {
        append('assistant', err.message || 'Network error', true);
      } finally {
        sendBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def get_dnd_html() -> str:
    """D&D improv page: Record/Stop (Web Speech API), transcript area, Send -> suggested dialogue."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D&D Improv - Galadrial</title>
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
    .controls { display: flex; gap: 0.5rem; padding: 0 1rem; margin-bottom: 0.5rem; }
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
    button.stop { background: #c55; color: #fff; }
    button.stop:hover { background: #b44; }
    label { display: block; margin: 0.5rem 1rem 0; font-size: 0.9rem; color: #9a9fb2; }
    #transcript {
      flex: 1;
      min-height: 120px;
      padding: 1rem;
      margin: 0.5rem 1rem 1rem;
      background: #11141a;
      border: 1px solid #222733;
      border-radius: 8px;
      color: #f7f7f7;
      font-size: 0.95rem;
      resize: vertical;
    }
    #transcript:focus { outline: none; border-color: #f0b34a; }
    #reply {
      padding: 1rem;
      margin: 0 1rem 1rem;
      background: #151922;
      border-radius: 8px;
      border-left: 4px solid #f0b34a;
      font-size: 1rem;
      line-height: 1.5;
      min-height: 60px;
    }
    #reply.empty { color: #9a9fb2; }
    .error { color: #e55; }
  </style>
</head>
<body>
  <h1>D&D Improv</h1>
  <div class="controls">
    <button type="button" id="recordBtn">Record</button>
    <button type="button" id="stopBtn" disabled class="stop">Stop</button>
    <button type="button" id="sendBtn">Get suggestion</button>
  </div>
  <label for="transcript">Transcript (editable)</label>
  <textarea id="transcript" placeholder="Record or paste the recent table conversation..."></textarea>
  <label>Read this aloud</label>
  <div id="reply" class="empty">Suggestions will appear here after you send the transcript.</div>
  <script>
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const sendBtn = document.getElementById('sendBtn');
    const transcript = document.getElementById('transcript');
    const replyEl = document.getElementById('reply');

    let recognition = null;
    let finalTranscript = '';

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) finalTranscript += e.results[i][0].transcript + ' ';
          else interim += e.results[i][0].transcript;
        }
        transcript.value = finalTranscript + interim;
      };
      recognition.onend = () => { recordBtn.disabled = false; stopBtn.disabled = true; }
    } else {
      recordBtn.textContent = 'Record (unsupported)';
      recordBtn.disabled = true;
    }

    recordBtn.addEventListener('click', () => {
      if (!recognition) return;
      finalTranscript = transcript.value.trim() + (transcript.value ? ' ' : '');
      recognition.start();
      recordBtn.disabled = true;
      stopBtn.disabled = false;
    });
    stopBtn.addEventListener('click', () => {
      if (recognition) recognition.stop();
    });

    sendBtn.addEventListener('click', async () => {
      const text = transcript.value.trim();
      if (!text) return;
      sendBtn.disabled = true;
      replyEl.textContent = 'Thinking...';
      replyEl.classList.remove('empty');
      try {
        const r = await fetch('/dnd-improv', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript: text })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          replyEl.textContent = data.detail || r.statusText || 'Request failed';
          replyEl.classList.add('error');
          return;
        }
        replyEl.textContent = data.reply || 'No response.';
        replyEl.classList.remove('error');
      } catch (err) {
        replyEl.textContent = err.message || 'Network error';
        replyEl.classList.add('error');
      } finally {
        sendBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def get_reminders_html() -> str:
    """Reminders dashboard: view active reminders, set new ones, cancel pending ones."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reminders - Galadrial</title>
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: #0b0d10;
      color: #f7f7f7;
      min-height: 100vh;
      padding: 2rem;
    }
    h1 {
      font-family: 'Cinzel', serif;
      font-size: 1.8rem;
      color: #f0b34a;
      margin-bottom: 0.25rem;
    }
    .subtitle {
      color: #9a9fb2;
      font-size: 0.9rem;
      margin-bottom: 2rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      max-width: 900px;
    }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

    .card {
      background: #11141a;
      border: 1px solid #222733;
      border-radius: 12px;
      padding: 1.5rem;
    }
    .card h2 {
      font-family: 'Cinzel', serif;
      font-size: 1rem;
      color: #f0b34a;
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* New reminder form */
    .form-row { display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }
    .form-row input {
      flex: 1;
      padding: 0.6rem 0.8rem;
      border: 1px solid #222733;
      border-radius: 8px;
      background: #151922;
      color: #f7f7f7;
      font-size: 0.9rem;
      font-family: inherit;
    }
    .form-row input:focus { outline: none; border-color: #f0b34a; }
    .form-row input[type="number"] { max-width: 100px; }
    button {
      padding: 0.6rem 1.2rem;
      background: #f0b34a;
      color: #050608;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      font-family: inherit;
    }
    button:hover { background: #e0a33a; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    button.cancel-btn {
      background: transparent;
      color: #e55;
      border: 1px solid #e55;
      padding: 0.3rem 0.6rem;
      font-size: 0.75rem;
    }
    button.cancel-btn:hover { background: rgba(238,85,85,0.1); }

    /* Reminder list */
    .reminder-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem;
      background: #151922;
      border-radius: 8px;
      margin-bottom: 0.5rem;
    }
    .reminder-info { flex: 1; }
    .reminder-msg {
      font-weight: 500;
      color: #f7f7f7;
      font-size: 0.9rem;
    }
    .reminder-time {
      color: #9a9fb2;
      font-size: 0.8rem;
      margin-top: 0.15rem;
    }
    .reminder-time .countdown {
      color: #f0b34a;
      font-weight: 600;
    }
    .empty-state {
      color: #9a9fb2;
      font-size: 0.85rem;
      font-style: italic;
      padding: 0.5rem 0;
    }

    /* Fired reminders */
    .fired-item {
      padding: 0.75rem;
      background: #1a1510;
      border-left: 3px solid #f0b34a;
      border-radius: 0 8px 8px 0;
      margin-bottom: 0.5rem;
    }
    .fired-item .reminder-msg { color: #f0b34a; }

    /* Status indicator */
    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #4ade80;
      margin-right: 0.5rem;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .back-link {
      display: inline-block;
      margin-top: 1.5rem;
      color: #9a9fb2;
      text-decoration: none;
      font-size: 0.85rem;
    }
    .back-link:hover { color: #f0b34a; }
  </style>
</head>
<body>
  <h1>Reminders</h1>
  <p class="subtitle"><span class="status-dot"></span>Auto-refreshes every 5 seconds</p>

  <div class="grid">
    <div class="card">
      <h2>Set a Reminder</h2>
      <div class="form-row">
        <input type="text" id="msg" placeholder="What to remember..." autocomplete="off" />
      </div>
      <div class="form-row">
        <input type="number" id="mins" placeholder="Min" min="0.1" step="0.5" value="10" />
        <button id="setBtn" onclick="setReminder()">Set Reminder</button>
      </div>
      <div id="setStatus" style="color:#4ade80; font-size:0.8rem; margin-top:0.5rem;"></div>
    </div>

    <div class="card">
      <h2>Recently Fired</h2>
      <div id="firedList"><span class="empty-state">None yet</span></div>
    </div>

    <div class="card" style="grid-column: 1 / -1;">
      <h2>Active Reminders</h2>
      <div id="activeList"><span class="empty-state">No active reminders</span></div>
    </div>
  </div>

  <a class="back-link" href="/">&larr; Back to chat</a>

  <script>
    async function refresh() {
      try {
        const r = await fetch('/reminders/data');
        const data = await r.json();
        renderActive(data.active || []);
        renderFired(data.fired || []);
      } catch (e) {
        console.error('Refresh failed:', e);
      }
    }

    function renderActive(items) {
      const el = document.getElementById('activeList');
      if (!items.length) {
        el.innerHTML = '<span class="empty-state">No active reminders</span>';
        return;
      }
      el.innerHTML = items.map(r => {
        const secs = Math.max(0, Math.round(r.remaining_seconds));
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        const timeStr = m > 0 ? m + 'm ' + s + 's' : s + 's';
        return '<div class="reminder-item">' +
          '<div class="reminder-info">' +
            '<div class="reminder-msg">' + esc(r.message) + '</div>' +
            '<div class="reminder-time"><span class="countdown">' + timeStr + '</span> remaining &middot; fires at ' + esc(r.fire_at) + '</div>' +
          '</div>' +
          '<button class="cancel-btn" onclick="cancelReminder(' + r.id + ')">Cancel</button>' +
        '</div>';
      }).join('');
    }

    function renderFired(items) {
      const el = document.getElementById('firedList');
      if (!items.length) {
        el.innerHTML = '<span class="empty-state">None yet</span>';
        return;
      }
      el.innerHTML = items.map(r => {
        return '<div class="fired-item">' +
          '<div class="reminder-msg">' + esc(r.message) + '</div>' +
          '<div class="reminder-time">Fired at ' + esc(r.fire_at) + '</div>' +
        '</div>';
      }).join('');
    }

    async function setReminder() {
      const msg = document.getElementById('msg').value.trim();
      const mins = parseFloat(document.getElementById('mins').value);
      if (!msg || !mins || mins <= 0) return;
      const btn = document.getElementById('setBtn');
      btn.disabled = true;
      try {
        const r = await fetch('/reminders/set', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg, minutes: mins })
        });
        const data = await r.json();
        if (data.success) {
          document.getElementById('msg').value = '';
          document.getElementById('setStatus').textContent = 'Reminder set!';
          setTimeout(() => document.getElementById('setStatus').textContent = '', 3000);
          refresh();
        }
      } catch (e) {
        document.getElementById('setStatus').textContent = 'Failed to set reminder';
        document.getElementById('setStatus').style.color = '#e55';
      } finally {
        btn.disabled = false;
      }
    }

    async function cancelReminder(id) {
      try {
        await fetch('/reminders/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reminder_id: id })
        });
        refresh();
      } catch (e) {
        console.error('Cancel failed:', e);
      }
    }

    function esc(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    // Auto-refresh every 5 seconds
    refresh();
    setInterval(refresh, 5000);

    // Allow Enter key in message input
    document.getElementById('msg').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); setReminder(); }
    });
  </script>
</body>
</html>
"""
