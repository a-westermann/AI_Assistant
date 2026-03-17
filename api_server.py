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
from daily_briefing import gather_briefing_data, format_briefing_for_web
from dnd_loader import (
    build_context_from_llm_selection,
    get_selection_catalog,
    load_images_as_data_urls,
    _parse_llm_selection_response,
)
from test import ask_lmstudio, ask_lmstudio_with_images

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
    """Serve the web client so you can chat from a browser."""
    return get_client_html()


@app.get("/dnd", response_class=HTMLResponse)
@app.get("/dnd/", response_class=HTMLResponse)
async def dnd_page():
    """Serve the D&D improv page: record mic, transcribe, get suggested dialogue."""
    return get_dnd_html()


@app.get("/briefing", response_class=HTMLResponse)
@app.get("/briefing/", response_class=HTMLResponse)
async def briefing_page():
    """Serve the daily briefing dashboard."""
    return get_briefing_html()


@app.get("/briefing/data")
async def briefing_data():
    """Return briefing data as JSON for the dashboard to poll."""
    try:
        data = gather_briefing_data()
        return format_briefing_for_web(data)
    except Exception as e:
        logger.exception("Briefing data failed")
        raise HTTPException(status_code=500, detail=str(e))


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


def get_briefing_html() -> str:
    """Daily briefing dashboard: shows time, email counts, light status at a glance."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Briefing - Galadrial</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: #0b0d10;
      color: #f7f7f7;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    h1 {
      font-family: 'Cinzel', serif;
      font-size: 1.6rem;
      color: #f0b34a;
      margin-bottom: 0.25rem;
      letter-spacing: 0.05em;
    }
    .subtitle {
      color: #9a9fb2;
      font-size: 0.9rem;
      font-weight: 300;
      margin-bottom: 2rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      width: 100%;
      max-width: 600px;
    }
    .card {
      background: #11141a;
      border-radius: 12px;
      padding: 1.25rem;
      border: 1px solid #1a1e28;
      transition: border-color 0.3s;
    }
    .card:hover { border-color: #f0b34a33; }
    .card.wide { grid-column: 1 / -1; }
    .card-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9a9fb2;
      margin-bottom: 0.5rem;
    }
    .card-value {
      font-size: 1.8rem;
      font-weight: 600;
      color: #f7f7f7;
    }
    .card-value.small { font-size: 1.1rem; font-weight: 400; }
    .card-value .unit {
      font-size: 0.9rem;
      font-weight: 300;
      color: #9a9fb2;
    }
    .email-breakdown {
      display: flex;
      gap: 1rem;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }
    .email-cat {
      font-size: 0.85rem;
      color: #9a9fb2;
    }
    .email-cat .count {
      color: #f0b34a;
      font-weight: 600;
    }
    .status-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 0.5rem;
      vertical-align: middle;
    }
    .status-dot.on { background: #4ade80; box-shadow: 0 0 8px #4ade8066; }
    .status-dot.off { background: #555; }
    .status-dot.unknown { background: #f0b34a; }
    .briefing-btn {
      display: block;
      margin: 2rem auto 0;
      padding: 0.75rem 2rem;
      background: #f0b34a;
      color: #050608;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .briefing-btn:hover { background: #e0a33a; }
    .briefing-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    #briefing-reply {
      margin-top: 1.5rem;
      padding: 1rem;
      max-width: 600px;
      width: 100%;
      background: #151922;
      border-radius: 8px;
      border-left: 4px solid #f0b34a;
      font-size: 0.95rem;
      line-height: 1.6;
      min-height: 0;
      display: none;
    }
    .loading { color: #9a9fb2; font-style: italic; }
    .error { color: #e55; }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.4s ease-out; }
    @media (max-width: 480px) {
      .grid { grid-template-columns: 1fr; }
      .card-value { font-size: 1.4rem; }
    }
  </style>
</head>
<body>
  <h1>Galadrial</h1>
  <div class="subtitle" id="datetime">Loading...</div>
  <div class="grid" id="grid">
    <div class="card">
      <div class="card-label">Unread Email</div>
      <div class="card-value" id="email-total">&mdash;</div>
    </div>
    <div class="card">
      <div class="card-label">Lights</div>
      <div class="card-value small" id="lights-status">&mdash;</div>
    </div>
    <div class="card wide">
      <div class="card-label">Email Breakdown</div>
      <div class="email-breakdown" id="email-breakdown">
        <span class="email-cat">Loading...</span>
      </div>
    </div>
  </div>
  <button class="briefing-btn" id="briefingBtn">Get Briefing</button>
  <div id="briefing-reply"></div>
  <script>
    const grid = document.getElementById('grid');
    const datetimeEl = document.getElementById('datetime');
    const emailTotal = document.getElementById('email-total');
    const lightsStatus = document.getElementById('lights-status');
    const emailBreakdown = document.getElementById('email-breakdown');
    const briefingBtn = document.getElementById('briefingBtn');
    const briefingReply = document.getElementById('briefing-reply');

    async function loadData() {
      try {
        const r = await fetch('/briefing/data');
        if (!r.ok) throw new Error('Failed to load');
        const d = await r.json();
        datetimeEl.textContent = (d.date || '') + '  \\u00b7  ' + (d.time || '');
        if (d.email_error) {
          emailTotal.textContent = '\\u2014';
          emailTotal.title = 'Gmail not configured';
        } else if (d.email_total !== null && d.email_total !== undefined) {
          emailTotal.textContent = d.email_total;
        } else {
          emailTotal.textContent = '\\u2014';
        }
        const state = d.lights_state;
        if (state === 'on') {
          lightsStatus.innerHTML = '<span class="status-dot on"></span>On';
        } else if (state === 'off') {
          lightsStatus.innerHTML = '<span class="status-dot off"></span>Off';
        } else {
          lightsStatus.innerHTML = '<span class="status-dot unknown"></span>Unknown';
        }
        const cats = [
          { key: 'email_primary', label: 'Primary' },
          { key: 'email_updates', label: 'Updates' },
          { key: 'email_social', label: 'Social' },
          { key: 'email_promotions', label: 'Promos' },
        ];
        const parts = cats
          .filter(c => d[c.key] !== null && d[c.key] !== undefined)
          .map(c => '<span class="email-cat">' + c.label + ': <span class="count">' + d[c.key] + '</span></span>');
        emailBreakdown.innerHTML = parts.length ? parts.join('') : '<span class="email-cat">No category data</span>';
        grid.classList.add('fade-in');
      } catch (err) {
        datetimeEl.textContent = 'Could not load briefing data';
      }
    }

    briefingBtn.addEventListener('click', async () => {
      briefingBtn.disabled = true;
      briefingReply.style.display = 'block';
      briefingReply.className = 'loading';
      briefingReply.textContent = 'Gathering your briefing...';
      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: 'good morning, give me my daily briefing' })
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          briefingReply.className = 'error';
          briefingReply.textContent = data.detail || 'Briefing failed';
          return;
        }
        briefingReply.className = 'fade-in';
        briefingReply.textContent = data.reply || 'No response.';
        loadData();
      } catch (err) {
        briefingReply.className = 'error';
        briefingReply.textContent = err.message || 'Network error';
      } finally {
        briefingBtn.disabled = false;
      }
    });

    loadData();
    setInterval(loadData, 60000);
  </script>
</body>
</html>
"""
