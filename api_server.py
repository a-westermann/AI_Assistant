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
from pomodoro import get_timer as get_pomodoro_timer
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


@app.get("/health")
async def health():
    """Simple health check for connectivity."""
    return {"status": "ok"}


@app.get("/pomodoro", response_class=HTMLResponse)
@app.get("/pomodoro/", response_class=HTMLResponse)
async def pomodoro_page():
    """Serve the pomodoro focus timer web UI."""
    return get_pomodoro_html()


@app.get("/pomodoro/status")
async def pomodoro_status():
    """Return current pomodoro timer state for polling."""
    timer = get_pomodoro_timer(log_fn=_log_event)
    return timer.get_status()


@app.post("/pomodoro/start")
async def pomodoro_start():
    """Start a focus session."""
    timer = get_pomodoro_timer(log_fn=_log_event)
    return timer.start_focus()


@app.post("/pomodoro/stop")
async def pomodoro_stop():
    """Stop the pomodoro timer."""
    timer = get_pomodoro_timer(log_fn=_log_event)
    return timer.stop()


@app.post("/pomodoro/skip")
async def pomodoro_skip():
    """Skip to the next phase."""
    timer = get_pomodoro_timer(log_fn=_log_event)
    return timer.skip()


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
    return _DND_HTML


_DND_HTML = """<!DOCTYPE html>
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


def get_pomodoro_html() -> str:
    """Pomodoro focus timer page with celestial elvish aesthetic."""
    return _POMODORO_HTML


_POMODORO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Focus Timer - Galadrial</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --phase-primary: #9a9fb2;
      --phase-glow: rgba(154, 159, 178, 0.2);
      --phase-glow-strong: rgba(154, 159, 178, 0.35);
      --phase-bg-glow: rgba(154, 159, 178, 0.04);
      --bg-deep: #0b0d10;
      --bg-surface: #11141a;
      --border-dim: #1a1e28;
      --text-primary: #f7f7f7;
      --text-muted: #6b7084;
      --text-secondary: #9a9fb2;
      --ring-size: min(300px, 72vw);
      --transition-speed: 0.8s;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Cormorant Garamond', Georgia, serif;
      background: var(--bg-deep);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      overflow-x: hidden;
      position: relative;
    }

    body::before {
      content: '';
      position: fixed;
      top: 30%;
      left: 50%;
      width: 600px;
      height: 600px;
      transform: translate(-50%, -50%);
      background: radial-gradient(circle, var(--phase-bg-glow) 0%, transparent 70%);
      pointer-events: none;
      transition: background var(--transition-speed) ease;
      z-index: 0;
    }

    .particles {
      position: fixed;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }

    .particle {
      position: absolute;
      width: 3px; height: 3px;
      border-radius: 50%;
      background: var(--phase-primary);
      opacity: 0;
      animation: drift linear infinite;
    }

    @keyframes drift {
      0%   { opacity: 0; transform: translateY(100vh) translateX(0px); }
      10%  { opacity: 0.4; }
      90%  { opacity: 0.15; }
      100% { opacity: 0; transform: translateY(-20vh) translateX(var(--sway)); }
    }

    .container {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 2rem 1rem;
      gap: 2rem;
    }

    .brand {
      font-family: 'Cinzel', serif;
      font-size: 0.85rem;
      font-weight: 500;
      letter-spacing: 0.35em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: -0.5rem;
    }

    .timer-wrap {
      position: relative;
      width: var(--ring-size);
      height: var(--ring-size);
    }

    .timer-svg {
      width: 100%; height: 100%;
      transform: rotate(-90deg);
      filter: drop-shadow(0 0 18px var(--phase-glow));
      transition: filter var(--transition-speed) ease;
    }

    .ring-bg {
      fill: none;
      stroke: var(--border-dim);
      stroke-width: 3;
    }

    .ring-progress {
      fill: none;
      stroke: var(--phase-primary);
      stroke-width: 4;
      stroke-linecap: round;
      transition: stroke var(--transition-speed) ease, stroke-dashoffset 0.3s linear;
    }

    .ring-inner {
      fill: none;
      stroke: var(--phase-primary);
      stroke-width: 0.5;
      opacity: 0.2;
      transition: stroke var(--transition-speed) ease;
    }

    .timer-content {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
    }

    .phase-label {
      font-family: 'Cinzel', serif;
      font-size: 0.75rem;
      font-weight: 500;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--phase-primary);
      transition: color var(--transition-speed) ease;
    }

    .time-display {
      font-family: 'JetBrains Mono', monospace;
      font-size: clamp(2.4rem, 8vw, 3.4rem);
      font-weight: 300;
      letter-spacing: 0.05em;
      font-variant-numeric: tabular-nums;
      color: var(--text-primary);
      text-shadow: 0 0 30px var(--phase-glow);
      transition: text-shadow var(--transition-speed) ease;
      line-height: 1;
    }

    .phase-sub {
      font-family: 'Cormorant Garamond', serif;
      font-size: 0.95rem;
      font-style: italic;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }

    .sessions {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.6rem;
    }

    .dots {
      display: flex;
      gap: 10px;
      align-items: center;
    }

    .dot {
      width: 12px; height: 12px;
      border-radius: 50%;
      border: 1.5px solid var(--phase-primary);
      background: transparent;
      transition: all var(--transition-speed) ease;
    }

    .dot.filled {
      background: var(--phase-primary);
      box-shadow: 0 0 8px var(--phase-glow);
    }

    .session-count {
      font-size: 0.9rem;
      color: var(--text-muted);
      letter-spacing: 0.05em;
    }

    .controls {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      justify-content: center;
    }

    .btn {
      font-family: 'Cinzel', serif;
      font-size: 0.8rem;
      font-weight: 500;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      padding: 0.7rem 1.6rem;
      border-radius: 100px;
      border: 1px solid var(--phase-primary);
      background: transparent;
      color: var(--phase-primary);
      cursor: pointer;
      transition: all 0.3s ease;
      outline: none;
    }

    .btn:hover {
      background: var(--phase-glow);
      box-shadow: 0 0 20px var(--phase-glow);
    }

    .btn:active { transform: scale(0.97); }

    .btn.primary {
      background: var(--phase-primary);
      color: var(--bg-deep);
    }

    .btn.primary:hover {
      box-shadow: 0 0 25px var(--phase-glow-strong);
    }

    .btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    .btn:disabled:hover {
      background: transparent;
      box-shadow: none;
    }

    .btn.primary:disabled { opacity: 0.5; }

    .btn.primary:disabled:hover {
      background: var(--phase-primary);
      box-shadow: none;
    }

    [data-phase="focus"] {
      --phase-primary: #4488FF;
      --phase-glow: rgba(68, 136, 255, 0.25);
      --phase-glow-strong: rgba(68, 136, 255, 0.4);
      --phase-bg-glow: rgba(68, 136, 255, 0.05);
    }

    [data-phase="short_break"] {
      --phase-primary: #F0B34A;
      --phase-glow: rgba(240, 179, 74, 0.25);
      --phase-glow-strong: rgba(240, 179, 74, 0.4);
      --phase-bg-glow: rgba(240, 179, 74, 0.05);
    }

    [data-phase="long_break"] {
      --phase-primary: #44BB88;
      --phase-glow: rgba(68, 187, 136, 0.25);
      --phase-glow-strong: rgba(68, 187, 136, 0.4);
      --phase-bg-glow: rgba(68, 187, 136, 0.05);
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .container > * { animation: fadeUp 0.6s ease both; }
    .container > :nth-child(1) { animation-delay: 0.05s; }
    .container > :nth-child(2) { animation-delay: 0.15s; }
    .container > :nth-child(3) { animation-delay: 0.25s; }
    .container > :nth-child(4) { animation-delay: 0.35s; }

    @media (max-width: 400px) {
      .btn { padding: 0.6rem 1.2rem; font-size: 0.7rem; }
      .brand { font-size: 0.75rem; }
    }
  </style>
</head>
<body data-phase="idle">
  <div class="particles" id="particles"></div>
  <div class="container">
    <div class="brand">Galadrial</div>
    <div class="timer-wrap">
      <svg class="timer-svg" viewBox="0 0 200 200">
        <circle class="ring-inner" cx="100" cy="100" r="78" />
        <circle class="ring-bg" cx="100" cy="100" r="88" />
        <circle class="ring-progress" id="ringProgress" cx="100" cy="100" r="88"
          stroke-dasharray="553" stroke-dashoffset="0" />
      </svg>
      <div class="timer-content">
        <div class="phase-label" id="phaseLabel">Ready</div>
        <div class="time-display" id="timeDisplay">25:00</div>
        <div class="phase-sub" id="phaseSub">Start when you are</div>
      </div>
    </div>
    <div class="sessions">
      <div class="dots" id="dots">
        <div class="dot"></div><div class="dot"></div>
        <div class="dot"></div><div class="dot"></div>
      </div>
      <div class="session-count" id="sessionCount">0 completed today</div>
    </div>
    <div class="controls">
      <button class="btn primary" id="startBtn" onclick="doStart()">Begin</button>
      <button class="btn" id="skipBtn" onclick="doSkip()" disabled>Skip</button>
      <button class="btn" id="stopBtn" onclick="doStop()" disabled>Stop</button>
    </div>
  </div>
  <script>
    const CIRCUMFERENCE = 2 * Math.PI * 88;
    const ringEl       = document.getElementById('ringProgress');
    const phaseLabel   = document.getElementById('phaseLabel');
    const timeDisplay  = document.getElementById('timeDisplay');
    const phaseSub     = document.getElementById('phaseSub');
    const sessionCount = document.getElementById('sessionCount');
    const dotsEl       = document.getElementById('dots');
    const startBtn     = document.getElementById('startBtn');
    const skipBtn      = document.getElementById('skipBtn');
    const stopBtn      = document.getElementById('stopBtn');

    const PHASE_LABELS = { idle:'Ready', focus:'Focus', short_break:'Rest', long_break:'Recharge' };
    const PHASE_SUBS   = { idle:'Start when you are', focus:'Deep work', short_break:'Breathe', long_break:'You have earned this' };

    let currentPhase = 'idle';

    function fmt(s) {
      return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0');
    }

    function updateUI(d) {
      const phase     = d.state || 'idle';
      const completed = d.completed_today || 0;
      const remaining = d.remaining_seconds || 0;
      const duration  = d.phase_duration_seconds || 1;

      if (phase !== currentPhase) {
        currentPhase = phase;
        document.body.setAttribute('data-phase', phase);
      }

      phaseLabel.textContent = PHASE_LABELS[phase] || phase;
      phaseSub.textContent   = PHASE_SUBS[phase] || '';

      if (phase === 'idle') {
        timeDisplay.textContent = '25:00';
        ringEl.style.strokeDashoffset = '0';
      } else {
        timeDisplay.textContent = fmt(remaining);
        ringEl.style.strokeDashoffset = String((1 - remaining/duration) * CIRCUMFERENCE);
      }

      const inSet = completed % 4;
      dotsEl.querySelectorAll('.dot').forEach((dot, i) => {
        dot.classList.toggle('filled', i < inSet);
      });
      sessionCount.textContent = completed + ' completed today';

      const idle = phase === 'idle';
      startBtn.disabled    = !idle;
      startBtn.textContent = idle ? 'Begin' : (phase === 'focus' ? 'Focusing' : 'Resting');
      startBtn.classList.toggle('primary', idle);
      skipBtn.disabled = idle;
      stopBtn.disabled = idle;
    }

    async function api(path) {
      try { const r = await fetch(path, {method:'POST'}); return await r.json(); }
      catch(e) { console.error(e); return null; }
    }

    async function poll() {
      try { const r = await fetch('/pomodoro/status'); updateUI(await r.json()); }
      catch(e) { console.error(e); }
    }

    async function doStart() { startBtn.disabled=true; const d=await api('/pomodoro/start'); if(d) updateUI(d); }
    async function doStop()  { stopBtn.disabled=true;  const d=await api('/pomodoro/stop');  if(d) updateUI(d); }
    async function doSkip()  { skipBtn.disabled=true;  const d=await api('/pomodoro/skip');  if(d) updateUI(d); }

    function createParticles() {
      const c = document.getElementById('particles');
      for (let i = 0; i < 14; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = (Math.random()*100)+'%';
        const sz = (2+Math.random()*2)+'px';
        p.style.width = sz; p.style.height = sz;
        p.style.setProperty('--sway', (Math.random()*60-30)+'px');
        p.style.animationDuration = (18+Math.random()*20)+'s';
        p.style.animationDelay = (Math.random()*20)+'s';
        c.appendChild(p);
      }
    }

    createParticles();
    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""
