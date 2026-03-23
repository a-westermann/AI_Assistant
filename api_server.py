"""
FastAPI server for Galadrial: POST /chat with a message, get the assistant reply.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Then open http://<this-pc-ip>:8000/ in a browser from this PC or another device on the same LAN.
"""

import logging
import socket
import asyncio
import os
import tempfile
import threading
import random
import requests
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from assistant_engine import handle_message
from faster_whisper import WhisperModel
from dnd_loader import (
    build_context_from_llm_selection,
    get_selection_catalog,
    load_images_as_data_urls,
    _parse_llm_selection_response,
)
from llm import ask_lmstudio, ask_lmstudio_with_images
from lights_client import (
    toggle_all_lights,
    set_lights_auto,
    set_lights_style,
    LightsClientError,
)
from nanoleaf.nanoleaf import (
    turn_on,
    turn_off,
    set_effect,
    set_brightness as set_nanoleaf_brightness,
    get_scene_list,
    get_token as get_nanoleaf_token,
    NANOLEAF_IP,
)
from user_memory import resolve_alias, resolve_alias_match
from weather_client import (
    get_current_weather_summary,
    get_day_weather_forecast_summary,
    get_weather_ui_payload,
    WeatherClientError,
)
from auto_lighting_sync import start_auto_lighting_sync, stop_auto_lighting_sync, is_auto_lighting_sync_live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stt_model: WhisperModel | None = None
stt_lock = asyncio.Lock()
_async_failures: deque[str] = deque(maxlen=20)
_async_failures_lock = threading.Lock()


def _log_event(msg: str) -> None:
    logger.info(msg)


def _looks_like_lighting_action_text(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    # Let full wake/sleep routines run normally (they include greeting/weather behavior).
    if "good morning" in t or "good night" in t:
        return False
    if "remember " in t and ("means" in t or "i mean" in t or "when i say" in t):
        # Explicit memory-writing commands should not become background light actions.
        return False
    device_words = ("light", "lights", "govee", "nanoleaf", "scene", "brightness", "color")
    command_words = ("turn", "set", "make", "dim", "bright", "scene", "auto", "off", "on", "pulse", "animation")
    return any(w in t for w in device_words) and any(w in t for w in command_words)


def _looks_like_background_lighting_action(message: str) -> tuple[bool, str | None]:
    """Detect direct light commands OR alias phrases that expand to light commands."""
    raw = (message or "").strip()
    if not raw:
        return False, None
    if _looks_like_lighting_action_text(raw):
        return True, None
    match = resolve_alias_match(raw)
    expanded = match[1] if match else resolve_alias(raw)
    if expanded and ("good morning" in expanded.lower() or "good night" in expanded.lower()):
        return False, None
    if expanded and expanded != raw and _looks_like_lighting_action_text(expanded):
        matched_key = match[0] if match else None
        return True, matched_key
    return False, None


_LIGHT_ACK_VARIANTS = [
    "Got it, I'll adjust the lights for you.",
    "Sounds good. I'll handle the lighting now.",
    "On it, updating the lights now.",
    "Absolutely. I'll set that lighting change now.",
    "Great, I'll take care of the lights.",
    "Understood. I'll adjust the lighting right away.",
    "Perfect, I'll update the lights now.",
    "Got it. I'll make that lighting change.",
]


def _ack_for_lighting_action() -> str:
    return random.choice(_LIGHT_ACK_VARIANTS)


def _ack_for_lighting_action_with_context(message: str, alias_key: str | None) -> str:
    if alias_key:
        return f"Updating the lights to {alias_key}."
    return _ack_for_lighting_action()


def _record_async_failure(text: str) -> None:
    with _async_failures_lock:
        _async_failures.append(text.strip())


def _pop_async_failure() -> str | None:
    with _async_failures_lock:
        if not _async_failures:
            return None
        return _async_failures.popleft()


def _run_background_action(message: str) -> None:
    """Run a command in background; record only failures."""
    try:
        reply = handle_message(message, log_fn=_log_event) or ""
        low = reply.lower()
        # Keep this conservative: only queue obvious failure responses.
        if any(x in low for x in ("failed", "error", "sorry")):
            _record_async_failure(reply)
    except Exception as e:
        logger.exception("Background action failed")
        _record_async_failure(f"I tried to run that command but it failed: {e}")


def _sanitize_app_reply(reply: str) -> str:
    """Hide internal LLM transport errors from app clients."""
    text = (reply or "").strip()
    if text.lower().startswith("error calling model:"):
        return "Sorry, the AI is not in right now."
    return text


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

    # Whisper STT is loaded lazily on the first /stt request so server startup
    # doesn't block on downloading the model.
    logger.info("Whisper STT will load on first /stt request.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Galadrial API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_name: str | None = None


class ChatResponse(BaseModel):
    reply: str


class DndImprovRequest(BaseModel):
    transcript: str


class SttResponse(BaseModel):
    transcript: str


class GoveeStateRequest(BaseModel):
    state: str


class GoveeBrightnessRequest(BaseModel):
    brightness: int


class GoveeTemperatureRequest(BaseModel):
    temperature_k: int


class NanoleafStateRequest(BaseModel):
    state: str


class NanoleafSceneRequest(BaseModel):
    scene: str


class NanoleafBrightnessRequest(BaseModel):
    brightness: int


class WeatherResponse(BaseModel):
    summary: str


class WeatherUiResponse(BaseModel):
    location: str
    current_temp_f: int | None
    current_desc: str
    current_icon: str
    hourly: list[dict]


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
                    "Pick only what is needed for the current conversation. Use plain ASCII text only; no emojis.\n\n"
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
            "Use plain ASCII text only. No emojis, no special typographic symbols, and no markdown.\n\n"
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


@app.post("/stt", response_model=SttResponse)
async def stt_audio(file: UploadFile = File(...)):
    """Server-side speech-to-text for the web UI mic button (Whisper)."""
    global stt_model

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    # WhisperModel expects a real file path; write to a temp file.
    suffix = Path(file.filename or "").suffix.lower() or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        async with stt_lock:
            if stt_model is None:
                whisper_model_name = os.environ.get("WHISPER_MODEL", "base.en")
                whisper_device = os.environ.get("WHISPER_DEVICE", "cpu")
                whisper_compute_type = os.environ.get(
                    "WHISPER_COMPUTE_TYPE",
                    "int8" if whisper_device.lower() == "cpu" else "float16",
                )
                stt_model = await asyncio.to_thread(
                    lambda: WhisperModel(
                        whisper_model_name,
                        device=whisper_device,
                        compute_type=whisper_compute_type,
                    )
                )

            def _do_transcribe() -> str:
                segments, _info = stt_model.transcribe(
                    tmp_path,
                    language="en",
                    task="transcribe",
                    vad_filter=True,
                    beam_size=1,
                )
                return " ".join((seg.text or "").strip() for seg in segments).strip()

            transcript = await asyncio.to_thread(_do_transcribe)
        return SttResponse(transcript=transcript or "")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/health")
async def health():
    """Simple health check for connectivity."""
    return {"status": "ok"}


@app.get("/lights/nanoleaf/scenes")
async def nanoleaf_scenes():
    """Return available Nanoleaf scene names from scenes.txt."""
    try:
        scenes = [s for s in get_scene_list() if s and not s.strip().startswith("#")]
        return {"scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/govee/state")
async def lights_govee_state(body: GoveeStateRequest):
    try:
        stop_auto_lighting_sync(log_fn=_log_event)
        state = (body.state or "").strip().lower()
        if state not in ("on", "off"):
            raise HTTPException(status_code=400, detail="state must be 'on' or 'off'")
        return toggle_all_lights(state)  # type: ignore[arg-type]
    except LightsClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/govee/auto")
async def lights_govee_auto():
    try:
        return start_auto_lighting_sync(log_fn=_log_event)
    except LightsClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lights/auto/status")
async def lights_auto_status():
    """Return whether backend Nanoleaf auto-sync worker is currently running."""
    return {"success": True, "auto_sync_live": is_auto_lighting_sync_live()}


@app.post("/lights/govee/brightness")
async def lights_govee_brightness(body: GoveeBrightnessRequest):
    try:
        stop_auto_lighting_sync(log_fn=_log_event)
        brightness = max(0, min(100, int(body.brightness)))
        return set_lights_style(state="on", brightness=brightness)
    except LightsClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/govee/temperature")
async def lights_govee_temperature(body: GoveeTemperatureRequest):
    try:
        stop_auto_lighting_sync(log_fn=_log_event)
        temperature_k = max(1000, min(10000, int(body.temperature_k)))
        return set_lights_style(state="on", color_temp_k=temperature_k)
    except LightsClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/nanoleaf/state")
async def lights_nanoleaf_state(body: NanoleafStateRequest):
    state = (body.state or "").strip().lower()
    if state not in ("on", "off"):
        raise HTTPException(status_code=400, detail="state must be 'on' or 'off'")
    try:
        if state == "on":
            turn_on()
        else:
            turn_off()
        return {"success": True, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/nanoleaf/scene")
async def lights_nanoleaf_scene(body: NanoleafSceneRequest):
    stop_auto_lighting_sync(log_fn=_log_event)
    scene = (body.scene or "").strip()
    if not scene:
        raise HTTPException(status_code=400, detail="scene must be non-empty")
    try:
        set_effect(scene)
        return {"success": True, "scene": scene}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/nanoleaf/brightness")
async def lights_nanoleaf_brightness(body: NanoleafBrightnessRequest):
    try:
        level = max(0, min(100, int(body.brightness)))
        set_nanoleaf_brightness(level)
        return {"success": True, "brightness": level}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lights/nanoleaf/status")
async def lights_nanoleaf_status():
    """Read Nanoleaf on/off and brightness from device state API."""
    try:
        token = get_nanoleaf_token()
        url = f"http://{NANOLEAF_IP}:16021/api/v1/{token}/state"
        resp = requests.get(url, timeout=5)
        if not resp.ok:
            raise HTTPException(status_code=500, detail=f"Nanoleaf status HTTP {resp.status_code}")
        parsed = resp.json()
        data = parsed if isinstance(parsed, dict) else {}
        on_val = ((data.get("on") or {}).get("value"))
        bri_val = ((data.get("brightness") or {}).get("value"))
        state = "on" if bool(on_val) else "off"
        brightness = int(bri_val) if isinstance(bri_val, (int, float)) else None
        return {"success": True, "state": state, "brightness": brightness}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/current", response_model=WeatherResponse)
async def weather_current():
    """Current weather summary from Open-Meteo wrapper."""
    try:
        return WeatherResponse(summary=get_current_weather_summary())
    except WeatherClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/forecast", response_model=WeatherResponse)
async def weather_forecast():
    """Today forecast summary (peak temp/time + rain likelihood)."""
    try:
        return WeatherResponse(summary=get_day_weather_forecast_summary())
    except WeatherClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/ui", response_model=WeatherUiResponse)
async def weather_ui():
    """Structured weather payload for mobile weather tab UI."""
    try:
        payload = get_weather_ui_payload(hours=12)
        return WeatherUiResponse(
            location=str(payload.get("location") or ""),
            current_temp_f=payload.get("current_temp_f"),
            current_desc=str(payload.get("current_desc") or ""),
            current_icon=str(payload.get("current_icon") or "mixed"),
            hourly=list(payload.get("hourly") or []),
        )
    except WeatherClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Send a message, get the assistant's reply. Runs routing + tools + LLM on this PC."""
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must be non-empty")
    try:
        # For lighting actions (including alias phrases that expand to light commands):
        # acknowledge immediately, run in background,
        # and avoid completion chatter unless there is a failure.
        is_bg_light, alias_key = _looks_like_background_lighting_action(message)
        if is_bg_light:
            threading.Thread(
                target=_run_background_action,
                args=(message,),
                daemon=True,
                name="galadrial-bg-action",
            ).start()
            return ChatResponse(reply=_ack_for_lighting_action_with_context(message, alias_key))

        effective_name = (body.user_name or "").strip() or "Guest"
        reply = handle_message(message, log_fn=_log_event, user_name=effective_name)
        pending_failure = _pop_async_failure()
        if pending_failure:
            if reply:
                reply = f"{pending_failure}\n\n{reply}"
            else:
                reply = pending_failure
        reply = _sanitize_app_reply(reply or "")
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
      height: 100vh;      /* fallback */
      height: 100dvh;    /* mobile browser dynamic viewport height */
      display: flex;
      flex-direction: column;
    }
    h1 { margin: 0.75rem 1rem 0.5rem; font-size: 1.25rem; color: #f0b34a; }
    #log {
      flex: 1 1 auto;
      overflow: auto;
      min-height: 0;
      padding: 1rem;
      margin: 0 1rem 0.5rem;
      background: #11141a;
      border-radius: 8px;
      font-size: 0.95rem;
      line-height: 1.5;
    }
    #log .you { color: #9a9fb2; margin-bottom: 0.5rem; }
    #log .assistant { color: #f7f7f7; margin-bottom: 0.75rem; }
    #form {
      display: flex;
      gap: 0.5rem;
      padding: 0.75rem 1rem;
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
  <div id="tts" style="padding: 0 1rem 0.5rem; display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">
    <label style="display:flex; gap:0.4rem; align-items:center; color:#cdd0e0; font-size:0.95rem;">
      <input type="checkbox" id="speakToggle" checked />
      Speak responses
    </label>
    <select id="voiceSelect" style="background:#151922; color:#f7f7f7; border:1px solid #222733; border-radius:8px; padding:0.4rem 0.6rem;"></select>
  </div>
  <div id="log"></div>
  <form id="form">
    <input type="text" id="input" placeholder="Message..." autocomplete="off" />
    <button type="button" id="mic" title="Voice input">Mic</button>
    <button type="submit" id="send">Send</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const micBtn = document.getElementById('mic');
    const speakToggle = document.getElementById('speakToggle');
    const voiceSelect = document.getElementById('voiceSelect');

    function _stripGaladrialPrefix(text) {
      const t = (text || '').toString().trim();
      return t.startsWith('Galadrial: ') ? t.slice('Galadrial: '.length) : t;
    }

    function speak(text) {
      if (!speakToggle || !speakToggle.checked) return;
      if (!window.speechSynthesis) return;
      const t = _stripGaladrialPrefix(text);
      if (!t) return;
      try {
        window.speechSynthesis.cancel();
        // On some mobile browsers, the voice list updates only after TTS is used.
        // Re-populate right before speaking so newly installed voices appear.
        populateVoices();
        const utter = new SpeechSynthesisUtterance(t);
        const voices = window.speechSynthesis.getVoices() || [];
        if (voiceSelect && voiceSelect.value) {
          const v = voices.find((vv) => vv.name === voiceSelect.value);
          if (v) {
            utter.voice = v;
            if (v.lang) utter.lang = v.lang;
          }
        }
        window.speechSynthesis.speak(utter);
      } catch (e) {
        // ignore speech errors
      }
    }

    function populateVoices() {
      if (!voiceSelect || !window.speechSynthesis) return;
      const voices = window.speechSynthesis.getVoices() || [];
      const saved = localStorage.getItem('galadrial_tts_voice');
      voiceSelect.innerHTML = '';
      voices.forEach((v) => {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.lang})`;
        voiceSelect.appendChild(opt);
      });
      if (saved && voices.some((v) => v.name === saved)) {
        voiceSelect.value = saved;
      } else if (voices.length) {
        const en = (v) => (v && (v.lang || '')).toLowerCase().startsWith('en');
        const isNatural = (v) => v && /neural|natural/i.test(v.name || '');
        const best =
          voices.find((v) => en(v) && isNatural(v)) ||
          voices.find((v) => isNatural(v)) ||
          voices.find((v) => en(v)) ||
          voices[0];
        voiceSelect.value = best.name;
      }
    }

    if (voiceSelect && window.speechSynthesis) {
      populateVoices();
      window.speechSynthesis.onvoiceschanged = () => populateVoices();
      // Some mobile browsers load/install voice packs asynchronously.
      // Retrying a few times helps ensure newly-installed voices show up.
      setTimeout(() => populateVoices(), 1000);
      setTimeout(() => populateVoices(), 2500);
      setTimeout(() => populateVoices(), 5000);
      voiceSelect.addEventListener('change', () => {
        localStorage.setItem('galadrial_tts_voice', voiceSelect.value);
      });
    }

    function append(sender, text, isError) {
      const div = document.createElement('div');
      div.className = sender + (isError ? ' error' : '');
      div.textContent = (sender === 'you' ? 'You: ' : 'Galadrial: ') + text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
      if (sender === 'assistant' && !isError) speak(text);
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

    // --- Voice input: browser records audio, server transcribes with Whisper ---
    let mediaRecorder = null;
    let micStream = null;
    let audioChunks = [];
    let currentMimeType = '';

    function _pickMimeType() {
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      for (const c of candidates) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
      }
      return '';
    }

    async function _transcribeBlob(blob, mimeType) {
      const fd = new FormData();
      const ext = mimeType && mimeType.toLowerCase().includes('ogg') ? 'ogg' : 'webm';
      fd.append('file', blob, `mic.${ext}`);
      const r = await fetch('/stt', { method: 'POST', body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText || 'STT failed');
      return (data.transcript || '').trim();
    }

    async function _startRecording() {
      input.value = '';
      micBtn.disabled = true;
      sendBtn.disabled = true;
      micBtn.textContent = 'Listening...';

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStream = stream;
      audioChunks = [];
      currentMimeType = _pickMimeType();
      const options = {};
      if (currentMimeType) options.mimeType = currentMimeType;

      mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          if (micStream) micStream.getTracks().forEach((t) => t.stop());

          const blob = new Blob(audioChunks, { type: currentMimeType || 'audio/webm' });
          micBtn.textContent = 'Mic';
          micBtn.disabled = false;
          sendBtn.disabled = false;

          if (!blob.size) return;
          const transcript = await _transcribeBlob(blob, currentMimeType);
          if (!transcript) return;
          input.value = transcript;
          if (input.value.trim()) {
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
          }
        } catch (e) {
          micBtn.textContent = 'Mic';
          micBtn.disabled = false;
          sendBtn.disabled = false;
          append('assistant', 'Mic transcription failed: ' + ((e && e.message) ? e.message : String(e)), true);
        } finally {
          mediaRecorder = null;
          micStream = null;
          audioChunks = [];
          currentMimeType = '';
        }
      };

      mediaRecorder.start();
      micBtn.disabled = false;
      micBtn.textContent = 'Stop';
    }

    if (micBtn && window.MediaRecorder) {
      micBtn.addEventListener('click', async () => {
        try {
          if (mediaRecorder && mediaRecorder.state === 'recording') {
            micBtn.disabled = true;
            mediaRecorder.stop();
            return;
          }
          await _startRecording();
        } catch (e) {
          micBtn.textContent = 'Mic';
          micBtn.disabled = false;
          sendBtn.disabled = false;
          append('assistant', 'Mic start failed: ' + ((e && e.message) ? e.message : String(e)), true);
        }
      });
    } else if (micBtn) {
      micBtn.disabled = true;
      micBtn.textContent = 'Mic (unsupported)';
    }
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

    // Whisper STT: browser records audio, server transcribes, then we fill the editable textarea.
    let mediaRecorder = null;
    let micStream = null;
    let audioChunks = [];
    let currentMimeType = '';

    function _pickMimeType() {
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
      ];
      for (const c of candidates) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
      }
      return '';
    }

    async function _transcribeBlob(blob, mimeType) {
      const fd = new FormData();
      const ext = mimeType && mimeType.toLowerCase().includes('ogg') ? 'ogg' : 'webm';
      fd.append('file', blob, `mic.${ext}`);
      const r = await fetch('/stt', { method: 'POST', body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText || 'STT failed');
      return (data.transcript || '').trim();
    }

    if (window.MediaRecorder) {
      recordBtn.addEventListener('click', async () => {
        try {
          recordBtn.disabled = true;
          stopBtn.disabled = false;
          replyEl.textContent = 'Recording...';
          replyEl.classList.remove('empty');
          replyEl.classList.remove('error');

          transcript.value = transcript.value || '';
          audioChunks = [];
          currentMimeType = _pickMimeType();

          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          micStream = stream;
          const options = {};
          if (currentMimeType) options.mimeType = currentMimeType;
          mediaRecorder = new MediaRecorder(stream, options);
          mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
          };
          mediaRecorder.onstop = async () => {
            try {
              if (micStream) micStream.getTracks().forEach((t) => t.stop());
              const blob = new Blob(audioChunks, { type: currentMimeType || 'audio/webm' });
              if (!blob.size) return;
              const text = await _transcribeBlob(blob, currentMimeType);
              transcript.value = text;
              replyEl.textContent = 'Transcript ready.';
              replyEl.classList.remove('error');
            } catch (e) {
              replyEl.textContent = 'Transcription failed: ' + ((e && e.message) ? e.message : String(e));
              replyEl.classList.add('error');
            } finally {
              recordBtn.disabled = false;
              stopBtn.disabled = true;
              mediaRecorder = null;
              micStream = null;
              audioChunks = [];
              currentMimeType = '';
            }
          };

          mediaRecorder.start();
        } catch (e) {
          recordBtn.disabled = false;
          stopBtn.disabled = true;
          replyEl.textContent = 'Mic start failed: ' + ((e && e.message) ? e.message : String(e));
          replyEl.classList.add('error');
        }
      });

      stopBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      });
    } else {
      recordBtn.textContent = 'Record (unsupported)';
      recordBtn.disabled = true;
      stopBtn.disabled = true;
    }

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
