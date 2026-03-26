"""
FastAPI server for Galadrial: POST /chat with a message, get the assistant reply.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Then open http://<this-pc-ip>:8000/ in a browser from this PC or another device on the same LAN.
"""

import logging
import socket
import asyncio
import os
import hashlib
import hmac
import tempfile
import threading
import time
import random
import re
import requests
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from assistant_engine import handle_message
from faster_whisper import WhisperModel
from dnd.dnd_loader import (
    build_context_from_llm_selection,
    get_selection_catalog,
    get_rag_context,
    load_images_as_data_urls,
    _parse_llm_selection_response,
)
from llm import ask_lmstudio, ask_lmstudio_with_images
from lighting.lights_client import (
    get_lights_state,
    toggle_all_lights,
    set_lights_auto,
    set_lights_style,
    LightsClientError,
)
from lighting.nanoleaf.nanoleaf import (
    turn_on,
    turn_off,
    set_effect,
    set_brightness as set_nanoleaf_brightness,
    get_scene_list,
    get_token as get_nanoleaf_token,
    get_last_flow_attempt,
    NANOLEAF_IP,
)
from misc_tools.user_memory import resolve_alias, resolve_alias_match
from misc_tools.weather_client import (
    get_current_weather_summary,
    get_day_weather_forecast_summary,
    get_weather_ui_payload,
    WeatherClientError,
)
from lighting.auto_lighting_sync import (
    start_auto_lighting_sync,
    stop_auto_lighting_sync,
    is_auto_lighting_sync_live,
)
from music.play_resolver import (
    PlayResolution,
    format_play_resolution_reply,
    looks_like_play_music_request,
    resolve_play_to_video_id,
)
from music.spotify_client import (
    SpotifyAuthRequiredError,
    SpotifyNotConfiguredError,
    list_connect_devices,
    spotify_credentials_configured,
    spotify_has_cached_token,
)
from music.spotify_resolver import (
    SpotifyResolution,
    format_spotify_resolution_reply,
    looks_like_spotify_pause_request,
    looks_like_spotify_play_request,
    pause_spotify_playback,
    resolve_spotify_play,
)
from misc_tools.shopping_list_store import (
    add_item as shopping_add_item,
    update_item as shopping_update_item,
    delete_item as shopping_delete_item,
    get_items as shopping_get_items,
    get_sort_order as shopping_get_sort_order,
    set_sort_order as shopping_set_sort_order,
    replace_all as shopping_replace_all,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


stt_model: WhisperModel | None = None
stt_lock = asyncio.Lock()
_async_failures: deque[str] = deque(maxlen=20)
_async_failures_lock = threading.Lock()


def _log_event(msg: str) -> None:
    logger.info(msg)


def _is_wake_sleep_phrase(text: str) -> bool:
    """Match good morning/night variants like 'goodnight' and 'good-night'."""
    t = (text or "").lower().strip()
    if not t:
        return False
    return bool(re.search(r"\bgood[\s-]*morning\b", t) or re.search(r"\bgood[\s-]*night\b", t))


def _needs_full_lighting_chat_sync(text: str) -> bool:
    """
    True for Nanoleaf custom / flow animations: full /chat pipeline + real reply + nanoleaf_flow.
    Must be broad: speech and typos often omit words like 'colors' that the old lighting detector needed.
    """
    if not (text or "").strip():
        return False
    t = (text or "").lower()
    if re.search(r"\banimation|animating|animated\b|\banimate\b", t):
        return True
    if "rainbow" in t or "pride" in t:
        return True
    if "multicolor" in t or "multi-color" in t or "multi color" in t:
        return True
    if "custom" in t and re.search(r"\banimation|animating|animate|effect|flow\b", t):
        return True
    if any(k in t for k in ("create", "make", "new", "build", "add")) and re.search(
        r"\banimation|animate|rainbow|flowing|multicolor|cycle\b", t
    ):
        return True
    if "flow" in t and any(
        w in t for w in ("color", "colors", "colour", "light", "lights", "nanoleaf", "panel", "cycle")
    ):
        return True
    return False


def _nanoleaf_flow_fresh_since(start_time: float) -> dict | None:
    """Return last flow attempt if it occurred during this /chat request (approx)."""
    nf = get_last_flow_attempt()
    if not nf:
        return None
    if float(nf.get("time_unix") or 0) >= start_time - 0.25:
        return dict(nf)
    return None


def _lighting_alias_expansion(message: str) -> str | None:
    raw = (message or "").strip()
    if not raw:
        return None
    match = resolve_alias_match(raw)
    expanded = match[1] if match else resolve_alias(raw)
    if expanded and expanded.strip() and expanded.strip() != raw:
        return expanded.strip()
    return None


def _use_background_lighting_ack(message: str) -> bool:
    """Instant canned reply + background handle_message for simple lighting only."""
    if _needs_full_lighting_chat_sync(message):
        return False
    exp = _lighting_alias_expansion(message)
    if exp and _needs_full_lighting_chat_sync(exp):
        return False
    return True


def _looks_like_lighting_action_text(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    if _is_wake_sleep_phrase(t):
        return False
    if "remember " in t and ("means" in t or "i mean" in t or "when i say" in t):
        return False
    device_words = (
        "light",
        "lights",
        "govee",
        "nanoleaf",
        "scene",
        "brightness",
        "color",
        "colour",
        "rainbow",
    )
    command_words = (
        "turn",
        "set",
        "make",
        "dim",
        "bright",
        "scene",
        "auto",
        "off",
        "on",
        "pulse",
        "animation",
        "create",
        "custom",
    )
    return any(w in t for w in device_words) and any(w in t for w in command_words)


def _looks_like_background_lighting_action(message: str) -> tuple[bool, str | None]:
    """Direct light commands or aliases that expand to light commands."""
    raw = (message or "").strip()
    if not raw:
        return False, None
    if _is_wake_sleep_phrase(raw):
        return False, None
    if _looks_like_lighting_action_text(raw):
        return True, None
    match = resolve_alias_match(raw)
    expanded = match[1] if match else resolve_alias(raw)
    if expanded and _is_wake_sleep_phrase(expanded):
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
    try:
        reply = handle_message(message, log_fn=_log_event) or ""
        low = reply.lower()
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

    try:
        yield
    finally:
        logger.info("Shutting down.")


app = FastAPI(title="Galadrial API", lifespan=lifespan)

# D&D endpoint lock (simple password + device cookie).
# Env var:
# - `DND_PWD`: password required to use `/dnd/ask` and `/dnd-improv`.
# After successful unlock, we set an HttpOnly cookie so the same browser/device
# can access without re-entering the password.
DND_PWD = (os.environ.get("DND_PWD") or "").strip()
DND_COOKIE_NAME = "galadrial_dnd_authed"
DND_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days
DND_PWD_HASH = hashlib.sha256(DND_PWD.encode("utf-8")).hexdigest() if DND_PWD else ""


def _dnd_authed_from_request(request: Request) -> bool:
    if not DND_PWD:
        return True
    cookie_val = request.cookies.get(DND_COOKIE_NAME)
    return bool(cookie_val) and hmac.compare_digest(cookie_val, DND_PWD_HASH)


def _require_dnd_authed(request: Request) -> None:
    if not _dnd_authed_from_request(request):
        raise HTTPException(status_code=401, detail="D&D is locked. Click the lock icon and enter the password.")



class ChatRequest(BaseModel):
    message: str
    user_name: str | None = None


class MusicSearchPayload(BaseModel):
    """YouTube search result for a \"Play …\" message (video id for a downstream player)."""

    ok: bool
    video_id: str | None = None
    title: str | None = None
    search_query: str | None = None
    raw_intent: str | None = None
    error: str | None = None


class SpotifyPlayPayload(BaseModel):
    """Spotify URI + optional Connect playback result (librespot / Pi as a Connect device)."""

    ok: bool
    uri: str | None = None
    kind: str = "track"
    name: str | None = None
    artists: str | None = None
    search_query: str | None = None
    raw_intent: str | None = None
    playback_started: bool = False
    device_id: str | None = None
    error: str | None = None
    playback_error: str | None = None


class ChatResponse(BaseModel):
    reply: str
    music: MusicSearchPayload | None = None
    spotify: SpotifyPlayPayload | None = None
    # Set on synchronous /chat when create_flow_effect ran this request (check ok / detail without seeing panels).
    nanoleaf_flow: dict | None = None
    # Server + optional client timings for profiling (see web UI RESPONSE PROFILE).
    timing: dict | None = None


class DndImprovRequest(BaseModel):
    transcript: str


class DndAskRequest(BaseModel):
    """
    D&D Q&A: ask questions on the fly during a session.

    `transcript` is optional; providing it improves retrieval relevance.
    """

    question: str
    transcript: str | None = None
    # Maps can cause very large prompts (base64 data urls). For on-the-fly Q&A,
    # we default to notes-only and let you enable maps explicitly later.
    use_maps: bool = False


class DndAuthRequest(BaseModel):
    password: str


class SttResponse(BaseModel):
    transcript: str
    transcribe_server_ms: float | None = None


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
    day_summary: str | None = None
    hourly: list[dict]


class BenchmarkResponse(BaseModel):
    """
    Fixed prompt benchmark response for measuring assistant timing.
    Useful for quick "hit endpoint in browser" speed checks.
    """

    prompt: str
    reply: str
    timing: dict | None = None


class ShoppingItemCreateRequest(BaseModel):
    name: str


class ShoppingItemUpdateRequest(BaseModel):
    name: str | None = None
    checked: bool | None = None
    quantity: int | None = None


class ShoppingSortOrderRequest(BaseModel):
    reference_order: list[str]


class ShoppingReplaceAllRequest(BaseModel):
    items: list[dict]


class MusicPrepareRequest(BaseModel):
    """Phrase like \"Play Polica\" (must match play intent — leading Play …)."""

    phrase: str


class MusicPrepareResponse(BaseModel):
    reply: str
    music: MusicSearchPayload


class SpotifyPrepareRequest(BaseModel):
    """Phrase like \"Play lo-fi on Spotify\" or \"Spotify play Metallica\" (see spotify_resolver patterns)."""

    phrase: str
    device_id: str | None = None
    attempt_playback: bool = True
    skip_llm_refinement: bool = False


class SpotifyPrepareResponse(BaseModel):
    reply: str
    spotify: SpotifyPlayPayload


def _music_search_payload(res: PlayResolution) -> MusicSearchPayload:
    return MusicSearchPayload(
        ok=res.ok,
        video_id=res.video_id,
        title=res.title,
        search_query=res.search_query,
        raw_intent=res.raw_intent,
        error=res.error if not res.ok else None,
    )


def _spotify_play_payload(res: SpotifyResolution) -> SpotifyPlayPayload:
    return SpotifyPlayPayload(
        ok=res.ok,
        uri=res.uri,
        kind=res.kind,
        name=res.name,
        artists=res.artists,
        search_query=res.search_query,
        raw_intent=res.raw_intent,
        playback_started=res.playback_started,
        device_id=res.device_id,
        error=res.error if not res.ok else None,
        playback_error=res.playback_error,
    )


@app.get("/dnd/auth/status")
async def dnd_auth_status(request: Request):
    return {"authed": _dnd_authed_from_request(request)}


@app.post("/dnd/auth")
async def dnd_auth(body: DndAuthRequest, response: Response):
    if not DND_PWD:
        return {"ok": True, "authed": True}
    if not body.password:
        raise HTTPException(status_code=400, detail="password is required")
    pwd_hash = hashlib.sha256(body.password.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(pwd_hash, DND_PWD_HASH):
        raise HTTPException(status_code=401, detail="Incorrect password")
    response.set_cookie(
        key=DND_COOKIE_NAME,
        value=DND_PWD_HASH,
        max_age=DND_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True, "authed": True}


@app.post("/dnd-improv", response_model=ChatResponse)
async def dnd_improv(body: DndImprovRequest, request: Request):
    """
    D&D improv: send transcript of table conversation, get suggested dialogue to read aloud.
    The LLM first chooses which note chunks and maps are relevant, then we send only those
    and get the dialogue suggestion (two LLM calls).
    """
    _require_dnd_authed(request)
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


@app.post("/dnd/ask", response_model=ChatResponse)
async def dnd_ask(body: DndAskRequest, request: Request):
    """
    D&D Q&A: ask a question during a session.

    Uses your existing D&D RAG retrieval (embeddings + map keyword hints) to pull
    relevant campaign chunks, then answers from those notes.
    """
    _require_dnd_authed(request)
    question = (body.question or "").strip()
    transcript = (body.transcript or "").strip() if body.transcript else ""
    use_maps = bool(body.use_maps)
    if not question:
        raise HTTPException(status_code=400, detail="question must be non-empty")

    # Retrieval query: question alone works, but transcript makes it much better.
    retrieval_query = question if not transcript else f"{transcript}\n\n--- Question ---\n{question}"

    try:
        # Reduce token bloat: fewer chunks, and maps disabled by default.
        ctx = get_rag_context(
            retrieval_query,
            top_k_text=4,
            max_maps=1 if use_maps else 0,
        )
        # Hard truncate notes to keep the request safely under LM Studio's token limits.
        notes_text = (ctx.notes_text or "").strip()
        if len(notes_text) > 8000:
            notes_text = notes_text[:8000] + "\n\n[notes truncated]"
        image_data_urls = (
            load_images_as_data_urls(ctx.image_paths) if (use_maps and ctx.image_paths) else []
        )
        prompt = (
            "You are Galadrial, helping a D&D DM.\n"
            "Answer the DM's question using the campaign notes provided below.\n"
            "Note that you can find recaps of the campaign in Campaign Recaps.txt\n"
            "Rules:\n"
            "- Use plain ASCII text only (no markdown, no emojis).\n"
            "- If the notes do not contain the answer, say you don't know.\n"
            "- Do not invent names, titles, or events.\n"
            "- Keep it short and directly useful for running the session.\n\n"
            "--- Campaign notes (relevant) ---\n"
            f"{notes_text}\n\n"
            "--- DM question ---\n"
            f"{question}"
        )

        response = ask_lmstudio_with_images(prompt, image_data_urls)
        reply = (response.get("output") or [{}])[0].get("content", "").strip() or "No response."
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.exception("D&D ask failed")
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

            tr0 = time.perf_counter()
            transcript = await asyncio.to_thread(_do_transcribe)
            tr_ms = (time.perf_counter() - tr0) * 1000
        return SttResponse(transcript=transcript or "", transcribe_server_ms=round(tr_ms, 1))
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


@app.get("/benchmark/hotdogs", response_model=BenchmarkResponse)
async def benchmark_hotdogs():
    """
    Run a consistent benchmark prompt and return assistant reply + timing breakdown.
    """
    prompt = "How many hotdogs does it take end-to-end to reach the moon?"
    timing_out: dict = {}
    reply = handle_message(
        prompt,
        log_fn=_log_event,
        user_name="Benchmark",
        timing_out=timing_out,
    )
    return BenchmarkResponse(
        prompt=prompt,
        reply=_sanitize_app_reply(reply or ""),
        timing=timing_out or None,
    )


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


@app.get("/lights/diagnostic")
async def lights_diagnostic():
    """
    One JSON snapshot of Govee + Nanoleaf + auto-sync so you can verify behavior without
    walking to the room. Open in a browser: http://localhost:8000/lights/diagnostic
    """
    out: dict = {
        "success": True,
        "hint": "auto_sync_live=false: Nanoleaf time-sync thread stopped. "
        "POST /chat uses an instant ack for simple lighting only; custom animations wait for the full reply. "
        "govee.mode (if present) comes from your Govee /status.",
        "simple_lighting_instant_ack": True,
        "full_chat_for_custom_animations": True,
        "auto_sync_live": is_auto_lighting_sync_live(),
        "govee": {},
        "nanoleaf": {},
    }
    try:
        out["govee"] = get_lights_state()
    except LightsClientError as e:
        out["govee"] = {"error": str(e)}
    try:
        token = get_nanoleaf_token()
        base = f"http://{NANOLEAF_IP}:16021/api/v1/{token}"
        st = requests.get(f"{base}/state", timeout=5)
        out["nanoleaf"]["state_http"] = st.status_code
        if st.ok:
            data = st.json()
            if isinstance(data, dict):
                on_v = (data.get("on") or {}).get("value")
                bri_v = (data.get("brightness") or {}).get("value")
                out["nanoleaf"]["on"] = bool(on_v) if on_v is not None else None
                out["nanoleaf"]["brightness_pct"] = (
                    int(bri_v) if isinstance(bri_v, (int, float)) else bri_v
                )
        sel = requests.get(f"{base}/effects/select", timeout=5)
        out["nanoleaf"]["select_http"] = sel.status_code
        if sel.ok:
            j = sel.json()
            if isinstance(j, dict):
                out["nanoleaf"]["selected_effect"] = j.get("select")
    except Exception as e:
        out["nanoleaf"]["error"] = str(e)
    out["nanoleaf"]["last_flow_attempt"] = get_last_flow_attempt()
    return out


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
        stop_auto_lighting_sync(log_fn=_log_event)
        if state == "on":
            turn_on()
        else:
            turn_off()
        return {"success": True, "state": state, "auto_sync_live": is_auto_lighting_sync_live()}
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
        return {"success": True, "scene": scene, "auto_sync_live": is_auto_lighting_sync_live()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lights/nanoleaf/brightness")
async def lights_nanoleaf_brightness(body: NanoleafBrightnessRequest):
    try:
        stop_auto_lighting_sync(log_fn=_log_event)
        level = max(0, min(100, int(body.brightness)))
        set_nanoleaf_brightness(level)
        return {"success": True, "brightness": level, "auto_sync_live": is_auto_lighting_sync_live()}
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
        return {
            "success": True,
            "state": state,
            "brightness": brightness,
            "auto_sync_live": is_auto_lighting_sync_live(),
        }
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


@app.get("/shopping/items")
async def shopping_items_get():
    try:
        return {"items": shopping_get_items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shopping/items")
async def shopping_items_add(body: ShoppingItemCreateRequest):
    try:
        item = shopping_add_item(body.name)
        return {"success": True, "item": item}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="duplicate item")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/shopping/items/{item_id}")
async def shopping_items_patch(item_id: str, body: ShoppingItemUpdateRequest):
    try:
        item = shopping_update_item(item_id, name=body.name, checked=body.checked, quantity=body.quantity)
        return {"success": True, "item": item}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="duplicate item")
    except KeyError:
        raise HTTPException(status_code=404, detail="item not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/shopping/items/{item_id}")
async def shopping_items_delete(item_id: str):
    try:
        shopping_delete_item(item_id)
        return {"success": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="item not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shopping/sort-order")
async def shopping_sort_order_get():
    try:
        return {"reference_order": shopping_get_sort_order()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/shopping/sort-order")
async def shopping_sort_order_put(body: ShoppingSortOrderRequest):
    try:
        out = shopping_set_sort_order(body.reference_order or [])
        return {"success": True, "reference_order": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/shopping/items/replace-all")
async def shopping_items_replace_all(body: ShoppingReplaceAllRequest):
    try:
        out = shopping_replace_all(body.items or [])
        return {"success": True, "items": out}
    except Exception as e:
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
        try:
            day_summary = get_day_weather_forecast_summary()
        except WeatherClientError:
            day_summary = None
        return WeatherUiResponse(
            location=str(payload.get("location") or ""),
            current_temp_f=payload.get("current_temp_f"),
            current_desc=str(payload.get("current_desc") or ""),
            current_icon=str(payload.get("current_icon") or "mixed"),
            day_summary=day_summary,
            hourly=list(payload.get("hourly") or []),
        )
    except WeatherClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _merge_chat_timing(
    *,
    path: str,
    server: dict | None = None,
    extra: dict | None = None,
) -> dict:
    out: dict = {"path": path}
    if server:
        out["server"] = dict(server)
    if extra:
        out.update(extra)
    return out


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Send a message, get the assistant's reply. Runs routing + tools + LLM on this PC."""
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must be non-empty")
    try:
        if looks_like_spotify_pause_request(message):
            ok, detail, dev = await asyncio.to_thread(pause_spotify_playback)
            if ok:
                return ChatResponse(
                    reply="Paused Spotify playback.",
                    spotify=SpotifyPlayPayload(ok=True, kind="control", device_id=dev),
                    timing=_merge_chat_timing(path="spotify_pause"),
                )
            return ChatResponse(
                reply=_sanitize_app_reply(f"I couldn't pause Spotify playback: {detail or 'unknown error'}"),
                spotify=SpotifyPlayPayload(ok=False, kind="control", device_id=dev, playback_error=detail),
                timing=_merge_chat_timing(path="spotify_pause"),
            )

        # Spotify before generic \"Play …\" so voice/phone requests can still work.
        explicit_spotify = "spotify" in (message or "").lower()
        if looks_like_spotify_play_request(message):
            t0 = time.perf_counter()
            sp_res = await asyncio.to_thread(resolve_spotify_play, message)
            resolve_ms = (time.perf_counter() - t0) * 1000
            # Soft fallback: if Spotify didn't start and user didn't explicitly ask for Spotify, use YouTube.
            if (not sp_res.ok or not sp_res.playback_started) and not explicit_spotify and looks_like_play_music_request(message):
                yt_res = await asyncio.to_thread(resolve_play_to_video_id, message)
                yt_reply = format_play_resolution_reply(yt_res)
                yt_payload = _music_search_payload(yt_res)
                return ChatResponse(
                    reply=_sanitize_app_reply(yt_reply),
                    music=yt_payload,
                    timing=_merge_chat_timing(path="youtube_fallback", extra={"resolve_ms": round(resolve_ms, 1)}),
                )
            reply = format_spotify_resolution_reply(sp_res)
            return ChatResponse(
                reply=_sanitize_app_reply(reply),
                spotify=_spotify_play_payload(sp_res),
                timing=_merge_chat_timing(path="spotify", extra={"resolve_ms": round(resolve_ms, 1)}),
            )
        # YouTube \"Play …\" → search → video_id (Pi / player hooks in later).
        if looks_like_play_music_request(message):
            t0 = time.perf_counter()
            res = await asyncio.to_thread(resolve_play_to_video_id, message)
            resolve_ms = (time.perf_counter() - t0) * 1000
            reply = format_play_resolution_reply(res)
            return ChatResponse(
                reply=_sanitize_app_reply(reply),
                music=_music_search_payload(res),
                timing=_merge_chat_timing(path="youtube", extra={"resolve_ms": round(resolve_ms, 1)}),
            )

        # D&D safety guard: prefer the dedicated D&D page for campaign Q&A.
        t = message.lower()
        dnd_terms = ("players", "dm", "table", "campaign", "session", "fort", "commander", "quest", "solria", "ter")
        lighting_terms = ("lights", "nanoleaf", "govee", "brightness", "color", "animation", "scene")
        if any(k in t for k in dnd_terms) and not any(k in t for k in lighting_terms):
            return ChatResponse(
                reply="For questions about your D&D campaign, open /dnd and use the question box (it searches your campaign notes).",
                nanoleaf_flow=None,
                timing=_merge_chat_timing(path="dnd_guard"),
            )

        # Custom / flow animations: never instant-ack (force_full is redundant with _use_background_lighting_ack).
        force_full = _needs_full_lighting_chat_sync(message)
        exp = _lighting_alias_expansion(message)
        if exp:
            force_full = force_full or _needs_full_lighting_chat_sync(exp)

        is_bg_light, alias_key = _looks_like_background_lighting_action(message)
        if is_bg_light and _use_background_lighting_ack(message) and not force_full:
            threading.Thread(
                target=_run_background_action,
                args=(message,),
                daemon=True,
                name="galadrial-bg-action",
            ).start()
            return ChatResponse(
                reply=_ack_for_lighting_action_with_context(message, alias_key),
                nanoleaf_flow=None,
                timing=_merge_chat_timing(path="lighting_instant_ack"),
            )

        effective_name = (body.user_name or "").strip() or "Guest"
        req_started = time.time()
        timing_out: dict = {}
        reply = handle_message(
            message,
            log_fn=_log_event,
            user_name=effective_name,
            timing_out=timing_out,
        )
        pending_failure = _pop_async_failure()
        if pending_failure:
            reply = f"{pending_failure}\n\n{reply}" if reply else pending_failure
        nf = _nanoleaf_flow_fresh_since(req_started)
        reply = _sanitize_app_reply(reply or "")
        return ChatResponse(
            reply=reply or "No response.",
            nanoleaf_flow=nf,
            timing=_merge_chat_timing(path="chat", server=timing_out or None),
        )
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/music/prepare", response_model=MusicPrepareResponse)
async def music_prepare(body: MusicPrepareRequest):
    """
    Resolve \"Play …\" → YouTube search → ``video_id``. Same as POST /chat for play phrases,
    without the rest of routing. (HTTP call to your Pi is not wired here yet.)
    """
    phrase = (body.phrase or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="phrase must be non-empty")
    if not looks_like_play_music_request(phrase):
        raise HTTPException(
            status_code=400,
            detail='phrase must look like a play request (e.g. "Play artist name")',
        )
    try:
        res = await asyncio.to_thread(resolve_play_to_video_id, phrase)
        return MusicPrepareResponse(
            reply=_sanitize_app_reply(format_play_resolution_reply(res)),
            music=_music_search_payload(res),
        )
    except Exception as e:
        logger.exception("music/prepare failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/music/spotify/devices")
async def music_spotify_devices():
    """
    List Spotify Connect devices (find your Pi / librespot ``id`` for SPOTIFY_DEVICE_ID).
    Requires Spotify app credentials and a completed login (token cache). Does not start OAuth
    in the browser (that would block this request); run ``python -m music.spotify_auth`` once first.
    """
    if not spotify_credentials_configured():
        return {"configured": False, "authorized": False, "devices": [], "hint": "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."}
    if not spotify_has_cached_token():
        return {
            "configured": True,
            "authorized": False,
            "devices": [],
            "hint": "Run once on this PC: python -m music.spotify_auth (then reload this page).",
        }
    try:
        devices = await asyncio.to_thread(list_connect_devices)
        return {"configured": True, "authorized": True, "devices": devices}
    except SpotifyAuthRequiredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SpotifyNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("music/spotify/devices failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/music/spotify/prepare", response_model=SpotifyPrepareResponse)
async def music_spotify_prepare(body: SpotifyPrepareRequest):
    """
    Resolve a Spotify-specific play phrase → search + optional Connect playback (same as /chat branch).
    """
    phrase = (body.phrase or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="phrase must be non-empty")
    if not looks_like_spotify_play_request(phrase):
        raise HTTPException(
            status_code=400,
            detail='phrase must look like a Spotify request, e.g. "Play chill metal on Spotify" or "Spotify play …"',
        )
    try:
        res = await asyncio.to_thread(
            resolve_spotify_play,
            phrase,
            skip_llm_refinement=body.skip_llm_refinement,
            attempt_playback=body.attempt_playback,
            device_id=body.device_id,
        )
        return SpotifyPrepareResponse(
            reply=_sanitize_app_reply(format_spotify_resolution_reply(res)),
            spotify=_spotify_play_payload(res),
        )
    except Exception as e:
        logger.exception("music/spotify/prepare failed")
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
    #profileWrap { padding: 0 1rem 0.5rem; font-size: 0.78rem; color: #9a9fb2; }
    #profileBar { display: flex; height: 10px; border-radius: 4px; overflow: hidden; background: #222733; margin-top: 0.35rem; }
    #profileLegend { margin-top: 0.35rem; font-family: ui-monospace, monospace; font-size: 0.72rem; line-height: 1.45; color: #cdd0e0; }
  </style>
</head>
<body>
  <h1>Galadrial</h1>
  <div id="profileWrap" style="display: none;">
    <label style="display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.25rem;">
      <input type="checkbox" id="showProfile" checked />
      <span>Response profile</span>
    </label>
    <div id="profileBar"></div>
    <div id="profileLegend"></div>
  </div>
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
    const profileWrap = document.getElementById('profileWrap');
    const showProfile = document.getElementById('showProfile');
    const profileBar = document.getElementById('profileBar');
    const profileLegend = document.getElementById('profileLegend');
    let _lastProfileContext = { serverTiming: null, local: {} };

    function _ms(n) {
      if (n == null || n === undefined || Number.isNaN(Number(n))) return null;
      const x = Math.round(Number(n));
      return x < 0 ? null : x;
    }

    function renderResponseProfile(serverTiming, localExtra) {
      if (!showProfile || !showProfile.checked || !profileBar || !profileLegend) {
        if (profileWrap) profileWrap.style.display = 'none';
        return;
      }
      const s = (serverTiming && serverTiming.server) || {};
      const loc = localExtra || {};
      const segs = [];
      const rec = _ms(loc.recording_ms);
      const tr = _ms(loc.transcribe_ms);
      const chat = _ms(loc.chat_fetch_ms);
      const tts = _ms(loc.tts_ms);
      const rtr = _ms(s.router_llm_ms);
      const rep = _ms(s.reply_llm_ms);
      const oth = _ms(s.other_ms);
      const tot = _ms(s.total_ms);
      if (rec != null) segs.push({ key: 'RECORD', ms: rec, color: '#4a90d9' });
      if (tr != null) segs.push({ key: 'TRANSCRIBE', ms: tr, color: '#9b59b6' });
      if (serverTiming && serverTiming.resolve_ms != null) {
        const rm = _ms(serverTiming.resolve_ms);
        if (rm != null) segs.push({ key: 'RESOLVE', ms: rm, color: '#3498db' });
      }
      if (rtr != null && rtr > 0) segs.push({ key: 'ROUTER_LLM', ms: rtr, color: '#2ecc71' });
      if (rep != null && rep > 0) segs.push({ key: 'REPLY_LLM', ms: rep, color: '#1abc9c' });
      if (oth != null && oth > 0) segs.push({ key: 'OTHER', ms: oth, color: '#7f8c8d' });
      if (tts != null) segs.push({ key: 'TTS', ms: tts, color: '#e8a088' });
      const sum = segs.reduce((a, b) => a + b.ms, 0);
      if (!sum && tot == null && !(serverTiming && serverTiming.path)) {
        if (profileWrap) profileWrap.style.display = 'none';
        return;
      }
      profileWrap.style.display = 'block';
      profileBar.innerHTML = '';
      if (sum > 0) {
        segs.forEach((seg) => {
          const d = document.createElement('div');
          d.style.width = ((seg.ms / sum) * 100).toFixed(2) + '%';
          d.style.background = seg.color;
          d.style.minWidth = seg.ms > 0 ? '3px' : '0';
          d.title = seg.key + ': ' + seg.ms + 'ms';
          profileBar.appendChild(d);
        });
      }
      let leg = '';
      segs.forEach((seg) => {
        leg += '\\u25a0 ' + seg.key + ' ' + seg.ms + 'ms  ';
      });
      if (chat != null) leg += '/chat RTT (client) ' + chat + 'ms  ';
      if (tot != null) leg += ' | SERVER total ' + tot + 'ms';
      if (loc.transcribe_server_ms != null) leg += ' | Whisper (server) ' + Math.round(loc.transcribe_server_ms) + 'ms';
      if (serverTiming && serverTiming.path) leg += ' | path: ' + serverTiming.path;
      profileLegend.textContent = leg.trim();
    }

    if (showProfile) {
      showProfile.addEventListener('change', () => {
        renderResponseProfile(_lastProfileContext.serverTiming, _lastProfileContext.local);
      });
    }

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
        utter.onstart = function () { utter._t0 = performance.now(); };
        utter.onend = function () {
          if (utter._t0 != null) {
            _lastProfileContext.local.tts_ms = performance.now() - utter._t0;
            renderResponseProfile(_lastProfileContext.serverTiming, _lastProfileContext.local);
          }
        };
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

    function append(sender, text, isError, skipTts) {
      const div = document.createElement('div');
      div.className = sender + (isError ? ' error' : '');
      div.textContent = (sender === 'you' ? 'You: ' : 'Galadrial: ') + text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
      if (sender === 'assistant' && !isError && !skipTts) speak(text);
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = '';
      append('you', message);
      sendBtn.disabled = true;
      try {
        const tChat0 = performance.now();
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const chatFetchMs = performance.now() - tChat0;
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          append('assistant', data.detail || r.statusText || 'Request failed', true);
          return;
        }
        const localExtra = { chat_fetch_ms: chatFetchMs };
        if (window.__voiceSeg) {
          Object.assign(localExtra, window.__voiceSeg);
          window.__voiceSeg = null;
        }
        _lastProfileContext = { serverTiming: data.timing || null, local: localExtra };
        renderResponseProfile(data.timing, localExtra);
        append('assistant', data.reply || 'No response.');
        if (data.music && data.music.video_id) {
          append('assistant', 'YouTube video id: ' + data.music.video_id, false, true);
        }
        if (data.spotify && data.spotify.uri) {
          const s = data.spotify;
          let line = 'Spotify ' + (s.kind || 'track') + ': ' + (s.name || s.uri);
          if (s.playback_started) line += ' (playback started)';
          else if (s.playback_error) line += ' (playback: ' + s.playback_error + ')';
          if (s.device_id) line += ' [device ' + s.device_id + ']';
          append('assistant', line, false, true);
        }
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
      const t0 = performance.now();
      const r = await fetch('/stt', { method: 'POST', body: fd });
      const wallMs = performance.now() - t0;
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || r.statusText || 'STT failed');
      return {
        transcript: (data.transcript || '').trim(),
        transcribe_ms: wallMs,
        transcribe_server_ms: data.transcribe_server_ms,
      };
    }

    async function _startRecording() {
      input.value = '';
      micBtn.disabled = true;
      sendBtn.disabled = true;
      micBtn.textContent = 'Listening...';
      window.__recordStart = performance.now();

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
          const recMs = window.__recordStart != null ? (performance.now() - window.__recordStart) : null;
          window.__recordStart = null;
          const tr = await _transcribeBlob(blob, currentMimeType);
          const transcript = tr.transcript;
          window.__voiceSeg = {
            recording_ms: recMs,
            transcribe_ms: tr.transcribe_ms,
            transcribe_server_ms: tr.transcribe_server_ms,
          };
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
    .topRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1rem 0;
    }
    h1 { margin: 0; font-size: 1.25rem; color: #f0b34a; }
    .controls { display: flex; gap: 0.5rem; padding: 0 1rem; margin-bottom: 0.5rem; }
    .iconBtn {
      background: #151922;
      color: #f0b34a;
      border: 1px solid #222733;
      padding: 0.5rem 0.7rem;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 700;
    }
    .iconBtn:hover { background: #1a1f2a; }
    #authBox {
      margin: 0.5rem 1rem 0.75rem;
      padding: 0.75rem;
      background: #11141a;
      border: 1px solid #222733;
      border-radius: 8px;
      display: none;
    }
    #dndPwdInput {
      width: 100%;
      margin-top: 0.35rem;
      padding: 0.75rem 0.9rem;
      border: 1px solid #222733;
      border-radius: 8px;
      background: #151922;
      color: #f7f7f7;
      font-size: 1rem;
    }
    #dndPwdInput:focus { outline: none; border-color: #f0b34a; }
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
      min-height: 60px;
      padding: 0.75rem;
      margin: 0.5rem 1rem 1rem;
      background: #11141a;
      border: 1px solid #222733;
      border-radius: 8px;
      color: #f7f7f7;
      font-size: 0.95rem;
      resize: vertical;
    }
    #transcript:focus { outline: none; border-color: #f0b34a; }
    #question {
      flex: 1;
      min-height: 60px;
      padding: 0.75rem;
      margin: 0.5rem 1rem 1rem;
      background: #11141a;
      border: 1px solid #222733;
      border-radius: 8px;
      color: #f7f7f7;
      font-size: 0.95rem;
      resize: vertical;
    }
    #question:focus { outline: none; border-color: #f0b34a; }
    #reply {
      padding: 1rem;
      margin: 0 1rem 1rem;
      background: #151922;
      border-radius: 8px;
      border-left: 4px solid #f0b34a;
      font-size: 1rem;
      line-height: 1.5;
      min-height: 140px;
    }
    #reply.empty { color: #9a9fb2; }
    .error { color: #e55; }
  </style>
</head>
<body>
  <div class="topRow">
    <h1>D&D Improv</h1>
    <button type="button" id="unlockBtn" class="iconBtn" title="Unlock D&D">&#128274;</button>
  </div>
  <div id="authBox">
    <label for="dndPwdInput">Enter D&D password</label>
    <input type="password" id="dndPwdInput" placeholder="Password" />
    <div class="controls" style="padding: 0; margin-top: 0.5rem;">
      <button type="button" id="authSubmitBtn">Unlock</button>
      <button type="button" id="authCancelBtn" class="stop">Cancel</button>
    </div>
    <div id="authMsg" class="error" style="min-height: 1.2rem;"></div>
  </div>
  <div class="controls">
    <button type="button" id="recordBtn">Record</button>
    <button type="button" id="stopBtn" disabled class="stop">Stop</button>
    <button type="button" id="sendBtn" disabled>Get suggestion</button>
  </div>
  <label for="transcript">Transcript (editable)</label>
  <textarea id="transcript" placeholder="Record or paste the recent table conversation..."></textarea>
  <label for="question">DM question (editable)</label>
  <textarea id="question" placeholder="e.g. Who was the commander the players encountered in a fort in Solria Ter?..."></textarea>
  <div class="controls" style="margin-top: 0.25rem;">
    <button type="button" id="recordQBtn">Record question</button>
    <button type="button" id="stopQBtn" disabled class="stop">Stop</button>
  </div>
  <div class="controls" style="margin-top: 0.25rem;">
    <button type="button" id="askBtn" disabled>Ask (use campaign notes)</button>
  </div>
  <label>Read this aloud</label>
  <div id="reply" class="empty">Suggestions will appear here after you send the transcript.</div>
  <script>
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const sendBtn = document.getElementById('sendBtn');
    const recordQBtn = document.getElementById('recordQBtn');
    const stopQBtn = document.getElementById('stopQBtn');
    const askBtn = document.getElementById('askBtn');
    const unlockBtn = document.getElementById('unlockBtn');
    const authBox = document.getElementById('authBox');
    const dndPwdInput = document.getElementById('dndPwdInput');
    const authSubmitBtn = document.getElementById('authSubmitBtn');
    const authCancelBtn = document.getElementById('authCancelBtn');
    const authMsgEl = document.getElementById('authMsg');
    const transcript = document.getElementById('transcript');
    const questionEl = document.getElementById('question');
    const replyEl = document.getElementById('reply');

    let isAuthed = false;

    function setAuthed(ok) {
      isAuthed = !!ok;
      sendBtn.disabled = !isAuthed;
      askBtn.disabled = !isAuthed;
      if (isAuthed) {
        authBox.style.display = 'none';
        unlockBtn.style.display = 'none';
        authMsgEl.textContent = '';
      } else {
        unlockBtn.style.display = 'inline-block';
      }
    }

    async function refreshAuth() {
      try {
        const r = await fetch('/dnd/auth/status', { credentials: 'same-origin' });
        const data = await r.json().catch(() => ({}));
        setAuthed(!!data.authed);
      } catch (e) {
        setAuthed(false);
      }
    }

    unlockBtn.addEventListener('click', () => {
      authMsgEl.textContent = '';
      authBox.style.display = 'block';
      setTimeout(() => { dndPwdInput.focus(); }, 0);
    });

    authCancelBtn.addEventListener('click', () => {
      authBox.style.display = 'none';
      authMsgEl.textContent = '';
    });

    authSubmitBtn.addEventListener('click', async () => {
      const pwd = (dndPwdInput.value || '').toString();
      authMsgEl.textContent = '';
      try {
        const r = await fetch('/dnd/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ password: pwd }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          authMsgEl.textContent = data.detail || 'Incorrect password';
          setAuthed(false);
          return;
        }
        dndPwdInput.value = '';
        setAuthed(true);
        authBox.style.display = 'none';
      } catch (e) {
        authMsgEl.textContent = 'Unlock failed.';
        setAuthed(false);
      }
    });

    refreshAuth();

    // Whisper STT: browser records audio, server transcribes, then we fill the editable textarea.
    let mediaRecorder = null;
    let micStream = null;
    let audioChunks = [];
    let currentMimeType = '';
    let mediaRecorderQ = null;
    let micStreamQ = null;
    let audioChunksQ = [];
    let currentMimeTypeQ = '';

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
      return {
        transcript: (data.transcript || '').trim(),
        transcribe_server_ms: data.transcribe_server_ms,
      };
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
              const tr = await _transcribeBlob(blob, currentMimeType);
              transcript.value = tr.transcript;
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

      recordQBtn.addEventListener('click', async () => {
        try {
          recordQBtn.disabled = true;
          stopQBtn.disabled = false;
          replyEl.textContent = 'Recording question...';
          replyEl.classList.remove('empty');
          replyEl.classList.remove('error');

          questionEl.value = questionEl.value || '';
          audioChunksQ = [];
          currentMimeTypeQ = _pickMimeType();

          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          micStreamQ = stream;
          const options = {};
          if (currentMimeTypeQ) options.mimeType = currentMimeTypeQ;
          mediaRecorderQ = new MediaRecorder(stream, options);
          mediaRecorderQ.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) audioChunksQ.push(e.data);
          };
          mediaRecorderQ.onstop = async () => {
            try {
              if (micStreamQ) micStreamQ.getTracks().forEach((t) => t.stop());
              const blob = new Blob(audioChunksQ, { type: currentMimeTypeQ || 'audio/webm' });
              if (!blob.size) return;
              const tr = await _transcribeBlob(blob, currentMimeTypeQ);
              questionEl.value = tr.transcript;
              replyEl.textContent = 'Question transcript ready.';
              replyEl.classList.remove('error');
            } catch (e) {
              replyEl.textContent = 'Question transcription failed: ' + ((e && e.message) ? e.message : String(e));
              replyEl.classList.add('error');
            } finally {
              recordQBtn.disabled = false;
              stopQBtn.disabled = true;
              mediaRecorderQ = null;
              micStreamQ = null;
              audioChunksQ = [];
              currentMimeTypeQ = '';
            }
          };
          mediaRecorderQ.start();
        } catch (e) {
          recordQBtn.disabled = false;
          stopQBtn.disabled = true;
          replyEl.textContent = 'Question mic start failed: ' + ((e && e.message) ? e.message : String(e));
          replyEl.classList.add('error');
        }
      });

      stopQBtn.addEventListener('click', () => {
        if (mediaRecorderQ && mediaRecorderQ.state === 'recording') {
          mediaRecorderQ.stop();
        }
      });
    } else {
      recordBtn.textContent = 'Record (unsupported)';
      recordBtn.disabled = true;
      stopBtn.disabled = true;
      recordQBtn.textContent = 'Record question (unsupported)';
      recordQBtn.disabled = true;
      stopQBtn.disabled = true;
    }

    sendBtn.addEventListener('click', async () => {
      if (!isAuthed) {
        authMsgEl.textContent = '';
        authBox.style.display = 'block';
        setTimeout(() => { dndPwdInput.focus(); }, 0);
        return;
      }
      const text = transcript.value.trim();
      if (!text) return;
      sendBtn.disabled = true;
      replyEl.textContent = 'Thinking...';
      replyEl.classList.remove('empty');
      try {
        const r = await fetch('/dnd-improv', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
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

    askBtn.addEventListener('click', async () => {
      if (!isAuthed) {
        authMsgEl.textContent = '';
        authBox.style.display = 'block';
        setTimeout(() => { dndPwdInput.focus(); }, 0);
        return;
      }
      const q = (questionEl.value || '').trim();
      const tr = (transcript.value || '').trim();
      if (!q) return;
      askBtn.disabled = true;
      replyEl.textContent = 'Thinking...';
      replyEl.classList.remove('empty');
      try {
        const r = await fetch('/dnd/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ question: q, transcript: tr || null })
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
        askBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
