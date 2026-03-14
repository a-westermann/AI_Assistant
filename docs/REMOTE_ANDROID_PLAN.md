# Remote Android + FastAPI Plan (Tailscale)

Use Galadrial from your Android device: the app sends a message to a FastAPI server on your PC; the server runs routing, tools, and the LLM (LM Studio) and returns the assistant’s reply.

---

## 1. Architecture

```
┌─────────────────────┐         Tailscale           ┌──────────────────────────────────────┐
│  Android device     │ ◄─────────────────────────►│  PC (this machine)                   │
│                     │         private network     │                                      │
│  • Client app       │   POST /chat {"message": …}  │  • FastAPI server (e.g. :8000)       │
│  • Tailscale app    │   ◄── {"reply": "…"}        │  • LM Studio (localhost:1234)        │
│                     │                              │  • Gmail, lights, Plex sync (local)  │
└─────────────────────┘                              └──────────────────────────────────────┘
```

- **Android**: Only needs to HTTP POST the user message and show the reply. No LLM or tools on the phone.
- **PC**: Runs FastAPI (same process can also run the Tk desktop app, or you run the server separately). FastAPI calls LM Studio at `localhost:1234` and runs lights/Gmail/Plex tools on the PC.
- **Tailscale**: Both devices join the same Tailscale network. Android uses the PC’s Tailscale IP or hostname (e.g. `http://100.x.x.x:8000`) so no port forwarding or public IP is needed.

---

## 2. What Runs Where

| Component              | Where it runs | Notes |
|------------------------|---------------|--------|
| LM Studio              | PC            | Already `localhost:1234` |
| Gmail (IMAP)           | PC            | Env vars on PC |
| Lights API client      | PC            | Calls your Govee automation from PC |
| Plex sync script       | PC            | Path on PC |
| Routing + tools + LLM  | PC            | Inside FastAPI (reuse same logic as GUI) |
| FastAPI server         | PC            | Binds to `0.0.0.0` so Tailscale IP is reachable |
| Android client         | Phone         | Only HTTP + UI |

---

## 3. FastAPI Server (on PC)

**Role**: Single HTTP API that accepts a user message and returns the assistant’s reply after routing, optional tool runs, and one LLM call.

**Suggested endpoint**:

- `POST /chat`  
  - Body: `{"message": "are the lights on"}`  
  - Response: `{"reply": "Yes, the living room lights are on.", "action": "lights.get_state"}` (action optional, for debugging).

**Auth**: So only you (or your devices) can call it:

- **Option A**: API key in header, e.g. `X-API-Key: <secret>`. FastAPI checks it; you store the same secret in the Android app (or in a config you load).
- **Option B**: Rely on Tailscale (same Tailscale network = trusted). E.g. allow only requests from Tailscale IP range, or run the server bound to Tailscale IP only. Simpler but less secure if someone gets on your Tailscale.

**CORS**: If the Android “app” is a web page (PWA or in-browser), set CORS so that page can call the API. If it’s a native app doing HTTP, CORS doesn’t apply.

**Binding**: Start with `uvicorn` binding to `0.0.0.0:8000` so the PC is reachable on its Tailscale IP (e.g. `http://100.101.102.103:8000`).

---

## 4. Reusing Existing Logic (refactor)

Right now routing, tools, and the model call live inside `ChatApp` in `chat_gui.py` and use GUI bits (`_log_event`, `root.after`, `append_message`). To support FastAPI without duplicating logic:

**Option A – Extract an “engine” module (recommended)**  
- New module, e.g. `assistant_engine.py`, that has:
  - `route(user_text: str) -> dict`
  - `handle_message(user_text: str) -> str`  
    - Runs route → runs tool if needed (lights, Gmail, Plex) → builds prompt → calls `ask_lmstudio` → returns the single reply string.
- Tools stay in existing modules (`lights_client`, `gmail_client`); the engine imports them and calls them.
- “Log” output from the engine can go to a list or logger instead of `_log_event`; FastAPI can log the same to stdout.
- `chat_gui.py` then:
  - Imports the engine (or shares the same routing/handle logic).
  - Keeps UI: on “Send”, call the engine (or current logic), then `append_message("Assistant", reply)`.

**Option B – Minimal refactor**  
- FastAPI endpoint imports a single function, e.g. `handle_message(user_text: str) -> str`, that you copy/paste or extract from the current `_handle_user_message` + `call_model_and_display` flow, with GUI calls replaced by no-ops or a small logger. Quick but more duplication.

**Recommendation**: Option A so the desktop app and FastAPI share one implementation and stay in sync.

---

## 5. Android Client Options

**A. Simple web client (fastest)**  
- Single HTML page (or tiny Flask/FastAPI static page served from PC or anywhere) with:
  - Input, Send button, area for reply.
  - JS: `fetch("http://<PC_TAILSCALE_IP>:8000/chat", { method: "POST", body: JSON.stringify({ message }), headers: { "Content-Type": "application/json", "X-API-Key": "..." } })`.
- You open that page in Chrome on Android (or add to home screen as PWA). No Play Store, no Android project.

**B. Native Android app**  
- Small app (Kotlin/Compose or similar) that does the same HTTP POST and shows the reply. Better UX, offline handling, and you can store the API key more carefully (e.g. BuildConfig or a local config). More work.

**C. Existing terminal/API client**  
- Use a generic REST client (e.g. HTTPie, Postman) on Android to hit `POST /chat` for quick tests before building any UI.

Start with A or C to validate the API and Tailscale; add B if you want a dedicated app.

---

## 6. Tailscale Setup

- **PC**: Install Tailscale, log in. Note the machine’s Tailscale IP (e.g. `100.x.x.x`) or hostname (e.g. `my-pc.tailnet-name.ts.net`).
- **Android**: Install Tailscale from Play Store, same account (or same tailnet). No need to expose ports; the phone and PC can talk over Tailscale.
- **Firewall**: On PC, allow inbound TCP to the port FastAPI uses (e.g. 8000). Tailscale doesn’t change that.

**Testing**: From the Android browser or app, open `http://<PC_TAILSCALE_IP>:8000/health` (or similar); then POST to `/chat` with a test message.

---

## 7. Implementation Order

1. **Refactor (Option A)**  
   - Add `assistant_engine.py`: routing + handle (tools + single LLM call) returning reply string.  
   - Optionally have `chat_gui.py` call into this engine so the desktop app stays the single source of truth.

2. **FastAPI app**  
   - New file, e.g. `api_server.py` or `main.py` in an `api/` folder.  
   - `POST /chat`: read `message`, call engine’s `handle_message(message)`, return `{"reply": ...}`.  
   - Add API-key middleware or Tailscale-only check.  
   - Optional: `GET /health` for connectivity checks.

3. **Run server**  
   - `uvicorn api_server:app --host 0.0.0.0 --port 8000` (or configurable port).  
   - Ensure LM Studio is running on the PC.

4. **Android**  
   - Build the simple web page or native client that POSTs to `http://<PC_TAILSCALE_IP>:8000/chat` and displays the reply.

5. **Security**  
   - Add/confirm API key or Tailscale-based restriction.  
   - Prefer HTTPS if you later expose the server beyond Tailscale (e.g. with a reverse proxy or Tailscale HTTPS).

---

## 8. Summary

- **PC**: FastAPI server + LM Studio + existing tools (Gmail, lights, Plex). One “engine” used by both the desktop GUI and the API.
- **Android**: Thin client that sends the message and shows the reply over Tailscale.
- **Tailscale**: Private network so the phone can reach the PC without opening your home router.

Next concrete step: implement the engine refactor and a minimal `POST /chat` FastAPI endpoint, then test from the PC with `curl` before testing from Android.
