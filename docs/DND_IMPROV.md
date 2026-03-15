# D&D Improv Assistant

Browser-based tool: record table conversation (mic), transcribe in the browser, send the transcript to the API; the server loads your campaign notes and maps and returns suggested dialogue to read aloud.

## Running

1. Start LM Studio with a vision-capable model (e.g. qwen3-vl-4b) on `localhost:1234`.
2. Start the API server: `uvicorn api_server:app --host 0.0.0.0 --port 8000`
3. Open **http://localhost:8000/dnd** in a browser.

## Campaign folder (notes and maps)

- **Default path:** `C:\Users\awest\Documents\DnD Campaigns`
- **Override:** set the environment variable **`DND_CONTEXT_DIR`** to a different folder path (e.g. on another machine or for a different campaign).

The server reads from this directory:

- **Text notes:** `.md`, `.txt`, `.odt` — chunked into ~600‑char pieces. The **LLM** decides what’s relevant: it sees the transcript plus a **catalog** (chunk number + short preview, and map filenames) and replies with e.g. `CHUNKS: 0, 2, 5` and `MAPS: tavern.png`. Only those chunks and maps are sent in the second call that produces the dialogue.
- **Maps:** `.png`, `.jpg`, `.jpeg`, `.webp` — the same catalog lists map filenames; the LLM includes whichever are relevant in its `MAPS: ...` line.

**Two-call flow:** (1) **Selection:** transcript + catalog → text-only LLM call → parsed `CHUNKS:` and `MAPS:`. (2) **Dialogue:** transcript + full text of selected chunks + selected map images → vision LLM → suggested dialogue. So the **LLM** chooses relevance; the Python layer only builds the catalog and assembles the final prompt from the LLM’s choices.

If the folder is missing or empty, the app still works; the dialogue call runs with no campaign context.
