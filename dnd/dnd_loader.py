"""
Load D&D campaign context: text notes (.md, .txt, .odt) and map images (.png, .jpg, .jpeg, .webp).
Supports full load (load_campaign_context) or RAG (get_rag_context) to send only relevant chunks + selected maps.
Default path: C:\\Users\\awest\\Documents\\DnD Campaigns (override with DND_CONTEXT_DIR).
"""

import base64
import os
from pathlib import Path
from typing import NamedTuple
import requests

# Optional: "map_descriptions.txt" in campaign folder, format "filename.png: keyword1 keyword2"
MAP_DESCRIPTIONS_FILENAME = "map_descriptions.txt"
# RAG: chunk size and overlap for text
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
# Default number of text chunks to retrieve per request
DEFAULT_TOP_K_TEXT = 10
# Default max maps to send per request
DEFAULT_MAX_MAPS = 2
# Preview length per chunk when building catalog for LLM selection
CATALOG_PREVIEW_CHARS = 180

# Module-level cache for RAG index (path -> (chunks_with_sources, embeddings))
_rag_cache: dict = {}
_embedding_model = None

# LM Studio embeddings (fixed for now).
# NOTE: This uses your already-downloaded model: `text-embedding-bge-small-en-v1.5`.
LMSTUDIO_EMBEDDINGS_MODEL = "text-embedding-bge-small-en-v1.5"
# LM Studio expects OpenAI-compatible embeddings at this route.
LMSTUDIO_EMBEDDINGS_URL = "http://localhost:1234/v1/embeddings"

# Default campaign folder; override with env DND_CONTEXT_DIR
DND_CONTEXT_DIR = os.environ.get(
    "DND_CONTEXT_DIR",
    r"C:\Users\awest\Documents\DnD Campaigns",
)
TEXT_EXTENSIONS = {".md", ".txt", ".odt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class CampaignContext(NamedTuple):
    """Loaded campaign context: text notes and map image paths (or base64 data_urls)."""

    notes_text: str
    image_paths: list[str]


class SelectionCatalog(NamedTuple):
    """
    Catalog for LLM to choose relevant chunks and maps.
    catalog_text: human-readable list of chunk previews and map names.
    chunks_with_sources: full (chunk, source_name) list so API can resolve selected indices.
    image_paths: full paths so API can resolve selected map filenames.
    """

    catalog_text: str
    chunks_with_sources: list[tuple[str, str]]
    image_paths: list[str]


def _read_odt(path: Path) -> str:
    """Extract plain text from an .odt file using odfpy."""
    try:
        from odf import text, teletype
        from odf.opendocument import load

        doc = load(str(path))
        parts = []
        for el in doc.getElementsByType(text.P):
            parts.append(teletype.extractText(el))
        return "\n".join(p for p in parts if p.strip()).strip()
    except Exception:
        return ""


def load_campaign_context(root_path: str | Path | None = None) -> CampaignContext:
    """
    Load all text notes and map image paths from the campaign folder.
    Returns CampaignContext(notes_text, image_paths). Handles missing/empty folder.
    """
    root = Path(root_path or DND_CONTEXT_DIR)
    notes_parts: list[str] = []
    image_paths: list[str] = []

    if not root.is_dir():
        return CampaignContext(notes_text="No notes loaded (campaign folder not found or not a directory).", image_paths=[])

    # Sort for deterministic order
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return CampaignContext(notes_text="No notes loaded (could not read campaign folder).", image_paths=[])

    for path in entries:
        if path.is_file():
            suf = path.suffix.lower()
            if suf in TEXT_EXTENSIONS:
                try:
                    if suf == ".odt":
                        content = _read_odt(path)
                    else:
                        content = path.read_text(encoding="utf-8", errors="replace")
                    notes_parts.append(f"--- {path.name} ---\n{content.strip()}")
                except Exception:
                    notes_parts.append(f"--- {path.name} ---\n[Could not read file]")
            elif suf in IMAGE_EXTENSIONS:
                image_paths.append(str(path.resolve()))

    notes_text = "\n\n".join(notes_parts) if notes_parts else "No notes loaded (no text files in campaign folder)."
    return CampaignContext(notes_text=notes_text, image_paths=image_paths)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for RAG."""
    if not text.strip():
        return []
    chunks = []
    # Prefer splitting on paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = []
    current_len = 0
    for p in paragraphs:
        if current_len + len(p) + 2 > chunk_size and current:
            chunks.append("\n\n".join(current))
            # overlap: keep last paragraph(s) that fit in overlap
            overlap_len = 0
            keep = []
            for x in reversed(current):
                if overlap_len + len(x) + 2 <= overlap:
                    keep.append(x)
                    overlap_len += len(x) + 2
                else:
                    break
            current = list(reversed(keep))
            current_len = sum(len(x) for x in current) + 2 * (len(current) - 1) if current else 0
        current.append(p)
        current_len += len(p) + (2 if current_len else 0)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def _extract_embeddings_from_lmstudio_response(resp_json: dict, expected_count: int) -> list[list[float]]:
    """
    Best-effort extraction for OpenAI-compatible embedding responses.
    Supports a few common response shapes.
    """
    # OpenAI-ish: {"data":[{"embedding":[...],"index":0}, ...]}
    data = resp_json.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        embeddings: list[list[float]] = []
        for item in data:
            emb = item.get("embedding")
            if isinstance(emb, list):
                embeddings.append([float(x) for x in emb])
        if len(embeddings) == expected_count:
            return embeddings

    # Some servers: {"embeddings":[[...],[...]]}
    embs = resp_json.get("embeddings")
    if isinstance(embs, list) and embs and isinstance(embs[0], list):
        embeddings = [[float(x) for x in emb] for emb in embs]
        if len(embeddings) == expected_count:
            return embeddings

    # Fallback: sometimes wrapped under "output"
    out = resp_json.get("output")
    if isinstance(out, list) and out and isinstance(out[0], dict):
        # Potential shapes:
        # - output[0].embedding as a list of floats (single input)
        # - output[i].embedding for each input
        embeddings: list[list[float]] = []
        for item in out:
            emb = item.get("embedding")
            if isinstance(emb, list):
                embeddings.append([float(x) for x in emb])
        if len(embeddings) == expected_count:
            return embeddings

    raise ValueError(f"LM Studio embeddings response shape not recognized (expected {expected_count} embeddings).")


def _lmstudio_embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings using LM Studio's embeddings endpoint.
    Returns list of embedding vectors aligned with input `texts`.
    """
    if not texts:
        return []

    payload = {
        "model": LMSTUDIO_EMBEDDINGS_MODEL,
        "input": texts,
        "store": False,
    }
    r = requests.post(LMSTUDIO_EMBEDDINGS_URL, json=payload, timeout=120)
    r.raise_for_status()
    resp_json = r.json() if r.text else {}
    return _extract_embeddings_from_lmstudio_response(resp_json, expected_count=len(texts))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings using LM Studio only (no fallback).
    """
    return _lmstudio_embed_batch(texts)


def _load_all_note_chunks(root: Path) -> list[tuple[str, str]]:
    """Load all text files and return list of (chunk, source_name)."""
    out: list[tuple[str, str]] = []
    entries = sorted(root.iterdir(), key=lambda p: p.name.lower()) if root.is_dir() else []
    for path in entries:
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            if path.suffix.lower() == ".odt":
                content = _read_odt(path)
            else:
                content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for chunk in _chunk_text(content):
            if chunk.strip():
                out.append((chunk.strip(), path.name))
    return out


def _build_rag_index(root: Path):
    """Build or return cached embedding index for the campaign folder."""
    root_str = str(root.resolve())
    cache_key = f"{root_str}|lmstudio|model={LMSTUDIO_EMBEDDINGS_MODEL}|url={LMSTUDIO_EMBEDDINGS_URL}"
    if cache_key in _rag_cache:
        return _rag_cache[cache_key]
    chunks_with_sources = _load_all_note_chunks(root)
    if not chunks_with_sources:
        _rag_cache[cache_key] = ([], [])
        return ([], [])
    chunks_only = [c for c, _ in chunks_with_sources]

    # Batch embedding requests to reduce LM Studio overhead and avoid giant payloads.
    # (Works for both LM Studio and local models.)
    batch_size = 32
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks_only), batch_size):
        batch = chunks_only[start : start + batch_size]
        embeddings.extend(_embed_texts(batch))

    _rag_cache[cache_key] = (chunks_with_sources, embeddings)
    return (chunks_with_sources, embeddings)


def _cosine_similarity(a: list[float], b: list[list[float]]) -> list[float]:
    """Return cosine similarity between vector a and each row of b."""
    import math
    import numpy as np
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    va = va / (np.linalg.norm(va) + 1e-9)
    vb = vb / (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-9)
    return (vb @ va).tolist()


def _select_maps_for_transcript(
    transcript: str, image_paths: list[str], root: Path, max_maps: int
) -> list[str]:
    """
    Select up to max_maps whose keywords (from map_descriptions.txt) appear in transcript.
    If no map_descriptions.txt, return [] to keep the call small.
    """
    if not image_paths or max_maps <= 0:
        return []
    desc_path = root / MAP_DESCRIPTIONS_FILENAME
    if not desc_path.is_file():
        return []
    try:
        lines = desc_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except Exception:
        return []
    # Parse "filename.png: keyword1 keyword2"
    name_to_keywords: dict[str, list[str]] = {}
    for line in lines:
        if ":" in line:
            name, _, kw = line.partition(":")
            name = name.strip().lower()
            keywords = [w.strip().lower() for w in kw.split() if w.strip()]
            if name and keywords:
                name_to_keywords[name] = keywords
    transcript_lower = transcript.lower()
    scored: list[tuple[int, str]] = []
    for path_str in image_paths:
        path = Path(path_str)
        name = path.name.lower()
        keywords = name_to_keywords.get(name, [])
        if not keywords:
            continue
        score = sum(1 for k in keywords if k in transcript_lower)
        if score > 0:
            scored.append((score, path_str))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [path_str for _, path_str in scored[:max_maps]]


def get_selection_catalog(root_path: str | Path | None = None) -> SelectionCatalog | None:
    """
    Build a catalog of note chunk previews and map names for the LLM to choose from.
    Returns None if campaign folder missing or empty. Otherwise returns SelectionCatalog
    with catalog_text (for the selection prompt), full chunks_with_sources, and image_paths.
    """
    root = Path(root_path or DND_CONTEXT_DIR)
    if not root.is_dir():
        return None
    chunks_with_sources = _load_all_note_chunks(root)
    image_paths = [
        str(p.resolve())
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not chunks_with_sources and not image_paths:
        return None
    lines = []
    if chunks_with_sources:
        lines.append("NOTES (reply with CHUNKS: and a comma-separated list of chunk numbers, e.g. CHUNKS: 0, 2, 5)")
        for i, (chunk, source) in enumerate(chunks_with_sources):
            preview = (chunk[:CATALOG_PREVIEW_CHARS] + "..." if len(chunk) > CATALOG_PREVIEW_CHARS else chunk).replace("\n", " ")
            lines.append(f"  [{i}] (from {source}) {preview}")
    if image_paths:
        map_names = [Path(p).name for p in image_paths]
        lines.append("MAPS (reply with MAPS: and a comma-separated list of filenames, e.g. MAPS: tavern.png, or MAPS: none)")
        lines.append("  " + ", ".join(map_names))
    catalog_text = "\n".join(lines)
    return SelectionCatalog(
        catalog_text=catalog_text,
        chunks_with_sources=chunks_with_sources,
        image_paths=image_paths,
    )


def build_context_from_llm_selection(
    chunks_with_sources: list[tuple[str, str]],
    selected_chunk_indices: list[int],
    image_paths: list[str],
    selected_map_filenames: list[str],
) -> CampaignContext:
    """
    Build CampaignContext from LLM-selected chunk indices and map filenames.
    Invalid indices/filenames are skipped.
    """
    name_to_path = {Path(p).name: p for p in image_paths}
    selected_paths = [name_to_path[n] for n in selected_map_filenames if n in name_to_path]
    parts = []
    for i in selected_chunk_indices:
        if 0 <= i < len(chunks_with_sources):
            chunk, source = chunks_with_sources[i]
            parts.append(f"--- {source} ---\n{chunk}")
    notes_text = "\n\n".join(parts) if parts else "No notes selected."
    return CampaignContext(notes_text=notes_text, image_paths=selected_paths)


def _parse_llm_selection_response(response_text: str) -> tuple[list[int], list[str]]:
    """Parse CHUNKS: ... and MAPS: ... from LLM selection response. Returns (indices, map_filenames)."""
    import re
    indices: list[int] = []
    map_filenames: list[str] = []
    chunks_match = re.search(r"CHUNKS?\s*:\s*([0-9,\s]+)", response_text, re.IGNORECASE)
    if chunks_match:
        for s in chunks_match.group(1).split(","):
            s = s.strip()
            if s.isdigit():
                indices.append(int(s))
    maps_match = re.search(r"MAPS?\s*:\s*([^\n]+)", response_text, re.IGNORECASE)
    if maps_match:
        raw = maps_match.group(1).strip().lower()
        if raw not in ("none", "n/a", "-", ""):
            for s in raw.split(","):
                name = s.strip().strip('"\'')
                if name:
                    map_filenames.append(name)
    return (indices, map_filenames)


def get_rag_context(
    transcript: str,
    root_path: str | Path | None = None,
    top_k_text: int = DEFAULT_TOP_K_TEXT,
    max_maps: int = DEFAULT_MAX_MAPS,
) -> CampaignContext:
    """
    Retrieve only relevant note chunks and a small set of maps for the given transcript.
    Uses embeddings + cosine similarity for text; optional map_descriptions.txt for maps.
    Returns CampaignContext(notes_text, image_paths) with a small payload.
    """
    root = Path(root_path or DND_CONTEXT_DIR)
    if not root.is_dir():
        return CampaignContext(
            notes_text="No notes loaded (campaign folder not found).",
            image_paths=[],
        )
    # Build index and get embeddings
    chunks_with_sources, embeddings = _build_rag_index(root)
    if not chunks_with_sources:
        notes_text = "No notes loaded (no text files in campaign folder)."
        image_paths = []
    else:
        query_embedding = _embed_texts([transcript or "conversation"])[0]
        # Keep existing cosine similarity behavior (normalize inside helper).
        query_embedding = list(query_embedding)
        sims = _cosine_similarity(query_embedding, embeddings)
        top_indices = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k_text]
        parts = []
        seen = set()
        for i in top_indices:
            chunk, source = chunks_with_sources[i]
            key = (chunk[:50], source)
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"--- {source} ---\n{chunk}")
        notes_text = "\n\n".join(parts) if parts else "No matching notes."
        # Map paths: collect all for this folder, then select by transcript
        image_paths = [
            str(p.resolve())
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_paths = _select_maps_for_transcript(transcript, image_paths, root, max_maps)
    return CampaignContext(notes_text=notes_text, image_paths=image_paths)


def load_images_as_data_urls(image_paths: list[str], max_size_mb: float = 10.0) -> list[str]:
    """
    Read image files and return a list of data URLs (data:image/...;base64,...).
    Skips files larger than max_size_mb to avoid overloading the LLM request.
    """
    data_urls: list[str] = []
    max_bytes = int(max_size_mb * 1024 * 1024)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

    for path_str in image_paths:
        path = Path(path_str)
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf not in mime:
            continue
        try:
            raw = path.read_bytes()
            if len(raw) > max_bytes:
                continue
            b64 = base64.b64encode(raw).decode("ascii")
            data_urls.append(f"data:{mime[suf]};base64,{b64}")
        except Exception:
            continue
    return data_urls
