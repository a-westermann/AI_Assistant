import imaplib
import os
import re
from email import message_from_bytes
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Literal, Set


class GmailClientError(Exception):
    """Raised when there is a problem talking to Gmail over IMAP."""


def _require_config() -> tuple[str, str]:
    username = os.getenv("GMAIL_IMAP_USERNAME", "")
    password = os.getenv("GMAIL_IMAP_PASSWORD", "")
    if not username or not password:
        raise GmailClientError(
            "Gmail IMAP not configured. "
            "Set GMAIL_IMAP_USERNAME and GMAIL_IMAP_PASSWORD (app password)."
        )
    return username, password


def _get_body_snippet(msg: Any, max_chars: int = 400) -> str:
    """Extract plain-text body from a parsed email and return first max_chars as one line."""
    chunks: List[str] = []
    for part in msg.walk():
        if part.get_content_type() != "text/plain":
            continue
        try:
            payload = part.get_payload(decode=True)
            if payload:
                chunks.append(payload.decode("utf-8", errors="replace"))
        except Exception:
            continue
    if not chunks:
        return ""
    text = " ".join(" ".join(c.split()) for c in chunks)
    return text[:max_chars].strip()


def _decode_header(value: Any) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for text, enc in parts:
        if isinstance(text, bytes):
            try:
                decoded.append(text.decode(enc or "utf-8", errors="replace"))
            except Exception:
                decoded.append(text.decode("utf-8", errors="replace"))
        else:
            decoded.append(text)
    return "".join(decoded)


def _extract_search_keywords(prompt: str) -> str:
    """
    Build a simple keyword query from a natural-language prompt.

    We strip filler words and keep the most meaningful terms so Gmail can find
    relevant messages without requiring an exact long phrase.
    """
    stopwords = {
        "how",
        "many",
        "do",
        "i",
        "have",
        "any",
        "emails",
        "email",
        "message",
        "messages",
        "in",
        "my",
        "gmail",
        "inbox",
        "for",
        "the",
        "a",
        "an",
        "about",
        "new",
        "unread",
        "unreads",
        "recent",
        "are",
        "there",
        "you",
        "can",
        "check",
        "what",
        "whats",
        "last",
        "being",
        "that",
        "this",
        "from",
        "with",
    }
    tokens = []
    for raw in prompt.split():
        w = raw.strip(".,!?\"'()[]{}").lower()
        if not w or len(w) <= 2:
            continue
        if w in stopwords:
            continue
        tokens.append(w)

    if not tokens:
        return prompt
    # Use at most 4–5 keywords so the search isn't overly strict; Gmail matches
    # emails containing these terms (AND), which is more likely to find results.
    max_tokens = 5
    keywords = " ".join(tokens[:max_tokens]) if len(tokens) > max_tokens else " ".join(tokens)
    return keywords


def _is_pure_category_count_query(query: str, category: str) -> bool:
    """
    True if the query is only asking for a count in a category (e.g. unread in Updates),
    so we should not add extra TEXT terms and the count matches the tab badge.
    """
    kw = _extract_search_keywords(query).strip().lower()
    if not kw:
        return True
    category_lower = category.lower()
    # Don't add TEXT filter when the only keywords are category/tab/count wording
    allowed = {
        category_lower,
        "box",
        "tab",
        "inbox",
        "unread",
        "unreads",
        "conversation",
        "conversations",
        "new",
        "promotions",
        "primary",
        "social",
        "updates",
        "forums",
        f"{category_lower} box",
    }
    return kw in allowed or all(w in allowed for w in kw.split())


def _fetch_unique_thread_count(mail: imaplib.IMAP4_SSL, message_ids: List[bytes]) -> int:
    """
    Fetch X-GM-THRID for each message in batches and return the number of unique
    threads (conversations). This matches the number Gmail shows on the tab badge.
    """
    if not message_ids:
        return 0
    thread_ids: Set[str] = set()
    batch_size = 200
    # Match only the number inside (X-GM-THRID N) so we don't capture sequence numbers
    # from "* 12345 FETCH (X-GM-THRID ...)".
    thrid_re = re.compile(rb"\(X-GM-THRID\s+(\d+)\)", re.IGNORECASE)
    for i in range(0, len(message_ids), batch_size):
        batch = message_ids[i : i + batch_size]
        seq_set = b",".join(batch)
        status, data = mail.fetch(seq_set, "(X-GM-THRID)")
        if status != "OK" or not data:
            continue
        for part in data:
            # imaplib can return (bytes, bytes) or just bytes for each FETCH line
            if isinstance(part, tuple) and len(part) >= 1:
                raw = part[0]
            elif isinstance(part, (bytes, str)):
                raw = part
            else:
                continue
            if isinstance(raw, str):
                raw = raw.encode("utf-8", errors="replace")
            for m in thrid_re.finditer(raw):
                thread_ids.add(m.group(1).decode("ascii"))
    return len(thread_ids)


SearchScope = Literal["all", "unread"]


def search_gmail(
    query: str,
    scope: SearchScope = "unread",
    max_results: int = 20,
    category: str | None = None,
    count_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Perform a simple full-text search over your Gmail inbox via IMAP.

    - `query`: arbitrary text to search for (subject, from, body).
    - `scope`: "unread" (default) searches only unread messages; "all" searches all.
    - `max_results`: soft cap for list results; ignored when count_only=True.
    - `category`: optional Gmail category (updates, primary, promotions, social, forums).
    - `count_only`: if True, return only the total match count (no message bodies fetched).

    Returns a list of dicts with keys: id, from, subject; or when count_only=True,
    a single dict with key "_count" (int) so the tab-style unread count is accurate.

    This uses IMAP with an app password. Configure via:
      - GMAIL_IMAP_USERNAME
      - GMAIL_IMAP_PASSWORD
    """
    username, password = _require_config()

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("INBOX")

        # Query is already interpreted by the LLM (short search terms); use as-is.
        keyword_query = " ".join(query.split()).strip() if query else ""
        debug_tries: List[Dict[str, Any]] = []

        # When no category and just "unread in inbox", try multiple search strategies
        # (Gmail IMAP can be picky; we'll use whichever returns data and log for debugging).
        if not category and scope == "unread" and not keyword_query:
            message_ids = []

            for label, search_args in [
                ("UNSEEN (standard IMAP)", (None, "UNSEEN")),
                ("X-GM-RAW 'is:unread' quoted", (None, "X-GM-RAW", '"is:unread"')),
                ("X-GM-RAW is:unread unquoted", (None, "X-GM-RAW", "is:unread")),
            ]:
                try:
                    status, data = mail.search(*search_args)
                    ids = data[0].split() if data and data[0] else []
                    debug_tries.append({
                        "label": label,
                        "status": status,
                        "count": len(ids),
                        "response_preview": str(data)[:200] if data else "None",
                    })
                    if ids:
                        message_ids = ids
                        break
                except Exception as e:
                    debug_tries.append({"label": label, "error": str(e)})
        else:
            # Category, or keywords: use Gmail X-GM-RAW so we match Gmail's behavior.
            if category:
                raw_parts = [f"category:{category}"]
                if scope == "unread":
                    raw_parts.append("is:unread")
                if keyword_query and not _is_pure_category_count_query(query, category):
                    raw_parts.append(keyword_query)
            else:
                raw_parts = []
                if scope == "unread":
                    raw_parts.append("is:unread")
                    raw_parts.append("in:inbox")
                elif not keyword_query:
                    raw_parts.append("in:inbox")
                if keyword_query:
                    raw_parts.append(keyword_query)
            raw_query = " ".join(raw_parts)
            status, data = mail.search(None, "X-GM-RAW", f'"{raw_query}"')
            if status != "OK":
                raise GmailClientError(f"Gmail search failed: {status} {data}")
            message_ids = data[0].split() if data and data[0] else []

        total_count = len(message_ids)

        if count_only:
            # When asking for a category unread count, also compute thread (conversation)
            # count so we can show the number that matches the tab badge (e.g. "Updates 3").
            thread_count: int | None = None
            if category and scope == "unread" and message_ids:
                thread_count = _fetch_unique_thread_count(mail, message_ids)
            mail.logout()
            result: Dict[str, Any] = {"_count": total_count}
            if thread_count is not None:
                result["_thread_count"] = thread_count
            if total_count == 0 and debug_tries:
                result["_debug"] = debug_tries
            return [result]

        # Show newest first, then cap for list view
        message_ids = list(reversed(message_ids))[:max_results]

        results: List[Dict[str, Any]] = []

        if not message_ids:
            return results

        for msg_id in message_ids:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not isinstance(msg_data[0], tuple):
                continue

            raw_email = msg_data[0][1]
            msg = message_from_bytes(raw_email)

            subject = _decode_header(msg.get("Subject"))
            from_header = _decode_header(msg.get("From"))
            date_str = ""
            try:
                dt = parsedate_to_datetime(msg.get("Date") or "")
                date_str = dt.strftime("%Y-%m-%d") if dt else ""
            except Exception:
                pass
            snippet = _get_body_snippet(msg, max_chars=400)

            results.append(
                {
                    "id": msg_id.decode("ascii", errors="ignore"),
                    "from": from_header,
                    "subject": subject,
                    "date": date_str,
                    "snippet": snippet,
                }
            )

        mail.logout()
        return results
    except GmailClientError:
        raise
    except Exception as e:
        raise GmailClientError(f"Unexpected Gmail error: {e}") from e

