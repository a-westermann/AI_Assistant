"""
Daily briefing module: gather email counts, light status, and time-of-day context
to power Galadrial's morning/evening briefing feature.
"""

import datetime

from gmail_client import search_gmail, GmailClientError
from lights_client import get_lights_state, LightsClientError


def get_time_context():
    """Return time-of-day info for briefing personalization."""
    now = datetime.datetime.now()
    hour = now.hour
    if 5 <= hour < 12:
        period = "morning"
        greeting = "Good morning"
    elif 12 <= hour < 17:
        period = "afternoon"
        greeting = "Good afternoon"
    elif 17 <= hour < 21:
        period = "evening"
        greeting = "Good evening"
    else:
        period = "night"
        greeting = "Good evening"
    return {
        "time": now.strftime("%I:%M %p"),
        "date": now.strftime("%A, %B %d"),
        "hour": hour,
        "period": period,
        "greeting": greeting,
    }


def get_time_of_day_light_mood():
    """Return a light mood description appropriate for current time of day."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 9:
        return "warm gentle sunrise, soft amber, brightness 50"
    elif 9 <= hour < 12:
        return "bright energizing daylight, cool white, brightness 80"
    elif 12 <= hour < 17:
        return "bright focused work light, neutral white, brightness 75"
    elif 17 <= hour < 20:
        return "warm relaxing golden hour, soft orange amber, brightness 60"
    elif 20 <= hour < 23:
        return "cozy dim evening, warm amber, brightness 40"
    else:
        return "very dim warm nightlight, deep amber, brightness 20"


def get_goodnight_light_mood():
    """Return a wind-down light mood for bedtime."""
    return "very dim, sleepy warm red-amber glow, brightness 15"


def gather_briefing_data():
    """
    Gather all briefing data: email counts, light status, time context.
    Returns a dict with all gathered info. Failures are captured gracefully.
    """
    data = {"time": get_time_context()}

    # Total unread count
    try:
        result = search_gmail(query="", scope="unread", count_only=True)
        if result and isinstance(result[0], dict) and "_count" in result[0]:
            data["email_unread"] = result[0]["_count"]
        else:
            data["email_unread"] = len(result)
    except GmailClientError:
        data["email_unread"] = None
        data["email_error"] = True

    # Per-category counts
    categories = {}
    for cat in ("primary", "updates", "promotions", "social"):
        try:
            result = search_gmail(query="", scope="unread", category=cat, count_only=True)
            if result and isinstance(result[0], dict):
                count = result[0].get("_thread_count") or result[0].get("_count", 0)
                categories[cat] = count
        except GmailClientError:
            categories[cat] = None
    data["email_categories"] = categories

    # Light status
    try:
        lights = get_lights_state()
        data["lights_state"] = lights.get("state", "unknown")
    except LightsClientError:
        data["lights_state"] = None

    return data


def format_briefing_for_llm(data):
    """Format gathered briefing data into a system note for the LLM."""
    time_ctx = data.get("time", {})
    lines = []
    lines.append(f"Current time: {time_ctx.get('time', 'unknown')}, {time_ctx.get('date', 'unknown')}.")
    lines.append(f"Time of day: {time_ctx.get('period', 'unknown')}.")

    if data.get("email_error"):
        lines.append("Email: Could not check Gmail (not configured or unreachable).")
    elif data.get("email_unread") is not None:
        total = data["email_unread"]
        lines.append(f"Unread emails: {total} total.")
        cats = data.get("email_categories", {})
        cat_parts = []
        for cat in ("primary", "updates", "social", "promotions"):
            count = cats.get(cat)
            if count is not None and count > 0:
                cat_parts.append(f"{cat.title()}: {count}")
        if cat_parts:
            lines.append("  Breakdown: " + ", ".join(cat_parts) + ".")

    lights = data.get("lights_state")
    if lights:
        lines.append(f"Lights are currently: {lights}.")
    else:
        lines.append("Lights: could not check status.")

    return "\n".join(lines)


def format_briefing_for_web(data):
    """Format briefing data as a dict suitable for JSON/web display."""
    time_ctx = data.get("time", {})
    email_cats = data.get("email_categories", {})
    return {
        "time": time_ctx.get("time", ""),
        "date": time_ctx.get("date", ""),
        "period": time_ctx.get("period", ""),
        "greeting": time_ctx.get("greeting", ""),
        "email_total": data.get("email_unread"),
        "email_error": data.get("email_error", False),
        "email_primary": email_cats.get("primary"),
        "email_updates": email_cats.get("updates"),
        "email_social": email_cats.get("social"),
        "email_promotions": email_cats.get("promotions"),
        "lights_state": data.get("lights_state"),
    }
