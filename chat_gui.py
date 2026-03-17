import json
import os
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext

from llm import ask_lmstudio
from lights_client import (
    get_lights_state,
    toggle_all_lights,
    LightsClientError,
    set_lights_auto,
)
from gmail_client import search_gmail, GmailClientError
from assistant_engine import AssistantEngine

# Plex sync app: run script in background and notify when done
PLEX_SYNC_DIR = r"H:\Coding\Python Projects\plex_sync"
PLEX_SYNC_PY = os.path.join(PLEX_SYNC_DIR, ".venv", "Scripts", "python.exe")
PLEX_SYNC_MAIN = os.path.join(PLEX_SYNC_DIR, "main.py")


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Galadrial - AI Assistant")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)

        # Color palette inspired by the reference UI
        self.colors = {
            "bg": "#050608",
            "sidebar_bg": "#0b0d10",
            "sidebar_active": "#1b1f26",
            "panel_bg": "#11141a",
            "panel_alt": "#151922",
            "border": "#222733",
            "accent": "#f0b34a",
            # Tkinter does not support alpha in hex colors; use a darker accent instead
            "accent_soft": "#403017",
            "text": "#f7f7f7",
            "text_muted": "#9a9fb2",
            "chip_bg": "#181c24",
        }

        # simple in-memory light state for the Living Room card
        self.living_room_on = False
        self.living_room_total = 2  # both lights are always controlled together

        # simple conversational state for routing and context
        self.last_route: dict | None = None
        self.last_user_message: str | None = None
        # Use shared assistant engine for routing + tools (lights.set_scene, Nanoleaf, Gmail, etc.)
        self._assistant_engine = AssistantEngine(log_fn=self._log_event)

        self._configure_root()
        self._build_layout()
        self._build_sidebar()
        self._build_chat_panel()
        self._build_right_panel()

    # ----- Layout & styling -------------------------------------------------

    def _configure_root(self) -> None:
        self.root.configure(bg=self.colors["bg"])
        self.root.columnconfigure(0, weight=0)  # sidebar
        self.root.columnconfigure(1, weight=3)  # chat
        self.root.columnconfigure(2, weight=2)  # right panel
        self.root.rowconfigure(0, weight=1)

        # App icon (Evenstar); keep a reference so it isn't garbage-collected
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "201505galadriel.png")
        if os.path.isfile(icon_path):
            self._icon_photo = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, self._icon_photo)
        else:
            self._icon_photo = None

    def _build_layout(self) -> None:
        # Sidebar
        self.sidebar = tk.Frame(
            self.root, bg=self.colors["sidebar_bg"], bd=0, highlightthickness=0
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Main chat column
        self.chat_column = tk.Frame(
            self.root, bg=self.colors["bg"], bd=0, highlightthickness=0
        )
        self.chat_column.grid(row=0, column=1, sticky="nsew", padx=(8, 8), pady=8)
        self.chat_column.rowconfigure(0, weight=1)
        self.chat_column.rowconfigure(1, weight=0)
        self.chat_column.columnconfigure(0, weight=1)

        # Right column
        self.right_column = tk.Frame(
            self.root, bg=self.colors["bg"], bd=0, highlightthickness=0
        )
        self.right_column.grid(row=0, column=2, sticky="nsew", padx=(0, 8), pady=8)
        self.right_column.rowconfigure(0, weight=1)
        self.right_column.columnconfigure(0, weight=1)

    def _sidebar_button(self, parent, text: str, active: bool = False) -> tk.Frame:
        bg = self.colors["sidebar_active"] if active else self.colors["sidebar_bg"]
        fg = self.colors["text"] if active else self.colors["text_muted"]

        frame = tk.Frame(parent, bg=bg)
        frame.pack(fill="x", pady=2)

        dot = tk.Canvas(
            frame,
            width=8,
            height=8,
            bg=bg,
            bd=0,
            highlightthickness=0,
        )
        if active:
            dot.create_oval(2, 2, 6, 6, fill=self.colors["accent"], outline="")
        dot.pack(side="left", padx=(12, 8))

        label = tk.Label(
            frame,
            text=text,
            bg=bg,
            fg=fg,
            anchor="w",
            font=("Segoe UI", 10, "bold" if active else "normal"),
        )
        label.pack(side="left", fill="x", expand=True, pady=6)

        return frame

    def _build_sidebar(self) -> None:
        # App title
        title_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar_bg"])
        title_frame.pack(fill="x", pady=(16, 24))

        title_label = tk.Label(
            title_frame,
            text="Galadrial",
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 16),
            anchor="w",
        )
        title_label.pack(side="left", padx=16)

        # Navigation
        nav_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar_bg"])
        nav_frame.pack(fill="both", expand=True)

        self._sidebar_button(nav_frame, "Dashboard", active=True)
        self._sidebar_button(nav_frame, "LLM")
        self._sidebar_button(nav_frame, "Lights")
        self._sidebar_button(nav_frame, "Steam")
        self._sidebar_button(nav_frame, "Riot")
        self._sidebar_button(nav_frame, "Sentinel")
        self._sidebar_button(nav_frame, "Docs")

        # Placeholder footer
        footer = tk.Label(
            self.sidebar,
            text="v0.1.0\nlocal assistant",
            bg=self.colors["sidebar_bg"],
            fg=self.colors["text_muted"],
            font=("Segoe UI", 8),
            justify="left",
        )
        footer.pack(anchor="sw", padx=16, pady=16)

    def _build_chat_panel(self) -> None:
        # Top bar
        header = tk.Frame(self.chat_column, bg=self.colors["bg"])
        header.grid(row=0, column=0, sticky="new", pady=(0, 6))

        title = tk.Label(
            header,
            text="CHAT",
            fg=self.colors["text_muted"],
            bg=self.colors["bg"],
            font=("Segoe UI", 9, "bold"),
        )
        title.pack(side="left")

        # Chat container
        chat_container = tk.Frame(
            self.chat_column,
            bg=self.colors["panel_bg"],
            bd=1,
            relief="solid",
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            highlightthickness=1,
        )
        chat_container.grid(row=0, column=0, sticky="nsew")

        chat_container.rowconfigure(0, weight=1)
        chat_container.columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            state="disabled",
            wrap="word",
            font=("Segoe UI", 10),
            bg=self.colors["panel_bg"],
            fg=self.colors["text"],
            bd=0,
            highlightthickness=0,
            insertbackground=self.colors["text"],
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # Input bar
        input_bar = tk.Frame(self.chat_column, bg=self.colors["bg"])
        input_bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        input_bar.columnconfigure(0, weight=1)

        input_container = tk.Frame(
            input_bar,
            bg=self.colors["panel_bg"],
            bd=1,
            relief="solid",
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            highlightthickness=1,
        )
        input_container.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        input_container.columnconfigure(0, weight=1)

        self.entry = tk.Text(
            input_container,
            height=3,
            wrap="word",
            font=("Segoe UI", 10),
            bg=self.colors["panel_bg"],
            fg=self.colors["text"],
            bd=0,
            highlightthickness=0,
            insertbackground=self.colors["text"],
        )
        self.entry.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        send_button = tk.Button(
            input_bar,
            text="➤",
            width=4,
            command=self.on_send_clicked,
            bg=self.colors["accent"],
            fg="#000000",
            activebackground=self.colors["accent"],
            activeforeground="#000000",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 11, "bold"),
        )
        send_button.grid(row=0, column=1)

        # Bind Enter to send (Shift+Enter for newline)
        self.entry.bind("<Return>", self.on_enter)
        self.entry.bind("<Shift-Return>", self.on_shift_enter)

    def _build_right_panel(self) -> None:
        panel = tk.Frame(
            self.right_column,
            bg=self.colors["panel_bg"],
            bd=1,
            relief="solid",
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["border"],
            highlightthickness=1,
        )
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)

        # Tabs row
        tabs_row = tk.Frame(panel, bg=self.colors["panel_bg"])
        tabs_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))

        def tab_button(text: str, active: bool = False) -> tk.Label:
            bg = self.colors["panel_bg"] if not active else self.colors["accent_soft"]
            fg = self.colors["text_muted"] if not active else self.colors["accent"]
            return tk.Label(
                tabs_row,
                text=text,
                bg=bg,
                fg=fg,
                padx=10,
                pady=4,
                font=("Segoe UI", 9, "bold" if active else "normal"),
            )

        lights_tab = tab_button("Lights", active=True)
        lights_tab.pack(side="left")
        steam_tab = tab_button("Steam")
        steam_tab.pack(side="left", padx=(4, 0))
        riot_tab = tab_button("Riot")
        riot_tab.pack(side="left", padx=(4, 0))

        # Lights header
        lights_header = tk.Frame(panel, bg=self.colors["panel_bg"])
        lights_header.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 4))

        lights_label = tk.Label(
            lights_header,
            text="Lights",
            bg=self.colors["panel_bg"],
            fg=self.colors["text"],
            font=("Segoe UI Semibold", 14),
            anchor="w",
        )
        lights_label.pack(side="left")

        # Rooms grid (placeholder cards)
        rooms_frame = tk.Frame(panel, bg=self.colors["panel_bg"])
        rooms_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 8))

        def room_card(parent, name: str, status: str = "0/0 on", on_click=None):
            card = tk.Frame(
                parent,
                bg=self.colors["panel_alt"],
                bd=1,
                relief="solid",
                highlightbackground=self.colors["border"],
                highlightthickness=1,
            )
            name_lbl = tk.Label(
                card,
                text=name,
                bg=self.colors["panel_alt"],
                fg=self.colors["text"],
                font=("Segoe UI", 10, "bold"),
                anchor="w",
            )
            name_lbl.pack(anchor="w", padx=10, pady=(8, 0))

            status_lbl = tk.Label(
                card,
                text=status,
                bg=self.colors["panel_alt"],
                fg=self.colors["text_muted"],
                font=("Segoe UI", 9),
                anchor="w",
            )
            status_lbl.pack(anchor="w", padx=10, pady=(0, 8))

            power_btn = tk.Label(
                card,
                text="⏻",
                bg=self.colors["accent_soft"],
                fg=self.colors["accent"],
                font=("Segoe UI", 11, "bold"),
                padx=10,
                pady=4,
            )
            power_btn.pack(side="right", padx=10, pady=8)

            if on_click is not None:
                power_btn.bind("<Button-1>", lambda _e: on_click())

            return card

        rooms = [
            ("Living Room", "0/2 off"),
            ("Study", "2/2 on"),
            ("Music room", "0/1 on"),
            ("Outside", "0/2 on"),
            ("Garage", "0/1 on"),
            ("Basement", "8/8 on"),
        ]

        for idx, (name, status) in enumerate(rooms):
            col = idx % 2
            row = idx // 2

            if name == "Living Room":
                # special card that controls all lights via your Govee automation service
                self.living_room_status = status
                self.living_room_status_label = None

                def on_toggle():
                    self._toggle_living_room()

                card = room_card(rooms_frame, name, status, on_click=on_toggle)

                # capture the status label for later updates
                # (last packed widget before power_btn in room_card)
                for child in card.winfo_children():
                    if isinstance(child, tk.Label) and child.cget("text") == status:
                        self.living_room_status_label = child
                        break

                # Auto button for the Living Room (uses /auto endpoint)
                auto_btn = tk.Label(
                    card,
                    text="Auto",
                    bg=self.colors["accent_soft"],
                    fg=self.colors["accent"],
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=4,
                )
                auto_btn.pack(side="right", padx=(0, 4), pady=8)
                auto_btn.bind("<Button-1>", lambda _e: self._set_living_room_auto())
            else:
                card = room_card(rooms_frame, name, status)

            card.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
            rooms_frame.columnconfigure(col, weight=1)

        # Event monitor (placeholder)
        event_frame = tk.Frame(panel, bg=self.colors["panel_bg"])
        event_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(4, 10))
        panel.rowconfigure(3, weight=1)

        event_header = tk.Frame(event_frame, bg=self.colors["panel_bg"])
        event_header.pack(fill="x")

        event_label = tk.Label(
            event_header,
            text="EVENT MONITOR",
            bg=self.colors["panel_bg"],
            fg=self.colors["text_muted"],
            font=("Segoe UI", 9, "bold"),
        )
        event_label.pack(side="left")

        event_count = tk.Label(
            event_header,
            text="1 events",
            bg=self.colors["chip_bg"],
            fg=self.colors["text_muted"],
            font=("Segoe UI", 9),
            padx=8,
            pady=2,
        )
        event_count.pack(side="right")

        self.event_log = scrolledtext.ScrolledText(
            event_frame,
            state="normal",
            wrap="word",
            font=("Consolas", 9),
            bg=self.colors["panel_alt"],
            fg=self.colors["text_muted"],
            bd=0,
            highlightthickness=0,
        )
        self.event_log.pack(fill="both", expand=True, pady=(4, 0))
        self.event_log.insert(
            "end",
            "[placeholder] Events, device logs, and tool calls will appear here.\n",
        )
        self.event_log.configure(state="disabled")

    def append_message(self, sender: str, text: str) -> None:
        """Append a message to the chat display."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {text}\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

        # Mirror only system-style messages into the event monitor
        # (Assistant messages stay in chat only; errors are logged via _log_event directly.)
        if sender == "System":
            self._log_event(f"{sender}: {text}")

    # ----- Lights integration ------------------------------------------------

    def _log_event(self, message: str) -> None:
        # Print to console for debugging
        print(message)

        # Also append to the EVENT MONITOR text box, if it exists
        if hasattr(self, "event_log") and self.event_log is not None:
            self.event_log.configure(state="normal")
            self.event_log.insert("end", message + "\n")
            self.event_log.see("end")
            self.event_log.configure(state="disabled")

    def _toggle_living_room(self) -> None:
        """Toggle all lights on/off via the external Govee automation service."""
        # Run HTTP call off the main thread to keep UI responsive.
        threading.Thread(target=self._toggle_living_room_worker, daemon=True).start()

    def _toggle_living_room_worker(self) -> None:
        target_state = "off" if self.living_room_on else "on"
        try:
            self._log_event(f"Sending lights {target_state} request...")
            toggle_all_lights(target_state)
            self.living_room_on = not self.living_room_on

            # Update UI back on main thread
            on_count = self.living_room_total if self.living_room_on else 0
            new_status = f"{on_count}/{self.living_room_total} " + (
                "on" if self.living_room_on else "off"
            )
            self.root.after(0, lambda: self._update_living_room_status(new_status))
        except LightsClientError as e:
            self._log_event(f"Lights error: {e}")
        except Exception as e:
            self._log_event(f"Unexpected lights error: {e}")

    def _set_living_room_state(self, desired_state: str) -> None:
        """Set lights explicitly on or off based on a chat command."""
        threading.Thread(
            target=self._set_living_room_state_worker,
            args=(desired_state,),
            daemon=True,
        ).start()

    def _set_living_room_state_worker(self, desired_state: str) -> None:
        if desired_state not in ("on", "off"):
            return

        try:
            self._log_event(f"Sending lights {desired_state} request (explicit)...")
            toggle_all_lights(desired_state)
            self.living_room_on = desired_state == "on"

            on_count = self.living_room_total if self.living_room_on else 0
            status_text = f"{on_count}/{self.living_room_total} " + (
                "on" if self.living_room_on else "off"
            )
            self.root.after(
                0, lambda: self._update_living_room_status(status_text)
            )
        except LightsClientError as e:
            self._log_event(f"Lights error (explicit): {e}")
        except Exception as e:
            self._log_event(f"Unexpected lights error (explicit): {e}")

    def _set_living_room_auto(self) -> None:
        """Set lights to automatic mode via the external service."""
        threading.Thread(target=self._set_living_room_auto_worker, daemon=True).start()

    def _set_living_room_auto_worker(self) -> None:
        try:
            self._log_event("Sending lights auto request...")
            set_lights_auto()

            # Update UI back on main thread; keep count but show auto state
            current_on = self.living_room_total if self.living_room_on else 0
            status_text = f"{current_on}/{self.living_room_total} auto"
            self.root.after(
                0, lambda: self._update_living_room_status(status_text)
            )
        except LightsClientError as e:
            self._log_event(f"Lights error (auto): {e}")
        except Exception as e:
            self._log_event(f"Unexpected lights error (auto): {e}")

    # ----- Chat input handling -----------------------------------------------

    def _maybe_handle_light_command(self, text: str) -> str | None:
        """
        Inspect the user's message for simple light commands and act on them.

        Supported patterns (case-insensitive):
        - "...lights...on..."  -> turn lights on
        - "...lights...off..." -> turn lights off
        - "...lights...auto..." -> set lights to auto
        """
        t = text.lower()

        if "light" not in t and "lights" not in t:
            return None

        # Auto takes precedence if mentioned
        if "auto" in t or "automatic" in t:
            self._log_event("Setting lights to auto.")
            self._set_living_room_auto()
            return "auto"

        # If both on and off are present, treat as ambiguous and do nothing
        has_on = " on" in t or "on " in t
        has_off = " off" in t or "off " in t

        if has_on and not has_off:
            self._log_event("Turning lights on.")
            self._set_living_room_state("on")
            return "on"
        elif has_off and not has_on:
            self._log_event("Turning lights off.")
            self._set_living_room_state("off")
            return "off"

        return None

    # ----- Command routing via LLM --------------------------------------------

    def _heuristic_route(self, user_text: str) -> dict:
        """
        Simple local fallback router used ONLY when the LLM router fails
        or returns invalid JSON. This is a safety net to avoid obviously
        wrong behaviour like saying it cannot access email when it should.
        """
        t = user_text.lower()

        # "Do that again" style follow-ups: reuse the last successful route
        if self.last_route is not None and any(
            phrase in t for phrase in ("again", "same thing", "do that", "check again")
        ):
            return self.last_route

        # Plex sync heuristic
        if "plex" in t and "sync" in t:
            return {"action": "plex_sync.run", "params": {}}

        # Lights heuristic
        if "light" in t or "lights" in t:
            # Asking about current state (are they on? status?)
            if any(phrase in t for phrase in ("are the", "are my", "is the", "status", "state", "check")):
                return {"action": "lights.get_state", "params": {}}
            state: str | None = None
            if "auto" in t or "automatic" in t:
                state = "auto"
            elif "on" in t and "off" not in t:
                state = "on"
            elif "off" in t and "on" not in t:
                state = "off"

            if state is not None:
                return {"action": "lights.set_state", "params": {"state": state}}

        # Gmail / email heuristic
        if any(w in t for w in ("gmail", "email", "inbox", "mail", "submittable", "moksha")):
            scope = "unread" if any(w in t for w in ("unread", "new", "recent")) else "all"
            result_type = "count" if any(w in t for w in ("how many", "number of", "count")) else "list"

            category = None
            if "updates" in t:
                category = "updates"
            elif "promotions" in t:
                category = "promotions"
            elif "primary" in t:
                category = "primary"
            elif "social" in t:
                category = "social"
            elif "forums" in t:
                category = "forums"

            params: dict[str, object] = {
                "query": user_text,
                "scope": scope,
                "result_type": result_type,
            }
            if category is not None:
                params["category"] = category

            return {
                "action": "gmail.search",
                "params": params,
            }

        # Default to no tool
        return {"action": "none", "params": {}}

    def _route_command(self, user_text: str) -> dict:
        """
        Ask the language model to interpret the user's message and choose
        an action + parameters, returned as JSON.

        Actions:
        - "none": no side effects, just chat
        - "lights.set_state": params { "state": "on" | "off" | "auto" }
        - "gmail.search": params {
              "query": string,
              "scope": "unread" | "all",
              "result_type": "count" | "list"
          }
        """
        prev_msg = self.last_user_message or ""
        prev_route_json = json.dumps(self.last_route) if self.last_route is not None else "null"

        routing_prompt = f"""
You are a command router for a desktop assistant called Galadrial.
Your job is to look at the user's message and decide which ACTION to take.

Valid actions:
- "none"  -> no side effects, just chat with the user.
- "lights.set_state" -> control my Govee lights.
    - params:
        - state: "on" | "off" | "auto"
- "lights.get_state" -> check whether the lights are currently on or off (no params).
- "plex_sync.run" -> run the user's Plex sync app (syncs media to a server). No params. Runs in the background; the app will tell the user when it finishes.
- "gmail.search" -> search my Gmail inbox via IMAP.
    - params:
        - query: string   (the search terms to send to Gmail—YOU interpret the user's intent)
        - scope: "unread" | "all"
        - result_type: "count" | "list"
        - category: one of "updates", "primary", "promotions", "social", "forums" (optional)
        - broad_search_terms: optional array of 2–6 short terms for "list" searches. When the user
          is asking to find/look through emails (e.g. "last email about X", "any acceptance from
          a literary mag"), set this to words that might appear in such emails so we cast a wider
          net (e.g. ["submission", "accepted", "acceptance", "literary", "magazine"]). Omit for
          count-only or when the user's query is already narrow.

Rules:
- If the user is clearly asking you to turn lights on/off/auto, use "lights.set_state".
- If the user is asking whether the lights are on, off, or what state they are in (e.g. "are the lights on?", "light status", "are my lights on?"), use "lights.get_state".
- If the user asks to run Plex sync, sync Plex, run the Plex sync app, start Plex sync, or similar, use "plex_sync.run".
- If the user is asking about Gmail, email, inbox, or messages, OR is clearly asking
  about things that would usually live in email (for example: literary magazine
  submissions, acceptances/rejections, stories being accepted, submission platforms
  like Submittable or Moksha), use "gmail.search".
  - For gmail.search, set params.query to a SHORT search string that expresses what
    to find. Interpret the user's message: e.g. "last email about literary mag
    submission being accepted" -> "submission accepted literary magazine"; "how many
    unread in Promotions" -> "promotions" or "unread". Use plain words only (no
    subject:, from:, or other operators). This query is sent directly to Gmail.
  - For result_type "list" when the user is asking to find or look through emails
    (e.g. "what's the last email about X", "any acceptance from a literary magazine"),
    also set params.broad_search_terms to an array of 2–8 terms that might appear in
    relevant emails. Include wording that appears in acceptance emails, not just
    rejections: e.g. "feature", "publish", "edits" (as in "work with me on edits"),
    "accepted", "acceptance", "submission", "literary", "magazine". We
    search with OR and then use the model to pick which emails actually match.
  - If the user clearly refers to a specific Gmail category such as "Updates",
    "Promotions", "Primary", "Social", or "Forums", set params.category to the
    corresponding lowercase name (e.g. "Updates" -> "updates").
  - Choose "unread" when they ask about unread, new, or recent mail; otherwise "all".
  - Choose "count" when they ask "how many", "number of", or clearly request a count.
  - Otherwise choose "list".
- If the message is purely conversational or does not match any tool, use "none".

You may use the previous user message and previous action to interpret short
follow-ups like "again", "same thing", or "check that for me". In those cases,
it is usually correct to repeat the same kind of action with similar parameters.

Respond with JSON ONLY, no explanation, in this exact schema:
{{
  "action": "<one of: none, lights.set_state, lights.get_state, plex_sync.run, gmail.search>",
  "params": {{}}
}}

User message:
\"\"\"
{user_text}
\"\"\"

Previous user message:
\"\"\"
{prev_msg}
\"\"\"

Previous routed action (JSON or null):
{prev_route_json}
"""
        try:
            response = ask_lmstudio(routing_prompt)
            raw = response["output"][0]["content"].strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1].strip()

            route = json.loads(raw)
            if not isinstance(route, dict):
                raise ValueError("Routing output was not a JSON object")

            action = str(route.get("action", "none")).strip() or "none"
            params = route.get("params") or {}
            if not isinstance(params, dict):
                params = {}

            cleaned = {"action": action, "params": params}
            # If the LLM returned an unknown action, fall back to heuristic routing.
            if action not in ("none", "lights.set_state", "lights.get_state", "plex_sync.run", "gmail.search"):
                self._log_event(
                    f"Router returned unknown action {action!r}; using heuristic router."
                )
                cleaned = self._heuristic_route(user_text)

            self._log_event(f"Router decided action: {cleaned}")
            return cleaned
        except Exception as e:
            self._log_event(f"Routing failed; using heuristic router instead: {e}")
            route = self._heuristic_route(user_text)
            self._log_event(f"Heuristic router decided action: {route}")
            return route

    def _update_living_room_status(self, status_text: str) -> None:
        if getattr(self, "living_room_status_label", None) is not None:
            self.living_room_status_label.config(text=status_text)

    def on_enter(self, event):
        """Handle Enter key (send message)."""
        self.on_send_clicked()
        return "break"  # prevent newline

    def on_shift_enter(self, event):
        """Allow Shift+Enter to insert a newline."""
        return None

    def on_send_clicked(self) -> None:
        """Send the current text to the model."""
        user_text = self.entry.get("1.0", "end").strip()
        if not user_text:
            return

        self.entry.delete("1.0", "end")
        self.last_user_message = user_text
        self.append_message("You", user_text)

        # Handle the message in a background thread (routing + tools + chat)
        threading.Thread(
            target=self._handle_user_message,
            args=(user_text,),
            daemon=True,
        ).start()

    def _handle_user_message(self, user_text: str) -> None:
        """Run routing and tools in the shared assistant engine, then display the reply."""
        try:
            reply = self._assistant_engine.handle_message(user_text)
        except Exception as e:
            reply = f"Error: {e}"
            self._log_event(str(e))
        self.last_route = self._assistant_engine.last_route
        self.last_user_message = user_text
        self.root.after(0, lambda: self.append_message("Galadrial", reply))

    def call_model_and_display(
        self,
        prompt: str,
        light_action: str | None,
        *,
        extra_note: str = "",
    ) -> None:
        """Call the LM Studio model and display its response."""
        try:
            # Give the model safe context about what is and is not under its control
            system_preamble = (
                "You are Galadrial, an AI assistant embedded in a desktop GUI.\n\n"
                "- The app can control my Govee lights via an API. When you see a "
                "system note telling you that the lights were set to a state, you may "
                "speak as if that action has already been performed.\n"
                "- The app may also display separate System messages with the results "
                "of tools (for example, Gmail searches, Plex sync). Those System messages are the "
                "source of truth about my real data.\n\n"
                "CRITICAL SAFETY RULES:\n"
                "- Do NOT invent or guess specific facts about my personal data, such "
                "as emails, literary magazine acceptances, bank balances, calendar "
                "events, or file contents.\n"
                "- If you are not given an explicit System note or tool result that "
                "contains those facts, you must say that you don't know instead of "
                "making something up.\n"
                "- You may still answer general questions with your own knowledge, but "
                "never fabricate concrete details about my life or accounts.\n\n"
            )

            action_note = ""
            if light_action in ("on", "off", "auto"):
                action_note = (
                    f"System note: The app has just set the lights to '{light_action}'.\n\n"
                )
            if extra_note:
                action_note = action_note + extra_note

            full_prompt = f"{system_preamble}{action_note}User: {prompt}"

            response = ask_lmstudio(full_prompt)
            # Adjust this if your response structure changes
            text = response["output"][0]["content"]
        except Exception as e:
            text = f"Error calling model: {e}"
            self._log_event(text)

        # Schedule UI update on the main thread
        self.root.after(0, lambda: self.append_message("Galadrial", text))

    # ----- Plex sync ----------------------------------------------------------

    def _run_plex_sync_worker(self) -> None:
        """Run the Plex sync script in a subprocess; notify on completion (called from background thread)."""
        try:
            if not os.path.isdir(PLEX_SYNC_DIR) or not os.path.isfile(PLEX_SYNC_MAIN):
                self.root.after(
                    0,
                    lambda: self._on_plex_sync_done(
                        -1, "", f"Plex sync path not found: {PLEX_SYNC_DIR!r}"
                    ),
                )
                return
            python_exe = PLEX_SYNC_PY if os.path.isfile(PLEX_SYNC_PY) else "python"
            result = subprocess.run(
                [python_exe, PLEX_SYNC_MAIN],
                cwd=PLEX_SYNC_DIR,
                capture_output=True,
                text=True,
            )
            self.root.after(
                0,
                lambda: self._on_plex_sync_done(
                    result.returncode, result.stdout or "", result.stderr or ""
                ),
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: self._on_plex_sync_done(-1, "", str(e)),
            )

    def _on_plex_sync_done(self, returncode: int, stdout: str, stderr: str) -> None:
        """Called on the main thread when Plex sync subprocess finishes."""
        if returncode == 0:
            # Parse script output: "uploaded {path}" and "skipping {path}"
            uploaded = sum(1 for line in stdout.splitlines() if "uploaded " in line)
            skipped = sum(1 for line in stdout.splitlines() if "skipping " in line)
            if uploaded or skipped:
                msg = (
                    f"Plex sync finished successfully. {uploaded} file(s) uploaded, {skipped} skipped."
                )
            else:
                msg = "Plex sync finished successfully. No files transferred (none needed or no output)."
            self._log_event(msg)
        else:
            err = stderr.strip() or stdout.strip() or f"Exit code {returncode}"
            msg = f"Plex sync finished with an error: {err}"
            self._log_event(f"Plex sync error: {err}")

    # ----- Gmail integration --------------------------------------------------

    def _interpret_email_list(self, user_question: str, messages: list[dict]) -> str | None:
        """Ask the model which emails match the user's question; return its reply or None on failure."""
        # Cap how many we send to the model so the prompt doesn't time out
        max_for_interpretation = 45
        if len(messages) > max_for_interpretation:
            messages = messages[:max_for_interpretation]
        if not messages:
            try:
                response = ask_lmstudio(
                    "The user asked about their email:\n\n"
                    f'"{user_question}"\n\n'
                    "We searched their Gmail and found no messages matching the search. "
                    "Reply in one short sentence that no matching emails were found."
                )
                return (response.get("output") or [{}])[0].get("content", "").strip() or None
            except Exception as e:
                self._log_event(f"Interpretation (no results) failed: {e}")
                return None
        lines = []
        for i, m in enumerate(messages, 1):
            from_ = m.get("from", "")
            subj = m.get("subject", "")
            date = m.get("date", "")
            snippet = m.get("snippet", "")
            parts = [f"{i}. From: {from_}", f"Subject: {subj}"]
            if date:
                parts.append(f"Date: {date}")
            line = " | ".join(parts)
            if snippet:
                line += "\n   Snippet: " + snippet
            lines.append(line)
        email_list_text = "\n".join(lines)
        prompt = (
            "The user asked about their email:\n\n"
            f'"{user_question}"\n\n'
            "Here are emails from their inbox (newest first), with a short body snippet for each:\n\n"
            f"{email_list_text}\n\n"
            "Which of these emails match what they're looking for? Use the snippet to recognize "
            "acceptances: e.g. 'would love to feature', 'we'd like to publish', 'accept your story', "
            "'contract', 'feature it', 'work with me on edits', 'I can publish it', 'publish it in the [issue]', "
            "'earliest available opening', 'would you be willing to work with me'—even if the subject doesn't say 'accepted'. "
            "If one or more match, say which one(s) and give the most relevant detail. "
            "If none match, say so briefly. Reply in 1–4 concise sentences; do not invent any emails not in the list."
        )
        try:
            response = ask_lmstudio(prompt)
            text = (response.get("output") or [{}])[0].get("content", "").strip()
            return text or None
        except Exception as e:
            self._log_event(f"Interpretation failed: {e}")
            return None

    def _search_gmail_worker(
        self,
        query: str,
        scope: str,
        result_type: str,
        category: str | None,
        user_question: str = "",
        broad_search_terms: list[str] | None = None,
    ) -> None:
        """Search Gmail via IMAP; for list type, optionally use broad OR search and LLM interpretation."""
        try:
            # For list type with broad terms, search with OR to get more candidates.
            # Use only submission-relevant terms so we don't pull in marketing (Venmo "review",
            # Wells Fargo "contract", etc.).
            _GENERIC_BROAD_TERMS = {"review", "feature", "contract"}
            search_query = query
            max_results = 20
            if result_type == "list" and broad_search_terms:
                narrow_terms = [
                    t for t in broad_search_terms
                    if t.lower() not in _GENERIC_BROAD_TERMS
                ]
                if not narrow_terms:
                    narrow_terms = list(broad_search_terms)
                search_query = " OR ".join(narrow_terms)
                max_results = 80
                self._log_event(
                    f"Searching Gmail with broad terms (OR) for interpretation: {search_query!r}"
                )
            self._log_event(
                f"Searching Gmail (scope={scope}, result={result_type}, category={category}) "
                f"for messages matching: {search_query!r}"
            )
            count_only = result_type == "count"
            messages = search_gmail(
                query=search_query,
                scope=scope,
                max_results=max_results,
                category=category,
                count_only=count_only,
            )  # type: ignore[arg-type]

            # When count_only, search_gmail may return [{"_count": N, "_thread_count": T}]
            # _thread_count = conversations (matches the tab badge, e.g. "Updates 3")
            total = len(messages)
            thread_count = None
            if count_only and messages and isinstance(messages[0], dict):
                m0 = messages[0]
                if "_count" in m0:
                    total = int(m0["_count"])
                if "_thread_count" in m0:
                    thread_count = int(m0["_thread_count"])
                if "_debug" in m0:
                    for entry in m0["_debug"]:
                        self._log_event(
                            f"Gmail debug: {entry.get('label', '?')} -> "
                            f"count={entry.get('count', '?')} status={entry.get('status', '?')} "
                            f"err={entry.get('error', '')} rsp={entry.get('response_preview', '')}"
                        )

            if total == 0 and (thread_count is None or thread_count == 0):
                if result_type == "count" and scope == "unread":
                    summary = "You have 0 unread Gmail messages."
                elif result_type == "count":
                    summary = "You have 0 Gmail messages matching that query."
                else:
                    if user_question.strip():
                        interpretation = self._interpret_email_list(user_question.strip(), [])
                        summary = interpretation or "No matching Gmail messages found."
                    else:
                        summary = "No matching Gmail messages found."
            else:
                if result_type == "count":
                    if thread_count is not None and category:
                        summary = (
                            f"You have {thread_count} unread conversation(s) in {category} "
                            f"({total:,} message(s))."
                        )
                    elif scope == "unread":
                        summary = (
                            f"You have {total:,} unread Gmail message(s)."
                        )
                    else:
                        summary = (
                            f"You have {total:,} Gmail message(s) "
                            "matching that query."
                        )
                else:
                    # List result: have the model interpret which emails match the user's question
                    if user_question.strip():
                        interpretation = self._interpret_email_list(
                            user_question.strip(), messages
                        )
                        if interpretation:
                            summary = interpretation
                        else:
                            # Don't dump a long raw list; give a short fallback so user can retry
                            self._log_event(
                                "Interpretation returned empty; showing short fallback."
                            )
                            summary = (
                                f"I found {total} email(s) matching your search but couldn't "
                                "interpret which one answers your question (model may have timed out). "
                                "Try asking again, or search your inbox for “Perseid Prophecies”, "
                                "“acceptance”, or the story title."
                            )
                    else:
                        lines = [
                            f"- {m.get('from', '')} — {m.get('subject', '')}"
                            for m in messages
                        ]
                        summary = (
                            f"Found {total} matching Gmail message(s):\n"
                            + "\n".join(lines)
                        )

            # Log result to Events only; have the assistant reply in chat
            self._log_event(summary)
            self.call_model_and_display(
                user_question,
                None,
                extra_note="The app ran a Gmail search. Use this result to answer the user in one concise message. Do not repeat the raw list; summarize or answer their question.\n\nResult:\n" + summary + "\n\n",
            )
        except GmailClientError as e:
            msg = f"Gmail error: {e}"
            self._log_event(msg)
            self.call_model_and_display(
                user_question,
                None,
                extra_note="System note: The app tried to search Gmail but failed. Tell the user briefly that the search failed and they can try again.\n\n",
            )


def main() -> None:
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

