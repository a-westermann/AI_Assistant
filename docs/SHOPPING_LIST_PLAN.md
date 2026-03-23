# Shopping List Tab - Requirements and Plan

## Goal
Add a shared Shopping List tab to the Android app, backed by data stored on the PC so two users can use the same list in real time.

## Scope

### In Scope (Phase 1)
- Shared shopping list storage on PC backend.
- Android Shopping List tab UI.
- Add item, edit item, delete item.
- Duplicate prevention on exact match.
- Incremental search behavior while typing:
  - if entered text matches an existing item, scroll to it during typing.
- Shopping mode:
  - one-tap mark-off behavior, visual gray text.
- Sort editor and deterministic autosort:
  - use user-provided reference ordering list.
  - new entries auto-positioned by that reference.
- Offline editing with delayed sync:
  - user can edit list while away from home network.
  - local changes merge to server when connectivity returns.

### Out of Scope (Phase 1)
- AI-assisted normalization/suggestions.
- OCR/voice-to-list parsing.
- Multi-list support (only one shared household list).

### Phase 2 Placeholder (AI)
- Add AI-assisted shopping workflows:
  - fuzzy dedupe (e.g. "bell pepper" vs "green pepper").
  - category inference and sort suggestions for unknown items.
  - voice command parsing ("add eggs, milk, and cereal to shopping list").
  - smart suggestions based on history.

## Functional Requirements

### FR1: Shared Data
- List data is persisted on PC filesystem.
- Both users hit same backend endpoints and see the same list.
- Writes are atomic to avoid corruption.

### FR2: Add Item
- User can enter text and submit.
- Trim surrounding whitespace.
- Exact-match duplicate check (case-insensitive exact string after normalization).
- If duplicate:
  - show alert/toast and cancel add.

### FR3: Inline Match/Scroll While Typing
- As user types in add box:
  - search current list for best exact/prefix match.
  - if match found, scroll list to that item (best effort, no jarring jump loop).

### FR4: Edit Item
- User can edit an existing entry text.
- Edited value re-runs duplicate validation:
  - if it would collide with another exact entry, reject edit with warning.
- Re-apply autosort after successful edit.

### FR5: Delete Item
- User can remove item from list.
- Delete updates shared backend immediately.

### FR6: Shopping Mode
- Toggle "Shopping mode" in tab.
- In mode:
  - tapping item toggles checked state.
  - checked item renders gray (and optional strikethrough).
- Checked state persists in backend.

### FR7: Sort Editor
- User can open "Sort editor" view in app.
- User can edit ordering reference list (one item per line).
- Save pushes full reference list to backend.

### FR8: Autosort on Insert/Edit
- When adding/editing item:
  - if item exists in reference order, place by that index.
  - if unknown, append to end of "unknown items" section (stable order).

### FR9: Offline-First Editing and Merge
- If server is unreachable:
  - app operates on a local device copy of shopping list.
  - add/edit/delete/check actions are stored as pending operations.
- When server becomes reachable again:
  - app fetches latest server list.
  - app merges local pending operations into server state.
  - app resolves duplicates using normalized-name exact match rules.
  - app pushes merged result (or operation sequence) to server.
- User should see sync status:
  - `Offline edits pending` when local queue exists.
  - `Synced` after successful merge/push.

## Data Model (Phase 1)

### shopping_list.json
- `items`: array of:
  - `id` (string UUID)
  - `name` (display text)
  - `normalized_name` (internal dedupe key)
  - `checked` (bool)
  - `created_at` (ISO)
  - `updated_at` (ISO)
- `version` (int) for optimistic update support (optional in v1).

### local_pending_ops.json (on Android local storage)
- queue of offline operations with timestamps:
  - `add(name, normalized_name)`
  - `edit(id_or_prev_name, new_name, new_normalized_name)`
  - `delete(id_or_name)`
  - `toggle_checked(id_or_name, checked)`
- `base_server_version` at time offline mode began (if known).

### shopping_sort_order.json
- `reference_order`: array of normalized item names in desired order.
- `updated_at`.

## Backend API Plan (FastAPI)

### Endpoints
- `GET /shopping/items`
- `POST /shopping/items` (add)
- `PATCH /shopping/items/{id}` (edit name and/or checked)
- `DELETE /shopping/items/{id}`
- `POST /shopping/items/{id}/toggle` (optional convenience)
- `GET /shopping/sort-order`
- `PUT /shopping/sort-order`

### Backend Rules
- Normalize names for dedupe (lowercase, trim, collapse spaces).
- Reject duplicate on add/edit with 409 status.
- Recompute sorted list after add/edit/sort-order update.
- Persist atomically:
  - write temp file, then replace.
- Provide stable IDs and versions so clients can merge safely.

## Android UI Plan

### New Tab
- Add `Shopping` as new bottom tab.

### Main Shopping View
- Input + add button.
- "Shopping mode" toggle.
- List (LazyColumn) with:
  - item text
  - edit action
  - delete action
- In shopping mode:
  - single tap toggles checked.
  - checked style = gray text.

### Typing-Scroll Behavior
- On text change:
  - search loaded list for exact/prefix match.
  - scroll to item using list state.
- Debounce slightly to avoid jitter.

### Sort Editor View
- Open via button in Shopping tab.
- Multiline editor for reference list.
- Save/cancel actions.

## Non-Functional Requirements
- Fast interactions (<300ms perceived for local LAN operations).
- No data loss on restart.
- Handle concurrent edits from two users safely.
- Clear error feedback for conflicts/network failures.

## Conflict Handling
- If two users edit simultaneously:
  - online direct conflict: return 409 on stale version.
  - offline merge conflict policy:
    - duplicate add/edit collisions: keep one canonical item (normalized-name key).
    - delete wins over stale toggle/edit to deleted item.
    - latest timestamp wins for competing edit vs edit on same item.
    - if an item is renamed to an existing normalized name, merge entries and preserve checked=true if either side had true.

## Implementation Phases

### Phase 1A - Backend
1. Add shopping storage module.
2. Add shopping endpoints.
3. Add sort-order persistence and autosort.
4. Add duplicate and validation handling.

### Phase 1B - Android Tab
1. Add Shopping tab shell.
2. Implement list fetch/render.
3. Add add/edit/delete.
4. Add shopping mode checked state.
5. Add typing match + scroll.

### Phase 1C - Sort Editor
1. Add sort editor UI.
2. Hook save/load endpoints.
3. Verify autosort behavior for known/unknown items.

### Phase 1D - QA
1. Two-device shared tests.
2. Duplicate/edit conflict tests.
3. Offline/reconnect behavior.
4. UX polish for alerts/toasts.

### Phase 1E - Offline Sync
1. Add local shopping cache and pending-op queue in app storage.
2. Add connectivity checks + background retry when app regains network.
3. Implement deterministic merge pipeline and duplicate compaction.
4. Add sync status indicators and manual `Sync now` action.

## Open Decisions
- Should duplicate detection ignore punctuation (`milk` vs `milk.`)?
- Should checked items stay interleaved or pinned bottom in shopping mode?
- Should unknown items be alphabetized or insertion-order?
- Should sorting reference support aliases (e.g. `bell pepper` -> `pepper`)?
- Should checked state merge be OR (`true` if either side checked) or timestamp-wins?
- Should offline adds without server IDs be matched by normalized name only, or name+created_at heuristic?

## Phase 2 AI Entry (Return Later)
- Revisit after Phase 1 stabilization:
  - AI item normalization and aliasing.
  - AI-assisted sort-order updates from observed usage.
  - Voice-to-shopping-list commands with extraction and confirmation.

## Offline Merge Design Note
- Keep deterministic, non-LLM merge behavior for offline queue replay.
- Use normalized-name duplicate checks and explicit conflict policies from `FR9`.
- Backend endpoint `PUT /shopping/items/replace-all` remains planned for efficient reconciliation.
