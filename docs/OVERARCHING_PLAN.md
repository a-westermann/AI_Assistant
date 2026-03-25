# Galadrial Overarching Plan

This is the high-level roadmap across assistant capabilities.  
Detailed execution plans stay in focused docs (for example `docs/SHOPPING_LIST_PLAN.md` and `docs/REMOTE_ANDROID_PLAN.md`).

## Current Focus Areas

- Stabilize core assistant routing and device control reliability.
- Expand media control (Spotify + YouTube coexistence).
- Add practical LLM-powered workflows where deterministic storage/actions already exist.

## Planned Initiatives

### Shopping + LLM

1. Natural-language shopping actions (selected)
   - Let users speak naturally: "add milk and eggs", "remove bananas", "mark coffee done", "clear checked".
   - LLM parses intent into strict JSON tool actions.
   - Backend still performs deterministic validation and writes via existing shopping store functions.
   - Keep guardrails: no direct free-form file edits from model output.

2. Recipe/meal ingredient expansion (selected)
   - Support prompts like "add ingredients for tacos tonight".
   - LLM proposes item list + optional quantities.
   - Require confirmation flow before bulk add to prevent bad/hallucinated inserts.
   - Start with simple ingredient extraction, then iterate with category and dedupe refinement.

## Sequencing Suggestion

1. Implement natural-language shopping actions first (lower risk, immediate UX gain).
2. Add recipe-to-list expansion next behind an explicit confirm step.
3. Revisit advanced shopping intelligence later (fuzzy dedupe, aisle inference, history-based suggestions).

## Related Plan Docs

- `docs/SHOPPING_LIST_PLAN.md`
- `docs/REMOTE_ANDROID_PLAN.md`
- `docs/DND_IMPROV.md`
