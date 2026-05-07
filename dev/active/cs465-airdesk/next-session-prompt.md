# AirDesk Next Session Prompt

Use this after clearing context:

---

You are working with Caden on AirDesk.

Project path:

`/home/caden/projects/AirDesk`

GitHub remote:

`git@github.com:caden-calderon/AirDesk.git`

Before doing anything:

1. Check `git status`.
2. Read `/home/caden/projects/AirDesk/AGENTS.md` if present; if not, follow the AGENTS instructions from this prompt.
3. Do not discard user changes.
4. Read:
   - `README.md`
   - `dev/active/cs465-airdesk/context.md`
   - `dev/active/cs465-airdesk/context-reset-prompt.md`
   - `dev/active/cs465-airdesk/handoff-prompt.md`
   - `dev/active/cs465-airdesk/dynamic-gesture-research.md`
   - `dev/active/cs465-airdesk/research-notes.md`
   - `dev/active/cs465-airdesk/sprint-4.md`
   - `dev/active/cs465-airdesk/sprint-5.md`
   - `dev/active/cs465-airdesk/tracking-samples.md`
   - `dev/active/cs465-airdesk/tasks.md`
5. Plan before editing.
6. Use `apply_patch` for manual edits.
7. Add/update tests alongside implementation.
8. Run `uv run ruff check .` and `uv run pytest`.
9. Commit meaningful chunks.
10. Push completed commits to `origin/main`.

Current stance:

AirDesk is a secondary spatial input layer for Hyprland, not a keyboard/mouse replacement. Keep claims narrow and evidence-based. MediaPipe is backend zero, not the project identity. Recording/replay/logging are core. Dry-run remains default until replay evidence supports guarded execution.

Current pivot:

Stop combo/chained swipe data collection until the two-hand path passes replay checks. Caden realized the one-hand default contaminated combo data: if one hand stays tracked in frame, the other hand's gesture may not become active. Feature export now emits per-hand rows, DTW/TCN windows are hand-scoped, event decoding merges decoded hand streams with cooldown suppression, and chart recording defaults to `--max-num-hands 2`.

Cleaned data:

- Deleted `data/recordings/sprint4-gpu-swipes-002-structured/*`.
- Deleted `data/labels/sprint4-gpu-swipes-002-structured/*`.
- Keep `data/recordings/sprint4-gpu-swipes-002-singles` for now, but treat it as legacy/single-hand-only until reviewed or recollected with two-hand background/rest conditions.

Next implementation chunk:

1. Check git status and read the active docs listed above.
2. Run replay checks on the two-hand feature/DTW/TCN/event-decoder path.
3. If replay checks pass, recollect combo charts with explicit `--max-num-hands 2` using the `sprint4-gpu-swipes-003-two-hand` commands in `tracking-samples.md`.
4. Export features immediately after each kept chart and verify both hands appear as separate `hand_id` streams.
5. Run DTW spotting, event decoding, and sequence scoring on the new charts.
6. Keep live desktop actions disabled.
7. Update README/tasks/tracking-samples/context docs with results and any revised collection commands.
8. Run `uv run ruff check .` and `uv run pytest`.
9. Commit and push.

Important evidence:

- Rule recognizer failed natural swipes.
- Same-batch DTW was optimistic.
- Plain DTW and TCN both missed held-out left swipes.
- Gated/window-feature DTW improved isolated holdout but remains tuned on existing evidence.
- Structured chained sessions showed plausible order but still misses/repeats.
- Live TCN is useful as diagnostic preview only.
- T550 GPU MediaPipe path works through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`.
- Current UI for chart collection is a stable HUD with a progress bar and fixed upcoming cards.
