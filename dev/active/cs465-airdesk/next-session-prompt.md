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

Pause broad combo/chained swipe data collection until the weak active-hand labels and decoder thresholds improve. Caden realized the one-hand default contaminated combo data: if one hand stays tracked in frame, the other hand's gesture may not become active. Feature export now emits per-hand rows, DTW/TCN windows are hand-scoped, event decoding merges decoded hand streams with cooldown suppression, and chart recording defaults to `--max-num-hands 2`.

Cleaned data:

- Deleted `data/recordings/sprint4-gpu-swipes-002-structured/*`.
- Deleted `data/labels/sprint4-gpu-swipes-002-structured/*`.
- Keep `data/recordings/sprint4-gpu-swipes-002-singles` for now, but treat it as legacy/single-hand-only until reviewed or recollected with two-hand background/rest conditions.

Next implementation chunk:

1. Check git status and read the active docs listed above.
2. Start from the 003+004 two-hand evidence already collected and avoid broad new data collection for a moment.
3. Improve active-hand weak-label assignment and decoder thresholds for shared per-hand TCN.
4. Verify mirrored/user-facing direction conventions before using raw dx sign for any label or gate.
5. If another data pass is needed, recollect targeted combo charts with explicit `--max-num-hands 2` using updated commands in `tracking-samples.md`.
6. Export features immediately after each kept chart and verify both hands appear as separate `hand_id` streams.
7. Run DTW spotting, TCN event decoding, and sequence scoring on the targeted charts.
8. Keep live desktop actions disabled.
9. Update README/tasks/tracking-samples/context docs with results and any revised collection commands.
10. Run `uv run ruff check .` and `uv run pytest`.
11. Commit and push.

Important evidence:

- Rule recognizer failed natural swipes.
- Same-batch DTW was optimistic.
- Plain DTW and TCN both missed held-out left swipes.
- Gated/window-feature DTW improved isolated holdout but remains tuned on existing evidence.
- Structured chained sessions showed plausible order but still misses/repeats.
- Live TCN is useful as diagnostic preview only.
- Shared per-hand TCN is now the recommended learned-model shape: one checkpoint run independently on each `hand_id` stream, followed by event decoding/merge/cooldown. Do not train separate `hand-0` and `hand-1` tracker-slot models yet.
- Two-hand motion-gated TCN target assignment exists via `--target-assignment motion-gated`. It gates on active-hand motion energy, not raw direction sign, because mirrored preview/raw camera direction conventions were brittle across 003+004.
- 003-to-004 shared per-hand TCN decoded holdout matched 27/48, missed 21, produced 11 false activations and 4 repeated fires. Improved, but not live-control-ready.
- T550 GPU MediaPipe path works through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`.
- Current UI for chart collection is a stable HUD with a progress bar and fixed upcoming cards.
