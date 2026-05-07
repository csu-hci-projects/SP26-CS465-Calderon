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

Stop combo/chained swipe data collection until the pipeline supports two simultaneously visible hands. Caden realized the one-hand default is contaminating combo data: if one hand stays tracked in frame, the other hand's gesture may not become active. Feature export currently consumes `frame.hands[0]`, so just passing `--max-num-hands 2` is not enough.

Cleaned data:

- Deleted `data/recordings/sprint4-gpu-swipes-002-structured/*`.
- Deleted `data/labels/sprint4-gpu-swipes-002-structured/*`.
- Keep `data/recordings/sprint4-gpu-swipes-002-singles` for now, but treat it as legacy/single-hand-only until reviewed or recollected with two-hand background/rest conditions.

Next implementation chunk:

1. Read existing feature, DTW, TCN, chart-record, and evaluation code before editing.
2. Add/adjust chart collection recommendations so future combo sessions use `--max-num-hands 2`.
3. Update feature export to emit per-hand rows, not only `frame.hands[0]`.
   - Keep independent motion history per `hand_id`.
   - Preserve no-hand/background rows when no hands are visible.
   - Include `hand_count` and `hand_id`.
4. Update DTW/template recognition and live DTW preview to score per-hand streams without mixing hand histories.
5. Update TCN dataset/evaluation paths to handle per-hand rows safely.
6. Update event decoding/merge logic so events from both hands can be decoded and then merged with cooldown/repeated-fire suppression.
7. Add tests for two-hand feature rows, per-hand motion history, per-hand DTW candidates, and merged event ordering.
8. Do not collect new combo data until this path is implemented and tested.
9. Keep live desktop actions disabled.
10. Update README/tasks/tracking-samples/context docs with the two-hand workflow and exact next collection commands.
11. Run `uv run ruff check .` and `uv run pytest`.
12. Commit and push.

Important evidence:

- Rule recognizer failed natural swipes.
- Same-batch DTW was optimistic.
- Plain DTW and TCN both missed held-out left swipes.
- Gated/window-feature DTW improved isolated holdout but remains tuned on existing evidence.
- Structured chained sessions showed plausible order but still misses/repeats.
- Live TCN is useful as diagnostic preview only.
- T550 GPU MediaPipe path works through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`.
- Current UI for chart collection is a stable HUD with a progress bar and fixed upcoming cards.

