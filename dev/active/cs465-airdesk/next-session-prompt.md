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
   - `deep-research-report.md` if present
   - `dev/active/cs465-airdesk/context.md`
   - `dev/active/cs465-airdesk/recognition-v2-plan.md`
   - `dev/active/cs465-airdesk/architecture.md`
   - `dev/active/cs465-airdesk/plan.md`
   - `dev/active/cs465-airdesk/tasks.md`
   - `dev/active/cs465-airdesk/context-reset-prompt.md`
   - `dev/active/cs465-airdesk/handoff-prompt.md`
   - `dev/active/cs465-airdesk/dynamic-gesture-research.md`
   - `dev/active/cs465-airdesk/research-notes.md`
   - `dev/active/cs465-airdesk/sprint-4.md`
   - `dev/active/cs465-airdesk/sprint-5.md`
   - `dev/active/cs465-airdesk/tracking-samples.md`
5. Plan before editing.
6. Use `apply_patch` for manual edits.
7. Add/update tests alongside implementation.
8. Run `uv run ruff check .` and `uv run pytest`.
9. Commit meaningful chunks.
10. Push completed commits to `origin/main`.

Current stance:

AirDesk is a secondary spatial input layer for Hyprland, not a keyboard/mouse replacement. Keep claims narrow and evidence-based. MediaPipe is backend zero, not the project identity. Recording/replay/logging are core. Dry-run remains default until replay evidence supports guarded execution.

Current pivot:

The Recognition V2 plan has now been reviewed against the current code
boundaries. It survives, but the first implementation slice should be narrower
than a broad package refactor. Caden read a deep research report and agreed this
is a real architecture shift. The current TCN work is useful evidence and
infrastructure, but it is still too close to sliding-window phase classification.
AirDesk needs a continuous gesture spotting architecture:

```text
per-hand normalized feature streams
  -> motion activity proposal
  -> recognizer/scorer
  -> event decoder
  -> command queue
  -> mode/profile/safety policy
```

Do not start by building TCN v2. Do not start with a wholesale
`airdesk/recognition/` package migration. The next implementation slice is a
small deterministic per-hand motion-event baseline at the existing gesture
boundary, reusing `features.landmarks.FeatureRowStream`,
`gestures.decoder.EventDecoder`, and the current evaluation utilities where they
fit.

Important evidence:

- Rule recognizer failed natural swipes.
- Same-batch DTW was optimistic.
- Plain DTW and TCN both missed held-out left swipes.
- Gated/window-feature DTW improved isolated holdout but remains tuned on existing evidence.
- Structured chained sessions showed plausible order but still misses/repeats.
- The one-hand default contaminated combo data; the old `sprint4-gpu-swipes-002-structured` batch was deleted.
- Feature export now emits per-hand rows with independent motion history.
- DTW/TCN windows are hand-scoped and decoded hand streams are merged.
- Shared per-hand model shape remains correct: one shared scorer/checkpoint run independently on each visible `hand_id`, then decoded/merged.
- Do not train separate `hand-0` / `hand-1` tracker-slot models unless stable physical-hand identity labels exist.
- Recovery-inclusive TCN collapsed into `recovery` during live preview.
- `phase-stroke` removed recovery but did not solve live recognition.
- Caden saw live `dx > 0.50` while `L=` / `R=` stayed flat, so the failure is not just a too-high motion gate.
- A more sensitive 0.20 motion-gate TCN improved 003-to-004 replay to 37/48, but with 18 false activations; diagnostic only.
- Chart labels are prompt-timing weak labels, not exact active-hand truth.
- Motion-peak auto-refinement worsened held-out TCN performance and should be used only for diagnostics/manual review.
- T550 GPU MediaPipe path works through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`.
- Keep live desktop actions disabled.

Next-session assignment:

1. Re-skim `recognition-v2-plan.md`, especially the Review Conclusion and Phase C.
2. Inspect current package boundaries around:
   - `src/airdesk/features/`
   - `src/airdesk/gestures/`
   - `src/airdesk/ml/`
   - `src/airdesk/analysis/`
   - `src/airdesk/cli.py`
3. Implement the smallest safe slice: deterministic per-hand motion-event baseline.
4. Prefer a small module such as `src/airdesk/gestures/motion.py` before creating a new recognition package.
5. Add replay-first CLI/evaluation output before any live preview:
   - `gesture spot-motion` or equivalent JSON candidate export;
   - `gesture evaluate-motion` or equivalent summary against labels.
6. Add tests for:
   - per-hand stream separation;
   - repeated same-direction swipes as separate events;
   - background/idle rejection;
   - merged event ordering across hands.
7. Keep broad combo collection paused unless the new baseline exposes a specific tiny targeted calibration need.
8. Keep all dynamic swipe outputs in replay/diagnostic surfaces only.
9. Update README/context/tasks/tracking-samples with whatever changes.
10. Run `uv run ruff check .` and `uv run pytest`.
11. Commit and push.

Do not:

- Do not collect broad new combo data first.
- Do not keep sweeping current TCN thresholds.
- Do not wire learned/DTW/motion swipes to live desktop actions.
- Do not train separate tracker-slot models as a shortcut.
- Do not turn `deep-research-report.md` citations into paper citations without verifying them; the report is useful for architecture direction, not final bibliography text.

Suggested first implementation direction after plan review:

Build a deterministic per-hand motion-event baseline that consumes existing
feature rows first, then live `FeatureRowStream` rows, and emits events like:

```text
GestureEvent(
  name="swipe_left" | "swipe_right",
  hand_id="hand-0",
  start_time=...,
  peak_time=...,
  end_time=...,
  confidence=...,
  evidence={dx, peak_velocity, direction_consistency}
)
```

It should use hand-normalized displacement, peak velocity, direction consistency, low-motion valleys, duration bounds, per-hand stream separation, and duplicate suppression by peak identity. The point is to prove whether AirDesk's current tracking/features can spot live swipes before committing to TCN v2.

The first version should emit existing `GestureCandidate` objects with metadata
for `window_start`, `window_end`, `peak_time`, normalized displacement, peak
velocity, direction consistency, and a stable peak/evidence id. Keep raw camera
`dx` sign as diagnostic until user-facing preview direction versus raw camera
direction is explicitly verified.
