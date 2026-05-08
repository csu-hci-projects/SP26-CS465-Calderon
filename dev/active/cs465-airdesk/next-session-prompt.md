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
`airdesk/recognition/` package migration. The first deterministic per-hand
motion-event baseline now exists at `src/airdesk/gestures/motion.py`, with
replay-first CLI surfaces:

```bash
uv run airdesk gesture spot-motion --recording data/recordings/... --out data/evaluations/.../motion-candidates.json
uv run airdesk gesture evaluate-motion --recording data/recordings/... --labels data/labels/... --out data/evaluations/.../motion-summary.json
```

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
3. Run the motion baseline on existing labeled replay data and compare it against DTW/TCN summaries.
4. Start from the first replay result: `sprint4-swipes-001` default mapping matched 0/16 with 12 false activations; flipped mapping matched 5/16 with 12 false activations; stricter flipped dx 1.0 matched 3/16 with 5 false activations; `sprint4-chained-003` default matched 4/10 with 3 repeated fires and 0 false activations.
5. Use `spot-motion --labels ...` and the `motion_diagnostics` JSON rows to inspect false activations, repeated fires, missed events, hand ids, direction metadata, and label phase/event context.
6. Tune only enough to expose whether tracking/features are viable; do not turn this into another broad threshold sweep.
7. Add live diagnostic preview only after replay output is useful, or explicitly frame it as a low-level feature probe.
8. Keep broad combo collection paused unless the baseline exposes a specific tiny targeted calibration need.
9. Keep all dynamic swipe outputs in replay/diagnostic surfaces only.
10. Update README/context/tasks/tracking-samples with whatever changes.
11. Run `uv run ruff check .` and `uv run pytest`.
12. Commit and push.

Do not:

- Do not collect broad new combo data first.
- Do not keep sweeping current TCN thresholds.
- Do not wire learned/DTW/motion swipes to live desktop actions.
- Do not train separate tracker-slot models as a shortcut.
- Do not turn `deep-research-report.md` citations into paper citations without verifying them; the report is useful for architecture direction, not final bibliography text.

Current implementation direction after plan review:

The deterministic per-hand motion-event baseline consumes existing feature rows
and emits events like:

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

The first version emits existing `GestureCandidate` objects with metadata for
`window_start`, `window_end`, `peak_time`, normalized displacement, peak
velocity, direction consistency, and a stable peak/evidence id. Keep raw camera
`dx` sign as diagnostic until user-facing preview direction versus raw camera
direction is explicitly verified.

Focused motion-diagnostic evidence: `normal-desk-motion-negative-007` contains
background lateral motion strong enough to pass the baseline, with normalized dx
around `1.5-1.7` and direction consistency `1.0`; `swipe-left-positive-007`
misses because its label-time normalized dx is only about `0.28` after tracking
dropout/reset; `swipe-right-positive-007` matches under flipped mapping and
confirms that raw negative dx corresponds to that user-facing right swipe in
this take. Next code should stay replay-only and target intent/background
separation, tracking-drop diagnostics, low-motion valleys/reset evidence, and
peak identity for repeated fires.
