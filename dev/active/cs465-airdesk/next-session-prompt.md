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
boundaries. Caden read a deep research report and agreed this is a real
architecture shift. The current TCN work is useful evidence and infrastructure,
but it is still too close to sliding-window phase classification. AirDesk needs
a continuous gesture spotting architecture:

```text
per-hand normalized feature streams
  -> motion activity proposal
  -> recognizer/scorer
  -> event decoder
  -> command queue
  -> mode/profile/safety policy
```

The first deterministic per-hand motion-event baseline now exists at
`src/airdesk/gestures/motion.py`, with replay-first CLI surfaces:

```bash
uv run airdesk gesture spot-motion --recording data/recordings/... --out data/evaluations/.../motion-candidates.json
uv run airdesk gesture spot-motion --recording data/recordings/... --labels data/labels/...labels.json --out data/evaluations/.../motion-candidates.json
uv run airdesk gesture evaluate-motion --recording data/recordings/... --labels data/labels/... --out data/evaluations/.../motion-summary.json
```

That baseline did its job: it exposed lower-level failure modes. Do not keep
polishing it forever. The first TCN v2 surface now exists; use old replay data
as a regression suite before collecting the targeted V2 slice.

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
- TCN v2 is now implemented as an initial replay/evaluation surface:
  `build-tcn-dataset --target-mode v2-evidence`, `train-tcn-v2`, and
  `evaluate-tcn-v2`.
- The v2 targets are framewise decoder-facing evidence heads:
  `intentional_motion`, `stroke_left`, `stroke_right`, `start`, and `end`.
  Windows remain causal compute context, not semantic gesture labels.
- The first old-data v2 smoke is complete. It validated the surface but not the
  recognizer: `sprint4-swipes-001` label-assigned evidence had healthy
  source-frame counts, a 5-epoch model trained cleanly, but event replay matched
  `0/16` at `0.35` thresholds and only `1/16` at permissive `0.30` thresholds;
  `sprint4-chained-003` stayed `0/10`.
- V2 manifest/evaluation cleanup from that smoke is in place: summaries include
  `evidence_frame_counts`, no-hand windows use explicit `__no_hand__` stream ids,
  and offline v2 evaluation decodes a deduplicated all-row causal-context stream.
- A first behavior-preserving CLI cleanup pass has started. The public entrypoint
  remains `airdesk.cli:app`, but offline TCN commands moved to
  `src/airdesk/cli_tcn.py`, label/feature commands moved to
  `src/airdesk/cli_labeling.py`, small camera/profile/Hyprland commands moved
  to `src/airdesk/cli_system.py`, and shared helper functions moved to
  `src/airdesk/cli_support.py`. `src/airdesk/cli.py` still owns live
  tracking/runtime/preview commands.

Next-session assignment:

1. Re-skim `recognition-v2-plan.md`, especially the Review Conclusion, TCN V2 Shape, and Phase E.
2. Inspect current package boundaries around:
   - `src/airdesk/features/`
   - `src/airdesk/gestures/`
   - `src/airdesk/ml/`
   - `src/airdesk/analysis/`
   - `src/airdesk/cli.py` and the extracted `src/airdesk/cli_*.py` modules
3. Inspect existing TCN manifest/model/evaluation code enough to design the
   smallest TCN v2 slice.
4. Review the TCN v2 data/model/evaluation surface:
   causal per-hand context windows, one shared model shape applied independently
   to each `hand_id`, and decoder-facing evidence outputs instead of one argmax
   gesture label per semantic window.
5. Treat old replay data as regression coverage, not final proof. The first v2
   smoke says the path is coherent but underconfident/direction-confused on old
   data.
6. Improve the next V2 target/calibration/data plan before collecting: repeated
   events need enough positive frame evidence, direction heads need separation,
   and negatives need intentional-motion rejection.
7. Plan or collect a small targeted
   continuous V2 training/testing slice: repeated same-direction swipes,
   alternating swipes, weak/tiny lefts, natural desk-motion negatives, hand
   enters/leaves frame, near/far starts, and two visible hands with one resting.
8. Keep broad combo collection paused until that targeted V2 slice has
   event-level replay evidence.
9. Keep all dynamic swipe outputs in replay/diagnostic/preview surfaces only.
10. Continue behavior-preserving CLI refactors only if they unblock the V2 work;
    `src/airdesk/cli.py` still owns live tracking/runtime/preview paths.
11. Update README/context/tasks/tracking-samples/next-session docs with whatever changes.
12. Run `uv run ruff check .` and `uv run pytest`.
13. Commit and push.

Do not:

- Do not collect broad new combo data first.
- Do not keep sweeping current TCN thresholds.
- Do not treat old data as final V2 proof; use it as regression coverage.
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

It uses hand-normalized displacement, peak velocity, direction consistency,
low-motion valleys, duration bounds, per-hand stream separation, and duplicate
suppression by peak identity. Its purpose now is diagnostic/regression support,
not live control and not a blocker to TCN v2.

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
