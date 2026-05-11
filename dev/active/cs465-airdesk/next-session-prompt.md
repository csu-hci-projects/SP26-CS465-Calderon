# AirDesk Next Session Prompt

You are working with Caden on AirDesk at:

`/home/caden/projects/AirDesk`

This is a context-reset continuation after the all-IPN TCN v2 live-preview
session. Start with a staff-level review stance, preserve user changes, plan
before editing, use `apply_patch` for manual edits, add/update tests with code,
run `uv run ruff check .` and `uv run pytest`, then commit/push meaningful
chunks to `origin/main`.

Before editing:

1. Run `git status`.
2. Read `AGENTS.md` if present.
3. Read:
   - `README.md`
   - `deep-research-report.md` if present
   - `dev/active/cs465-airdesk/context.md`
   - `dev/active/cs465-airdesk/recognition-v2-plan.md`
   - `dev/active/cs465-airdesk/architecture.md`
   - `dev/active/cs465-airdesk/plan.md`
   - `dev/active/cs465-airdesk/tasks.md`
   - `dev/active/cs465-airdesk/public-dataset-survey.md`
   - `dev/active/cs465-airdesk/tracking-samples.md`
4. Do not discard user changes.

Current code/docs state:

- Latest pushed commit at closeout: the docs closeout commit
  `Refresh AirDesk mode-aware recognition docs`.
- Historical sprint plans and stale reset/handoff prompts were archived under
  `dev/archive/cs465-airdesk/2026-05-11/`.
- Working tree should be clean.
- No live desktop actions should be wired to learned/DTW/motion gestures.

Best all-IPN checkpoint:

`data/models/gestures/tcn-v2-ipn-all-w16-80ep-h64-l4.pt`

Held-out all-IPN final-frame head eval:

- `gesture_macro_f1=0.521`
- `gesture_micro_f1=0.742`
- gesture-positive top-1 final-frame accuracy `0.757`
- top-3 `0.934`

Boundary eval on the same checkpoint:

- about `start_f1=0.455`, `end_f1=0.468` at +/-0.5s
- about `0.53` at +/-1.0s

Important interpretation:

- The all-IPN evaluation is fairer than the old two-head false-fire evaluation:
  every non-`D0X` IPN gesture is a named evidence head, and `D0X` remains
  background.
- Fair IPN held-out scoring does **not** mean safe AirDesk command use.
- Caden's live preview showed high-confidence false activations from normal hand
  presence/motion, especially `Throw up`, `Open twice`, `Zoom out`, and
  point-like heads.
- Latest parsed live calibration log:
  `data/logs/live-ipn-all-tcn-v2-calibration-20260511-122007.jsonl`
  had 328 predictions. Top heads above `0.80`: `Open twice` 28, `Throw up` 26,
  `Throw left` 11, `Throw down` 9, `Point one finger` 8.

Current product/architecture decision:

Do **not** enable all 13 IPN heads globally.

Use mode groups:

- all-IPN/debug mode: show everything, no actions.
- command mode: only robust command gestures after AirDesk negative testing;
  keep `Throw up`, `Open twice`, and `Zoom out` disabled globally for now.
- cursor mode: click/double-click plus zoom heads only. IPN point heads are
  suppressed because direct MediaPipe pose logic is cleaner if pointing is
  needed later.
- zoom/media mode: zoom heads only, also disabled globally by default.

Next implementation target:

Build a mode-aware learned-recognition filter around TCN v2 preview/evaluation.

Recommended first slice:

1. Add a small vocabulary/mode map for custom TCN v2 heads.
2. Add CLI options to `watch-tcn-v2` for diagnostic recognition mode, while
   preserving an all-head debug view.
3. For custom heads, support per-head thresholds, top-vs-runner-up margin, short
   persistence, and cooldown before showing "Recognized" as confident.
4. Add a replay/log evaluator for live prediction JSONL so the latest
   calibration log can be scored under proposed filters without rerunning live.
5. Update dashboard/history so it shows enabled heads and explains when a head
   is suppressed by mode/filter.
6. Add focused tests for mode membership, thresholds, margins, persistence, and
   JSONL replay scoring.

Only after that:

- Plan or collect a small AirDesk hard-negative set: open-hand idle, accidental
  pointing, reaching, resting, cursor-like motion, and normal desk motion with
  hands visible.
- Consider fine-tuning/retraining with those hard negatives.

Official IPN model note:

The official IPN baselines/checkpoints are RGB/video models such as
ResNeXt/ResNet variants, not MediaPipe-landmark TCNs. They are useful references
or a separate heavier fallback experiment, but they are not drop-in replacements
for AirDesk's current feature stream and do not remove the need for
AirDesk-specific mode gating and hard negatives.

Safety stance:

- Keep learned/DTW/motion gestures out of live Hyprland actions.
- Keep preview/evaluation no-action unless Caden explicitly asks otherwise.
- Cursor movement remains its own guarded/dry-run-first path; pointer click/drag
  injection is not enabled yet.
