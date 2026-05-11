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
- `dev/active/cs465-airdesk/public-dataset-survey.md`
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
- TCN v2 is now implemented as a replay/evaluation/live-preview surface:
  `build-tcn-dataset --target-mode v2-evidence`, `train-tcn-v2`,
  `evaluate-tcn-v2`, `diagnose-tcn-v2-events`, and no-action `watch-tcn-v2`.
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
- TCN v2 frame-evidence target construction is now isolated in
  `src/airdesk/ml/tcn_v2_evidence.py`; generic manifest/window construction
  remains in `src/airdesk/ml/dataset.py`.
- TCN v2 sequence-evidence training/prediction is now isolated in
  `src/airdesk/ml/tcn_v2_train.py`, and replay decoder/evaluation glue is
  isolated in `src/airdesk/analysis/tcn_v2.py`. Public exports remain stable
  through `airdesk.ml` and `airdesk.analysis`; compatibility wrappers remain in
  `src/airdesk/analysis/evaluation.py` for older direct imports.
- The first pre-training TCN v2 architecture cleanup is complete. New
  `train-tcn-v2` checkpoints use schema version `2`, a residual dilated causal
  TCN with per-frame layer normalization/dropout, default `hidden_channels=32`,
  `levels=3`, and a 29-frame receptive field at `kernel_size=3`. Training uses
  weighted/focal BCE with extra weighting for sparse `start` / `end` positives,
  stores per-head calibration thresholds and metrics, and predicts manifest
  windows in batches. Schema-1 v2 checkpoints still load for replay
  compatibility.
- `evaluate-tcn-v2` now uses boundary heads in the decoder contract: `start`
  evidence can boost a moderate stroke into activation, while `end` evidence
  suppresses stroke scores and raises background/release pressure. Boundary
  heads are no longer metadata-only. V2 evaluation/diagnostics also support
  `--early-match-tolerance-seconds` so causal peaks just before a hand-labeled
  event start are not counted as both a miss and a false activation.
- The stronger schema-2 v2 replay check is complete. Current-code
  `sprint4-swipes-001` evidence counts are `intentional_motion=187`,
  `stroke_left=88`, `stroke_right=99`, `start=16`, `end=16`. A 25-epoch
  schema-2 model reached `train_frame_accuracy=0.990` and
  `validation_frame_accuracy=0.985`; metadata stored calibration thresholds
  `intentional_motion=0.85`, `stroke_left=0.90`, `stroke_right=0.90`,
  `start=0.80`, and `end=0.75`, with weak validation `start` F1 (`0.316`).
  At `activation=0.35`, `release=0.2`, `min_peak=0.35`, isolated replay scored
  `9/16` without early tolerance, but diagnostics showed all 7 misses peaked
  0.02-0.22 s before label start. With `--early-match-tolerance-seconds 0.25`,
  isolated swipes scored `16/16` with 5 false activations, all normal desk
  negatives. The same model scored `8/10` on `sprint4-chained-003` with
  0 false activations, 3 repeated fires, and about 1.75 s mean latency.
- `airdesk gesture watch-tcn-v2` is now available for a no-action live/replay
  schema-2 v2 feel-test. It loads `causal_tcn_v2_evidence` checkpoints, runs the
  shared model per visible hand stream, and defaults to a resizable dashboard
  with webcam landmarks, per-hand evidence bars, decoded-gesture history,
  emit/peak delay, prediction/candidate counts, and tracker timing. The old
  compact overlay remains available with `--preview-layout camera`, and
  `--camera-buffer-size` defaults to `1` to reduce camera backlog where OpenCV
  honors it. The command decodes candidates through the same start/end-aware
  decoder, avoids flushing open live events before release evidence, can write
  prediction/candidate JSONL with `--events-out`, and does not call runtime
  policy or action targets.
- A staff-level cleanup chunk is complete. Shared hand/no-hand feature stream
  helpers live in `src/airdesk/feature_streams.py` and are re-exported through
  `airdesk.features`; DTW, motion, TCN dataset windows, and live preview now use
  the same stream grouping contract. V2 evidence generation keeps no-hand /
  tracking-drop rows background-only and scopes `start` / `end` targets to
  tracked intentional evidence inside each labeled event interval.
- CLI cleanup has continued without changing the public entrypoint. The public
  entrypoint remains `airdesk.cli:app`; offline TCN commands live in
  `src/airdesk/cli_tcn.py`, replay/offline gesture diagnostics live in
  `src/airdesk/cli_gesture_replay.py`, label/feature commands live in
  `src/airdesk/cli_labeling.py`, small camera/profile/Hyprland commands live in
  `src/airdesk/cli_system.py`, shared CLI helpers live in
  `src/airdesk/cli_support.py`, live preview/status formatting helpers live in
  `src/airdesk/cli_live.py`, live tracking/watch diagnostic commands live in
  `src/airdesk/cli_live_commands.py`, recording/collection/chart workflows live
  in `src/airdesk/cli_recording.py`, runtime/live-action commands and guarded
  execution policy live in `src/airdesk/cli_runtime.py`, and shared tracker
  construction lives in `src/airdesk/cli_tracking.py`. `src/airdesk/cli.py` is
  now about 60 LOC of app wiring plus `doctor` / `analyze`.

Latest public-dataset update:

The primary-source survey is now in
`dev/active/cs465-airdesk/public-dataset-survey.md`. Recommendation: start with
IPN Hand because it is continuous, RGB-webcam based, CC BY 4.0, includes natural
non-gesture hand motion, and has lateral `G05 Throw left` / `G06 Throw right`
classes that can proxy AirDesk left/right atomic evidence. IPN does not contain
AirDesk swipe gestures. Jester is still useful later for large clip-level
pretraining but is less aligned with AirDesk's boundary/continuous-spotting
problem. EgoGesture and ChaLearn ConGD
are continuous alternatives, but EgoGesture is egocentric RGB-D and requires an
agreement, while ChaLearn is broad RGB-D challenge data with less direct desktop
gesture mapping. IPN HandS may be valuable if its refined skeleton annotations
are available locally, but verify access before building around it.

Implemented importer:

```bash
uv run airdesk public-data ipn-convert \
  --videos-dir data/public/ipn/videos \
  --annotations-dir data/public/ipn/annotations-download \
  --out-dir data/public/ipn/airdesk \
  --split train \
  --limit 1 \
  --manifest-out data/public/ipn/airdesk/tcn-v2-ipn-smoke-manifest.json \
  --mapping-out data/public/ipn/airdesk/ipn-airdesk-mapping.csv
```

It runs downloaded IPN MP4 videos through MediaPipe, writes AirDesk replay JSONL,
maps only IPN `G05 Throw left` / `G06 Throw right` to AirDesk's left/right
atomic evidence labels as a lateral-motion proxy, exports normal
`FrameFeatureRow` CSVs, and can build a `stream-invariant-v2` `v2-evidence`
manifest. Other IPN classes stay background/negative for the first left/right
TCN pass. Raw public dataset downloads and generated artifacts should stay
ignored under `data/public/`.

2026-05-10 update: official IPN Hand annotations and all five video archives are
downloaded under ignored `data/public/ipn/`; extraction produced 200 `.avi`
files in `data/public/ipn/videos/`. The importer now supports the official Drive
annotation filenames directly, and a one-video 120-frame smoke conversion
succeeded.

All-IPN pre-launch update: the generated all-IPN labels/manifests have now had
a staff-level review pass. Labels match the official non-`D0X` annotation events
and keep `D0X` as background; manifests use `stream-invariant-v2` and only
`data/public/ipn/...` artifacts. The correct held-out quality metric for the
all-IPN checkpoint is `airdesk gesture evaluate-tcn-v2-heads`, not the older
AirDesk left/right `evaluate-tcn-v2` event decoder. The reviewed launch script
is `scripts/train-ipn-all-tcn-v2.sh`.

All-IPN training update: the 0.8s-window `h64/l4` run finished cleanly and
scored held-out `gesture_macro_f1=0.505`, `gesture_micro_f1=0.695`. A controlled
1.6s-window rerun with the same architecture is the current best checkpoint:
`data/models/gestures/tcn-v2-ipn-all-w16-80ep-h64-l4.pt`, with
`gesture_macro_f1=0.521`, `gesture_micro_f1=0.742`, gesture-positive top-1
final-frame accuracy `0.757`, and top-3 `0.934`. A wider 1.6s `h96` run scored
lower held-out `gesture_macro_f1=0.503`, so do not assume more width is the next
fix. `start` / `end` stayed weak across runs and should be treated as a boundary
target/evaluation problem before event-decoder use.

Next-session assignment:

Continue from the new public-dataset / atomic-gesture planning pivot. The
feature-contract audit is complete: new V2 manifests should use
`--feature-preset stream-invariant-v2`, which excludes absolute palm position,
raw image-space palm motion, `hand_scale`, `hand_count`, and unscaled
finger/pinch geometry from classifier input while preserving those fields in
logs/dashboard diagnostics. Caden has also collected an initial targeted V2
local slice under `data/recordings/v2-*` with train/holdout/invariance
recordings, but the next session should pause before blindly training on only
that data. The strategic question is whether public datasets, especially IPN
Hand, can provide useful atomic gesture priors.

Architectural stance:

- Train the TCN to detect atomic events and boundaries, not combo classes.
- Good TCN outputs are still `intentional_motion`, `stroke_left`,
  `stroke_right`, `start`, and `end`.
- A second command-grammar layer should turn emitted event streams such as
  `R, R, L` into optional combos. Do not train labels like `right_right_left`
  unless a future experiment proves this is necessary.
- Public datasets are training aids, not AirDesk's final success metric.
  AirDesk source-held-out recordings remain the authority for pass/fail.

1. Check `git status`, reread the active docs, and verify the latest tests if
   the checkout has changed.
2. Use the 1.6s `h64/l4` all-IPN checkpoint as the current public-data prior
   candidate unless a new controlled run beats it on the official held-out split.
3. Review the weak `start` / `end` heads with boundary tolerance or wider target
   labels before treating all-IPN evidence as an event decoder.
4. Decide which IPN heads are useful as AirDesk priors and compare
   AirDesk-only vs IPN-only vs IPN-pretrain/AirDesk-fine-tune or hybrid training
   on the AirDesk source-held-out V2 recordings.
5. Keep the old atomic `G05` / `G06` checkpoint interpretation narrow: it was
   IPN-only lateral evidence, not an AirDesk swipe model.
6. Preserve current behavior unless a bug is found and fixed intentionally. Keep
   live desktop actions disabled/dry-run by default.
7. Update README/context/tasks/tracking-samples/next-session docs with whatever
   changes.
8. Run `uv run ruff check .` and `uv run pytest`.
9. Commit meaningful chunks and push to `origin/main`.

Do not:

- Do not collect broad new combo data first.
- Do not train every combo as its own TCN class.
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
