# AirDesk Context Reset Prompt

Use this after clearing context:

---

You are working with Caden on **AirDesk**.

Project path:

`/home/caden/projects/AirDesk`

GitHub remote:

`git@github.com:caden-calderon/AirDesk.git`

Before doing anything:

1. Check `git status`.
2. Do not discard user changes.
3. Read `/home/caden/projects/AirDesk/AGENTS.md` if present; if not, follow the
   instructions in this prompt.
4. Read the active docs listed below.
5. Plan before editing.
6. Use `apply_patch` for manual file edits.
7. Add or update tests alongside implementation.
8. Run `ruff` and `pytest` before finishing.
9. Commit meaningful chunks.
10. Push commits to `origin/main` when the chunk is complete.

## Project Summary

AirDesk is a CS465 HCI / 3DUI research project and personal computing prototype. It explores webcam-based mid-air hand gestures as an OS-level spatial input layer for a Hyprland Linux desktop.

The motivation is **situationally impaired interaction**: times when keyboard/mouse input is inconvenient, unavailable, dirty, painful, or physically costly.

The broader product ambition is a pluggable, profile-driven desktop control system where webcam, optional depth sensors, hand gestures, keyboard, mouse, and desktop context can blend into command, cursor, media, presentation, accessibility, virtual keyboard, and hybrid interaction modes.

Important stance:

- Do not frame gestures as a full replacement for keyboard/mouse.
- Research claims must stay narrow and evidence-based.
- Product ambition can remain broad.
- Do not make MediaPipe the identity of the project; treat it as one replaceable backend.
- Recording/replay/logging are core architecture.
- Dry-run is the default until reliability evidence supports guarded real execution.
- Cursor mode and virtual keyboard are later separate scopes.

## Read These First

1. `/home/caden/projects/AirDesk/README.md`
2. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/context.md`
3. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/recognition-v2-plan.md`
4. `/home/caden/projects/AirDesk/deep-research-report.md` if present
5. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/plan.md`
6. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/architecture.md`
7. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/research-notes.md`
8. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/dynamic-gesture-research.md`
9. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-3.md`
10. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-4.md`
11. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-5.md`
12. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tracking-samples.md`
13. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tasks.md`

## Current Implementation State

AirDesk currently has:

- Python/uv project skeleton,
- ruff and pytest,
- typed data structures for frames, landmarks, gestures, profiles, actions, and event logs,
- profile schema and sample profiles,
- dry-run action target,
- Hyprland action wrapper,
- capture/tracking interfaces,
- OpenCV camera backend,
- MediaPipe Tasks Hand Landmarker backend,
- MediaPipe model/threshold/hand-count tuning flags,
- camera probing/mode reporting,
- JSONL recording/replay,
- replay analysis,
- mock/replay backend,
- static recognizers for open palm, fist, and pinch,
- command-mode policy,
- profile binding resolver,
- dry-run runtime path,
- mirrored live webcam preview,
- visual landmark/gesture indicators,
- live tuning and benchmark commands,
- prompted preview-first collection with keep/redo/skip,
- runtime `--events-out` JSONL logging with session start/finish events,
- guarded opt-in Hyprland execution for allowlisted commands,
- explicit `airdesk cursor run` mode where pinch-hold moves the real Hyprland cursor through `movecursor`,
- continuous label schema and CLI (`label init`, `label validate`, `label add-phase`, `label add-event`, `label suggest`),
- deterministic feature export,
- rule and DTW recognizer evaluation,
- dependency-free DTW/template calibration and replay evaluation.

Important live findings:

- `/dev/video0` needs OpenCV index normalization plus `--fourcc MJPG` to honor `640x480 @ 30 FPS`.
- CLI live commands default to one tracked hand for latency.
- MediaPipe Tasks exposes model asset path and confidence/hand-count options, not the old `model_complexity` flag.
- `/dev/video0` does not appear to support 60 FPS; requesting `640x480 @ 60 FPS MJPG` falls back to 30 FPS.
- On Caden's Hyprland/Arch/T550 setup, use `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu` when testing MediaPipe GPU tracking on the T550. Plain `--hand-delegate gpu` may still use Intel/Mesa EGL. Confirm the MediaPipe startup log contains `OpenGL ES 3.2 NVIDIA` and `NVIDIA T550 Laptop GPU`.
- `airdesk benchmark` reports live timing slices; short bounded smokes showed T550 inference around 4 ms per frame, but capture remains camera-paced around 30 FPS.
- Two-hand combo data remains the next blocker, but the broad collection pause has moved from "pipeline not implemented" to "weak labels and decoder need tightening." The old one-hand default was wrong for alternating-hand/chained combos: MediaPipe collection defaulted to one hand, and feature export used to consume only `frame.hands[0]`. AirDesk now exports per-hand feature rows, keeps independent per-hand motion history, scores DTW/TCN streams per hand, and decodes hand streams before merging events with cooldown suppression. The chart recorder defaults to `--max-num-hands 2`. The `sprint4-gpu-swipes-002-structured` combo recordings were deleted.
- Recognition V2 pivot: Caden added `deep-research-report.md`; the plan review is complete and the first deterministic per-hand motion-event baseline is implemented at `src/airdesk/gestures/motion.py`. Use `airdesk gesture spot-motion` for replay JSON candidates/diagnostics and `airdesk gesture evaluate-motion` for label evaluation. First bounded replay evidence says this is diagnostic only: `sprint4-swipes-001` default raw-positive-dx mapping matched 0/16 with 12 false activations; flipped mapping matched 5/16 with 12 false activations; stricter flipped dx 1.0 matched 3/16 with 5 false activations; `sprint4-chained-003` default matched 4/10 with 3 repeated fires and 0 false activations. Label-aware motion diagnostics showed background lateral motion can pass motion-only gates, weak-left examples can fall below the displacement gate after tracking dropout/reset, and raw camera sign must stay explicit. Current decision: do not keep polishing the motion baseline forever. The first TCN v2 surface now exists through `build-tcn-dataset --target-mode v2-evidence`, `train-tcn-v2`, and `evaluate-tcn-v2`. The first old-data v2 smoke validates the surface but not quality: a quick `sprint4-swipes-001` model matched `0/16` at `0.35` thresholds, `1/16` at permissive `0.30` thresholds, and `0/10` on `sprint4-chained-003`. The first staff-level review/refactor chunk is complete: shared hand/no-hand feature stream helpers now drive DTW, motion, TCN dataset windows, and live preview; V2 no-hand/tracking-drop rows stay background-only; V2 start/end targets are scoped to tracked intentional evidence inside each labeled event. Keep live actions disabled.
- First pre-training TCN v2 architecture cleanup is complete. New v2 checkpoints are schema `2` residual dilated causal TCNs with per-frame layer normalization/dropout, default 29-frame receptive field, weighted/focal BCE with extra sparse-boundary weighting, per-head calibration threshold metadata, per-head metrics, and batched manifest prediction. Schema-1 v2 checkpoints still load for replay compatibility. `evaluate-tcn-v2` now lets `start` boost boundary-backed stroke activation and lets `end` force release/background pressure instead of treating boundary heads as passive metadata. The next step is to replay-check this stronger architecture on old regression data before targeted V2 collection.
- CLI cleanup has continued without changing the public entrypoint. `airdesk.cli:app` now owns app wiring plus `doctor` / `analyze`, while offline TCN commands are in `src/airdesk/cli_tcn.py`, replay/offline gesture diagnostics are in `src/airdesk/cli_gesture_replay.py`, label/features commands are in `src/airdesk/cli_labeling.py`, small camera/profile/Hyprland commands are in `src/airdesk/cli_system.py`, shared helper functions are in `src/airdesk/cli_support.py`, live preview/status formatting helpers are in `src/airdesk/cli_live.py`, live tracking/watch diagnostic commands are in `src/airdesk/cli_live_commands.py`, recording/collection/chart workflows are in `src/airdesk/cli_recording.py`, runtime/live-action commands and guarded execution policy are in `src/airdesk/cli_runtime.py`, and shared tracker construction is in `src/airdesk/cli_tracking.py`. `src/airdesk/cli.py` is now about 60 LOC. TCN v2 frame-evidence target construction now lives in `src/airdesk/ml/tcn_v2_evidence.py`, sequence-evidence training/prediction lives in `src/airdesk/ml/tcn_v2_train.py`, replay decoder/evaluation glue lives in `src/airdesk/analysis/tcn_v2.py`, and generic manifest/window construction remains in `src/airdesk/ml/dataset.py`. A `vulture` dead-code scan found no confirmed removable production code in the latest pass; its high-confidence hits were required keyword names in runner protocols/test doubles. Caden wants the next session to keep pushing the review/refactor more aggressively and do what is right/best, while preserving public behavior and dry-run safety. The next cleanup target is likely a focused CLI test-file split or smaller production audit before switching into collection.
- New two-hand shared TCN evidence: batches 003+004 are local under ignored `data/`. Use one shared TCN checkpoint independently on each `hand_id` stream, not separate tracker-slot models. Motion-gated two-hand manifests keep weak prompt-time labels from training a stationary visible hand as active. Recovery-inclusive TCN collapsed into `recovery`; `phase-stroke` removed that class but still failed live. Caden saw live `dx > 0.50` while stroke probabilities stayed flat, so stop rescuing the current TCN with threshold sweeps.
- `airdesk gesture diagnose-tcn-events` exists for decoded TCN failure reports. On the 003-to-004 split, most misses had nearest same-gesture candidates outside the 0.5 s tolerance window; increasing match tolerance to 3.0 s raised matches to 36/48 while leaving 9 false activations. Treat chart labels as prompt-time weak labels until active-hand/timestamp alignment improves.
- Hyprland 0.54.3 supports `hyprctl dispatch movecursor x y`, which is how the first real cursor mode works.
- `ydotool`/`wtype` were not installed during the cursor spike, so click/drag injection remains pending.

## Dynamic Gesture Strategy

Do not jump straight to "train an LSTM," and do not spend the next sprint comparing every model family.

The original Sprint 3 research conclusion was:

> AirDesk's best current bet is intent-gated gesture phrases plus a small causal TCN trained on phase-labeled continuous landmark features.

The May 2026 update refines that:

> The current TCN window classifier is a scaffold, not the destination. AirDesk should move toward continuous gesture spotting: position-invariant skeleton features, phase/event labels, event decoding, and a small hybrid recognizer that can later grow toward graph/transformer memory.

May 2026 two-hand update:

> Combo/chained gestures should support both hands visible at once. The next implementation chunk should add per-hand feature rows, per-hand recognizer scoring, and cross-hand event merging before collecting new combo data.

Why:

- continuous OS input is a gesture spotting and intent problem,
- isolated clip accuracy is misleading,
- rolling buffers create boundary, chaining, and false activation issues,
- rule/DTW recognizers are useful as safety/debug scaffolding, low-data fallback, and calibration tools,
- causal TCN is still the preferred first learned scaffold because it is easier to debug than a transformer on small data,
- the useful target is stream/phase/event prediction, not fixed-window classification,
- LSTM/GRU is deferred unless the causal TCN path fails or a later comparison is worth the time,
- ST-GCN/Transformer are later options after dataset growth and event decoding.

The target feel is "conducting a choir for your computer": subtle, low-fatigue, intentional wrist/finger phrases, not dragging the whole arm across the screen.

## Current Roadmap

### Sprint 3: Pilot-Safe Live Command Mode

Goal:

Make live command mode observable, logged, and pilot-safe.

Main tasks:

- record/analyze deliberate live samples,
- add runtime `--events-out` JSONL logs,
- add session start/finish runtime events,
- add intent-gated phrase recognizer foundation,
- add flick/swipe left/right recognition or defer with evidence,
- add continuous positive/negative recording protocol,
- decide whether point left/right is reliable enough,
- show command-mode state in `run --show`,
- add pause/kill-switch behavior,
- add guarded opt-in Hyprland execution,
- add Caden-only pilot protocol.

### Sprint 4: Gesture Dataset, Labeling, and Causal TCN Recognition

Goal:

Build the dataset, labeling, feature pipeline, and first causal TCN recognizer.

Main tasks:

- define continuous gesture label schema,
- add `airdesk label init` and `airdesk label validate`,
- add deterministic feature extraction,
- export features,
- add `airdesk gesture evaluate`,
- train/evaluate one small causal TCN on continuous sessions,
- keep rule/DTW as fallback/calibration rather than the main bet,
- explicitly defer LSTM/GRU unless TCN disappoints,
- document the Sprint 5 recognizer decision.

### Sprint 5: Study Tooling, Pilot, and Paper Evidence

Goal:

Convert the prototype into study evidence.

Main tasks:

- add `studies/pilot-0.md`,
- define study/trial event schema,
- add study logging CLI,
- add CSV/summary export,
- integrate runtime logs with study session/task IDs,
- document keyboard/mouse baseline workflow,
- run Caden-only baseline and AirDesk dry-run pilot,
- optionally run guarded execute-mode pilot if safe,
- add paper outline with evidence placeholders.

## Recent Dataset And Evidence

Caden recorded `data/recordings/sprint4-swipes-001`:

- 8 `swipe-left-positive` takes,
- 8 `swipe-right-positive` takes,
- 8 `normal-desk-motion-negative` takes,
- 238 frames per take at about 29.65 FPS.

Generated local artifacts are intentionally ignored:

- labels: `data/labels/sprint4-swipes-001`
- features: `data/features/sprint4-swipes-001`
- rule evaluations: `data/evaluations/sprint4-swipes-001`
- DTW model: `data/models/gestures/caden-dtw-sprint4-swipes-001.json`
- DTW evaluations: `data/evaluations/sprint4-swipes-001-dtw`

Rule recognizer evidence:

- 16 intended positive swipe events,
- 0 matched,
- 16 missed,
- 1707 positive-take candidates,
- 1543 positive-take false activations,
- 1221 negative-take false activations, mainly crude `fist` and `pinch`.

DTW baseline evidence on the same calibration/evaluation batch:

- 16 intended,
- 16 matched,
- 0 missed,
- 18 candidates,
- 2 false activations,
- 0 repeated fires,
- about 0.44 s mean latency,
- 0 candidates on the 8 negative/background recordings.

This DTW result is promising but optimistic because calibration and evaluation used the same small batch. Do not claim live reliability from it yet.

DTW holdout evidence:

- Command: `uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout.json --train-per-gesture 6 --test-per-gesture 2 --train-negatives 6 --test-negatives 2`
- Split: train on takes 001-006 for each positive gesture and negative/background group; test on takes 007-008.
- Result: 4 intended held-out swipes, 2 matched, 2 missed, 2 candidates, 0 false activations, 0 repeated fires, about 0.40 s mean latency on matched events.
- Per gesture: `swipe_right` matched 2/2; `swipe_left` matched 0/2.
- Diagnostic update: the holdout JSON includes closest rejected DTW windows. The left threshold is clamped by similar negative motion, and loosening it enough to recover both held-out left swipes introduces false activations. Treat this as a feature-separation problem, not a simple threshold issue.
- Gated variant: `uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary-gated.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --train-per-gesture 6 --test-per-gesture 2 --train-negatives 6 --test-negatives 2 --negative-distance-margin 1.3 --min-palm-dx-fraction 0.65`
- Gated result: 4 intended held-out swipes, 4 matched, 0 missed, 4 candidates, 0 false activations, 0 repeated fires, about 0.36 s mean latency.
- Gated interpretation: promising, but tuned after seeing this holdout. It needs validation on a fresh chained continuous recording before live-control use.
- Interpretation: DTW is still useful as a personalized baseline, but the left-swipe holdout misses mean it is not ready for live swipe control or reliability claims.
- TCN scaffold evidence: deterministic manifest/window building, optional PyTorch training, same-batch evaluation, and holdout evaluation exist. Same-batch TCN matched 16/16 with 1 false activation, but holdout TCN matched 2/4 and missed both held-out left swipes.
- Feature diagnostics: `airdesk gesture diagnose-features` now compares the same holdout split. On `sprint4-swipes-001`, held-out left swipes are weaker/slower than train-left examples (`palm_dx` about `0.181` vs `0.235`, normalized displacement about `1.387` vs `1.857`, max speed about `3.230` vs `5.163`) while label/frame alignment is roughly one frame inside the event interval.
- Window-feature update: feature export now includes causal trailing-window signed displacement, hand-scale-normalized displacement, peak horizontal velocity, and direction consistency. DTW saved-model inference remains backward-compatible with old feature vectors.
- Rerun evidence: with regenerated features, DTW plus `--negative-distance-margin 0.75` matched 4/4 held-out swipes with 0 false activations on the isolated holdout. TCN still matched only 2/4 and missed both held-out left swipes. On the structured chained session, the looser `1.3` gated DTW window-feature variant scored 9/10 in order with 1 extra-or-wrong-order detection, while the conservative `0.75` variant under-detected badly.
- Fresh timestamp-aware continuous evidence: Caden recorded `data/recordings/sprint4-chained-003/chained-structured-swipes-001.jsonl` with 10 seconds active / 10 seconds rest and intended sequence `R L R R L L R R L L`. Coarse half-window labels were created. Old gated DTW matched 7/10 event windows and scored 8/10 in order with 0 extra sequence detections. The looser window-feature gated DTW matched 8/10 event windows and scored 10/10 in order, but produced 3 extra-or-wrong-order sequence detections and 2 repeated fires. Conservative `0.75` DTW matched 1/10, and TCN matched 3/10.
- `airdesk gesture watch-tcn` exists as a live/replay classifier preview. It shows the latest TCN target/probabilities in the webcam preview and prints non-background predictions by default. It does not trigger desktop actions.

Fresh chained-session evidence:

- Recording: `data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl`
- Health: 2669 frames, 1384 hand-present frames, about 29.66 FPS, roughly 90 seconds.
- Caden reports roughly 15 swipes with some natural movement and back-to-back swipes.
- Command: `uv run airdesk gesture spot-dtw --recording data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --out data/evaluations/sprint4-chained-001/gated-dtw-candidates.json`
- Result: 16 DTW candidates, 10 `swipe_right` and 6 `swipe_left`, with timestamps recorded in `tracking-samples.md`.
- Caveat: this is candidate spotting, not event-level accuracy yet. Human timestamp review or labels are needed before reporting matched/missed/false-activation metrics.

Structured chained-session evidence:

- Recording: `data/recordings/sprint4-chained-002/chained-structured-swipes-001.jsonl`
- Intended movement-direction sequence: `R L R R L L R R L L`
- Health: 2670 frames, 1220 hand-present frames, about 29.66 FPS, roughly 90 seconds.
- Gated DTW detected sequence: `R L R R L R R L`
- Command: `uv run airdesk gesture score-sequence --candidates data/evaluations/sprint4-chained-002/gated-dtw-candidates.json --expected-sequence "R L R R L L R R L L" --out data/evaluations/sprint4-chained-002/gated-dtw-sequence-score.json`
- Result: 8/10 matched in order, 2 missed-or-wrong-order gestures, 0 extra-or-wrong-order detections.
- Interpretation: promising for a lightweight personalized baseline, but still not reliable enough for live desktop actions.

## Current Next Task

The first recognizer-pivot implementation pass has landed:

- position-invariant TCN feature preset: `--feature-preset stream-invariant`;
- phase target mode: `--target-mode phase` with `background`, `stroke_left`, `stroke_right`, and `recovery`;
- label vocabulary support for `recovery` / `reset`;
- coarse sequence helper: `airdesk label add-sequence --sequence "R L R R L L"`;
- replayable event decoder: `airdesk gesture evaluate-tcn --event-decoder` and `airdesk gesture decode-candidates`.

Initial evidence:

- Current window-feature gated DTW remains best on isolated holdout: 4/4 matched, 0 false activations, 0 repeated fires.
- Current window-feature TCN remains weak: 2/4 matched, 1 false activation.
- TCN plus event decoder matched 3/4 on isolated holdout but introduced 2 false activations at permissive thresholds.
- Stream-invariant phase TCN plus event decoder matched 2/4 with 1 false activation.
- Event-decoded chained DTW candidates on `sprint4-chained-003` reduced repeated fires from 2 to 0, but dropped matches from 8/10 to 6/10.

The deterministic motion baseline and label-aware `motion_diagnostics` have now
made the known lower-level failures visible. Old replay data should stay in use,
but only as a regression suite. It should not block the project from starting
TCN v2 or stand in as final proof of V2 quality.

Do not wire DTW, motion, or TCN swipes into live desktop actions yet.

Recommended next chunk:

1. Use the TCN v2 data/model/evaluation surface on old replay data:
   causal per-hand context windows, shared model shape over `hand_id` streams,
   and decoder-facing outputs rather than one argmax label per semantic window.
2. Use old replay data (`sprint4-swipes-001`, `sprint4-chained-003`, and the
   motion diagnostic files) as a regression suite for sign convention,
   weak-left/tracking-drop, negative-motion false activations, and repeated
   fires.
3. After the regression check, collect a targeted continuous V2 training/test
   slice: repeated same-direction swipes, alternating swipes, weak/tiny lefts,
   natural desk-motion negatives, hand enters/leaves frame, near/far starts, and
   two visible hands with one resting.
5. Keep broad combo collection paused until that V2 slice has event-level replay
   evidence.

## Useful Commands

```bash
uv sync --dev
uv sync --dev --extra live
uv run airdesk --help
uv run pytest
uv run ruff check .
uv run airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG
uv run airdesk view --device /dev/video0
uv run airdesk tune --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --show
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label open-palm-hold --out data/recordings/open-palm-hold.jsonl
uv run airdesk analyze data/recordings/open-palm-hold.jsonl
uv run airdesk run --backend replay --recording tests/fixtures/replay-one-frame.jsonl --profile configs/profiles/study-safe.toml --dry-run
uv run airdesk gesture calibrate --kind dtw --recording data/recordings/sprint4-swipes-001/swipe-left-positive-001.jsonl --labels data/labels/sprint4-swipes-001/swipe-left-positive-001.labels.json --out data/models/gestures/caden-dtw.json
uv run airdesk gesture evaluate --recognizer dtw --model data/models/gestures/caden-dtw.json --recording data/recordings/sprint4-swipes-001/swipe-left-positive-001.jsonl --labels data/labels/sprint4-swipes-001/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-dtw.json
```

---
