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
3. Read `/home/caden/projects/AirDesk/AGENTS.md`.
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
3. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/plan.md`
4. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/architecture.md`
5. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/research-notes.md`
6. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/dynamic-gesture-research.md`
7. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-3.md`
8. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-4.md`
9. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-5.md`
10. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tracking-samples.md`
11. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tasks.md`

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
- Hyprland 0.54.3 supports `hyprctl dispatch movecursor x y`, which is how the first real cursor mode works.
- `ydotool`/`wtype` were not installed during the cursor spike, so click/drag injection remains pending.

## Dynamic Gesture Strategy

Do not jump straight to "train an LSTM," and do not spend the next sprint comparing every model family.

The current research conclusion is:

> AirDesk's best current bet is intent-gated gesture phrases plus a small causal TCN trained on phase-labeled continuous landmark features.

Why:

- continuous OS input is a gesture spotting and intent problem,
- isolated clip accuracy is misleading,
- rolling buffers create boundary, chaining, and false activation issues,
- rule/DTW recognizers are useful as safety/debug scaffolding, low-data fallback, and calibration tools,
- causal TCN is the preferred first learned model once continuous phase-labeled logs exist,
- LSTM/GRU is deferred unless the causal TCN path fails or a later comparison is worth the time,
- ST-GCN/Transformer are later options after dataset growth.

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

## Current Next Task

Implement **DTW holdout evaluation** for `sprint4-swipes-001`.

Recommended chunk:

1. Add a repeatable CLI or helper for train/test DTW evaluation.
2. Split the current batch into train/test, for example:
   - train: 6 left + 6 right + negatives,
   - test: 2 left + 2 right + negatives.
3. Calibrate DTW on train recordings only.
4. Evaluate on held-out positives and all/held-out negatives.
5. Export JSON summary with matched, missed, false activations, repeated fires, and latency.
6. Document results in `tracking-samples.md` and `tasks.md`.
7. Run `uv run ruff check .` and `uv run pytest`.
8. Commit and push.

If holdout stays promising, ask Caden to record a 60-90 second chained continuous session with multiple left/right swipes and normal motion between them. Then evaluate DTW on that conductor-style recording before starting the causal TCN.

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
