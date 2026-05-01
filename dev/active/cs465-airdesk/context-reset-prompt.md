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
3. Read the active docs listed below.
4. Plan before editing.
5. Use `apply_patch` for manual file edits.
6. Add or update tests alongside implementation.
7. Run `ruff` and `pytest` before finishing.
8. Commit meaningful chunks.
9. Push commits to `origin/main` when the chunk is complete.

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
- live tuning and benchmark commands.

Important live findings:

- `/dev/video0` needs OpenCV index normalization plus `--fourcc MJPG` to honor `640x480 @ 30 FPS`.
- CLI live commands default to one tracked hand for latency.
- MediaPipe Tasks exposes model asset path and confidence/hand-count options, not the old `model_complexity` flag.

## Dynamic Gesture Strategy

Do not jump straight to "train an LSTM."

The current research conclusion is:

> Use intent-gated gesture phrases with rule/DTW baselines now, then train/evaluate a causal TCN on phase-labeled continuous data. LSTM/GRU should be included as a baseline, not the primary bet.

Why:

- continuous OS input is a gesture spotting and intent problem,
- isolated clip accuracy is misleading,
- rolling buffers create boundary, chaining, and false activation issues,
- DTW/template matching is strong for low-data personalized conductor-like gestures,
- TCN is the preferred first learned model once continuous phase-labeled logs exist,
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

### Sprint 4: Gesture Dataset, Labeling, and Model Evaluation

Goal:

Build the dataset, labeling, and model-evaluation loop.

Main tasks:

- define continuous gesture label schema,
- add `airdesk label init` and `airdesk label validate`,
- add deterministic feature extraction,
- export features,
- add DTW/template recognizer,
- add `airdesk gesture evaluate`,
- compare rule, DTW, LSTM/GRU, and causal TCN if enough data exists,
- document model-selection decision for Sprint 5.

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

## Current Next Task

Start Sprint 3 implementation.

Recommended first implementation chunk:

1. Add runtime `--events-out` JSONL logging over the replay backend.
2. Add session start and session finish events.
3. Add tests for replay runtime event log creation.
4. Update README/tasks if the CLI changes.
5. Run `uv run ruff check .` and `uv run pytest`.
6. Commit and push.

After that, move to live sample recording/analysis and the intent-gated phrase recognizer foundation.

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
```

---
