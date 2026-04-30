# AirDesk Sprint 0 Plan

## Purpose

Sprint 0 is the foundation sprint. The goal is not to make AirDesk impressive yet. The goal is to make future impressive work cheap.

This sprint should answer:

- Can the repo support a long-lived, modular Python application?
- Can AirDesk capture, normalize, record, and replay tracking data?
- Can actions be routed safely through dry-run and Hyprland adapters?
- Can the codebase support profiles, modes, and future UI surfaces without a rewrite?
- Can we measure tracking quality before depending on it?

## Sprint Theme

> Build the skeleton that lets AirDesk become a real OS spatial input layer.

## Non-Goals

Do not attempt these in Sprint 0:

- full gesture vocabulary
- real cursor takeover
- virtual keyboard
- Kinect integration beyond planning hooks
- LSTM/GRU/TCN model training
- polished control center
- destructive Hyprland actions
- multi-OS support

These are valid roadmap items, but Sprint 0 should make them easier later.

## Technical Direction

### Package and Tooling

Use a Python package with `src/airdesk/`.

Recommended tooling:

- `uv` for Python project/dependency management
- `ruff` for formatting/linting
- `pytest` for tests
- `mypy` or `pyright` later if type-checking friction is low

Use Python 3.11 or 3.12 unless a dependency forces otherwise.

### First Runtime Shape

Create a CLI-first daemon/debug app before building a polished UI:

```text
airdesk doctor
airdesk camera list
airdesk camera probe
airdesk track --backend mediapipe --dry-run
airdesk record --backend mediapipe --out data/recordings/...
airdesk replay data/recordings/...
airdesk hyprland dry-run workspace-next
```

The CLI should make early debugging and study tooling simple.

### UI Direction

Start with a debug OpenCV window or lightweight local visualizer for landmark debugging.

For a real Wayland overlay, likely options:

- GTK + layer-shell: native Wayland overlay path, Python-accessible through GObject introspection, good for status/OSD.
- Eww: useful for simple widgets and Hyprland-adjacent desktop UI, less ideal for custom real-time gesture visualization.
- Web/Tauri control center: likely better later for a full settings app than for the first overlay.

Sprint 0 should not lock the final UI. It should keep daemon state observable over logs/events so any UI can subscribe later.

### Tracking Direction

Implement interfaces first:

- `CaptureBackend`
- `HandTrackerBackend`
- `RecordingWriter`
- `RecordingReader`

Then add MediaPipe as backend zero if dependency setup works.

Recorded replay is required in Sprint 0. It can start with landmark JSONL recordings before full video/frame recordings.

### Action Direction

Implement actions behind adapters:

- `DryRunActionTarget`
- `HyprlandActionTarget`

Sprint 0 can verify safe commands only:

- workspace next/previous
- focus direction
- fullscreen toggle only in dry-run unless explicitly enabled

Do not make gesture recognition call shell commands directly.

## Proposed Repo Structure

```text
src/airdesk/
  __init__.py
  cli.py
  capture/
  tracking/
  state/
  gestures/
  modes/
  profiles/
  actions/
  logging/
  study/
  config/
tests/
configs/
  profiles/
studies/
scripts/
data/
  recordings/
```

`data/` should be ignored by Git except placeholder files if needed.

## Sprint 0 Deliverables

### 1. Project Skeleton

Acceptance criteria:

- `uv sync` creates a working environment.
- `uv run airdesk --help` works.
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- `src/airdesk/` package exists with clear module boundaries.

### 2. Core Types

Acceptance criteria:

- Typed dataclasses exist for:
  - captured frame metadata
  - hand landmarks
  - normalized hand state
  - tracking frame
  - gesture candidate
  - gesture confirmation
  - profile
  - action request/result
  - event log entry
- Unit tests cover serialization for core event/log types.

### 3. Config and Profiles

Acceptance criteria:

- Baseline profile config format exists.
- At least two sample profiles exist:
  - `study-safe`
  - `window-manager`
- Config loader validates required fields.
- Unit tests cover good and bad profile files.

### 4. Action Layer

Acceptance criteria:

- `DryRunActionTarget` logs intended actions without executing them.
- `HyprlandActionTarget` wraps `hyprctl dispatch` behind one module.
- Commands are represented as typed action requests.
- Tests cover action request mapping without requiring Hyprland.

### 5. Capture and Tracking Interfaces

Acceptance criteria:

- Capture backend interface exists.
- Hand tracker backend interface exists.
- Mock/replay backend exists.
- MediaPipe backend is scaffolded or implemented depending on dependency viability.
- Camera probing command reports available device/open status.

### 6. Recording and Replay

Acceptance criteria:

- Landmark/event JSONL recording format exists.
- Replay command can feed recorded tracking frames back through the pipeline.
- A tiny fixture recording exists in `tests/fixtures/`.
- Tests prove replay produces deterministic frames/events.

### 7. First Gesture Primitives

Acceptance criteria:

- Rule recognizer interface exists.
- Open palm, fist, and pinch can be represented as gesture candidates.
- Recognizers can operate on replayed/mock hand states.
- Tests cover at least synthetic open palm/fist/pinch cases.

### 8. Developer Docs

Acceptance criteria:

- README includes setup, test, and first CLI commands.
- `architecture.md` stays aligned with actual package structure.
- `tasks.md` reflects completed Sprint 0 items.

## Recommended First Implementation Order

1. Project skeleton and tooling.
2. Core types and event schema.
3. Config/profile schema.
4. Dry-run action target.
5. Hyprland action wrapper.
6. Recording/replay format.
7. Mock/replay tracking backend.
8. Camera probe command.
9. MediaPipe backend spike.
10. Synthetic gesture primitive tests.

This order keeps tests useful before the webcam stack gets involved.

## Key Research Findings For Sprint 0

- `uv` uses `pyproject.toml` as the project root and creates a local `.venv`, which fits a repo-local development workflow.
- MediaPipe Hand Landmarker supports video/live-stream modes and uses tracking in those modes to avoid palm detection on every frame, reducing latency.
- Hyprland exposes synchronous command IPC and socket2 event IPC. The synchronous socket must be handled carefully; `hyprctl dispatch` is safer for the first adapter.
- GTK layer-shell is a viable native Wayland overlay path and supports Python through GObject introspection, but the first sprint should avoid committing to a full UI framework.

## Open Questions Before Implementation

- Should Python target 3.11 or 3.12?
- Should CLI use Typer, Click, or stdlib `argparse`?
- Should configs be YAML or TOML?
- Should runtime events be internal Python queues first, or should we expose a local socket/API immediately?
- Should full video recording be avoided by default for privacy and storage reasons?
- Should MediaPipe be a required dependency immediately or an optional extra?

## Current Recommendation

Use:

- Python 3.12 if MediaPipe installs cleanly, otherwise 3.11
- `uv`
- `ruff`
- `pytest`
- Typer for CLI if dependency budget feels acceptable; otherwise `argparse`
- TOML for project config and YAML or TOML for profiles
- JSONL for event/landmark recordings
- MediaPipe as optional backend dependency if possible

