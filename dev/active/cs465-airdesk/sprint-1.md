# AirDesk Sprint 1 Plan

## Purpose

Sprint 1 turns the Sprint 0 skeleton into a live signal pipeline.

The goal is not to control the desktop yet. The goal is to answer whether AirDesk can reliably:

- open a real webcam,
- run a replaceable hand-tracking backend,
- normalize live landmarks into AirDesk data structures,
- record landmark streams,
- replay those streams deterministically,
- and inspect enough tracking quality to decide whether the next sprint should focus on gesture policy or tracker alternatives.

## Sprint Theme

> Get real hands into the replayable pipeline.

## Product / Research Stance

Sprint 1 should keep the research claims narrow:

- AirDesk is still evaluating mid-air gestures as a secondary input channel.
- MediaPipe is a backend candidate, not the product identity.
- Recorded landmark streams are first-class study/debug artifacts.
- Raw video recording should be avoided by default for privacy and storage reasons.
- No gesture should execute real desktop commands until tracking quality, gesture spotting, and safety policy are tested.

## Non-Goals

Do not attempt these in Sprint 1:

- real Hyprland action execution from live gestures
- cursor takeover
- virtual keyboard
- polished Wayland overlay
- Kinect integration
- ML training
- large gesture vocabulary
- user study execution
- destructive or risky desktop actions

These remain valid roadmap items, but Sprint 1 should keep the live pipeline observable and safe.

## Key Decisions

### Python Runtime

Sprint 0 currently allows Python `>=3.12`, and `uv` selected Python 3.14 on this machine. That is fine for the skeleton, but live tracking dependencies may not support Python 3.14 yet.

Sprint 1 should test MediaPipe and OpenCV dependency viability first. If MediaPipe does not support Python 3.14, constrain the project to Python 3.12 for now:

```toml
requires-python = ">=3.12,<3.13"
```

This is worth doing early because live tracking will be the dependency with the most packaging friction.

### Backend Zero

Use MediaPipe Hand Landmarker as backend zero if it installs and runs cleanly on Caden's machine.

If MediaPipe packaging or runtime behavior is poor, do not contort the architecture around it. Preserve the interface and document the failure. Then spike one fallback path:

- OpenCV capture plus mock/replay only, if tracking install is blocked.
- OpenVINO hand tracking, if MediaPipe is unstable but camera capture is fine.

### Capture Library

Use OpenCV for Sprint 1 camera capture/probing unless it creates unacceptable packaging friction.

Reason:

- fastest path to open `/dev/video*`
- enough for frame dimensions, FPS hints, and debug windows
- easy to keep behind `CaptureBackend`

Keep OpenCV behind the capture package boundary. Do not let recognizers depend on it.

### Recording Policy

Sprint 1 recordings should store normalized tracking frames and runtime events as JSONL.

Default behavior:

- record landmarks, metadata, confidence, handedness, frame dimensions, sequence, timestamps
- do not record raw RGB frames by default
- allow raw image/video hooks later, but require explicit opt-in

This keeps the study/debug format useful while reducing privacy and storage risk.

## Target CLI Shape

By the end of Sprint 1, these commands should exist or be expanded:

```text
airdesk doctor
airdesk camera list
airdesk camera probe --device /dev/video0
airdesk track --backend mediapipe --device /dev/video0 --max-frames 300 --show
airdesk record --backend mediapipe --device /dev/video0 --out data/recordings/sample.jsonl --max-frames 300
airdesk replay data/recordings/sample.jsonl
```

Useful flags:

- `--backend`: `mediapipe`, `mock`, `replay`
- `--device`: camera path or index
- `--max-frames`: bounded runs for tests/debugging
- `--show / --no-show`: debug visualization toggle
- `--dry-run`: do not dispatch real actions
- `--profile`: future-proof profile selection, even if action routing stays inert in Sprint 1

## Deliverables

### 1. Dependency and Runtime Decision

Acceptance criteria:

- OpenCV install path is tested with `uv`.
- MediaPipe install path is tested with `uv`.
- `pyproject.toml` reflects the actual supported Python version range.
- README documents the Python version requirement and backend dependency notes.
- If MediaPipe fails, the failure mode is documented in `research-notes.md` or Sprint 1 notes.

### 2. OpenCV Capture Backend

Acceptance criteria:

- `OpenCVCaptureBackend` implements `CaptureBackend`.
- `airdesk camera list` reports likely camera devices.
- `airdesk camera probe --device ...` attempts to open the camera, reads at least one frame, and reports:
  - open status
  - frame width/height
  - FPS if available
  - backend/library used
- Tests cover device-index/path parsing and probe result formatting without requiring a real webcam.

### 3. MediaPipe Tracking Backend

Acceptance criteria:

- MediaPipe backend maps detected hands into `TrackingFrame` / `NormalizedHand` / `HandLandmarks`.
- Backend exposes handedness and confidence when available.
- Backend handles zero-hand frames without error.
- Backend can be bounded by `--max-frames` for repeatable debugging.
- Tests cover landmark conversion with fake MediaPipe-like objects or adapter-level fixtures.

### 4. Track Command

Acceptance criteria:

- `airdesk track --backend mediapipe --max-frames N` runs the live pipeline without recording.
- The command prints compact per-frame or per-gesture debug summaries.
- Optional debug window draws landmarks if OpenCV display is available.
- The command exits cleanly on max frame count or keyboard interrupt.
- No desktop actions are executed by this command.

### 5. Record Command

Acceptance criteria:

- `airdesk record ... --out path.jsonl` records normalized tracking frames and events.
- Output directories are created if needed under ignored `data/recordings/`.
- A recording metadata/header event is written at start.
- A completion/interruption event is written at end when feasible.
- The recorded file can be replayed by existing replay tooling.

### 6. Replay + Recognizer Smoke Path

Acceptance criteria:

- Replay can optionally run the static recognizer over recorded frames.
- CLI reports counts for frames, hands, open palm candidates, fist candidates, and pinch candidates.
- At least one checked-in fixture remains tiny and deterministic.
- Tests prove recorded tracking frames still replay deterministically.

### 7. Tracking Quality Notes

Acceptance criteria:

- Document a short live-tracking report after Caden runs the camera:
  - camera used
  - resolution/FPS observed
  - lighting conditions
  - hand distance from camera
  - common failure cases
  - rough landmark jitter observations
- Decide whether Sprint 2 should proceed with command-mode policy or detour into tracker fallback/tuning.

## Recommended Implementation Order

1. Test and decide Python/runtime constraints for OpenCV and MediaPipe.
2. Add OpenCV as an optional capture dependency or direct dependency, depending install friction.
3. Implement camera probe/open backend with tests that do not require hardware.
4. Implement MediaPipe landmark adapter separately from the live backend.
5. Implement bounded live tracking loop.
6. Add `track` CLI command.
7. Add `record` CLI command and recording metadata events.
8. Extend replay summaries to run static recognizers.
9. Run a short real-camera smoke test on Caden's machine.
10. Document tracking quality and next-sprint decision.

## Risks and Mitigations

### Python 3.14 Dependency Gaps

Risk:

- MediaPipe or OpenCV wheels may not be available for Python 3.14.

Mitigation:

- Pin Sprint 1 to Python 3.12 if needed.
- Keep tracker imports lazy so tests and replay remain usable without live backend dependencies.

### Webcam Device Variability

Risk:

- `/dev/video0` may not be the useful camera, or camera formats may behave differently across machines.

Mitigation:

- Keep camera probing explicit.
- Support both numeric indices and device paths.
- Avoid hard-coding frame size assumptions.

### MediaPipe Instability

Risk:

- Tracking may be jittery or unreliable under desk lighting.

Mitigation:

- Record landmark streams before tuning recognizers.
- Log confidence and zero-hand frames.
- Keep OpenVINO as the first fallback candidate.

### Debug Window Problems on Wayland

Risk:

- OpenCV display windows may behave poorly under Wayland/Hyprland.

Mitigation:

- Make display optional.
- Keep CLI summaries and JSONL recordings as the primary observability path.
- Defer native overlay decisions.

## Definition of Done

Sprint 1 is done when:

- `uv sync --dev` works on the selected Python runtime.
- `uv run airdesk camera probe --device ...` can attempt a real camera probe.
- `uv run airdesk track --backend mediapipe --max-frames ...` runs or produces a documented backend-install/runtime failure.
- `uv run airdesk record ... --out data/recordings/...jsonl` produces a replayable landmark/event file when tracking is available.
- `uv run airdesk replay ...` summarizes recorded frames and recognizer candidates.
- `uv run pytest` passes.
- `uv run ruff check .` passes.
- README and active docs describe setup, limitations, and the Sprint 2 decision.

## Sprint 2 Gate

At the end of Sprint 1, choose one of two paths:

### Path A: Tracking Is Good Enough

Proceed to Sprint 2 command-mode interaction:

- clutch/open-palm hold policy
- gesture cooldowns
- profile binding resolution
- dry-run action routing from gestures
- Hyprland execution behind explicit opt-in

### Path B: Tracking Is Not Good Enough

Spend Sprint 2 on tracking robustness:

- smoothing/filtering
- lighting/exposure notes
- camera setting controls
- OpenVINO fallback spike
- adjusted gesture primitives based on recorded data

The decision should be based on recorded data, not vibes.

