# AirDesk Sprint 2 Plan

## Purpose

Sprint 2 should turn the live pipeline from "it runs" into "we can judge whether it is safe to build commands on it."

Sprint 1 proved:

- OpenCV can open the camera.
- MediaPipe Tasks Hand Landmarker can run locally.
- normalized tracking frames can be recorded and replayed.
- the CLI can track, record, and summarize frames.

Sprint 1 did not yet prove:

- the selected camera mode is fast enough,
- landmarks are stable enough for command gestures,
- deliberate open palm / fist / pinch samples are recognized from live data,
- or that command-mode policy should execute desktop actions.

Sprint 2 should therefore focus on camera quality, landmark stability, live gesture samples, and a dry-run-only command-mode policy path.

## Sprint Theme

> Make live tracking measurable, then route gestures safely in dry-run.

## Product / Research Stance

Keep AirDesk framed as an evidence-based secondary input layer.

- Do not claim gestures replace keyboard/mouse.
- Do not execute real Hyprland commands from live gestures by default.
- Do not tune only by eye. Record short samples and replay them.
- Treat camera settings, false negatives, false positives, and fatigue as first-class design data.
- Keep MediaPipe replaceable. Sprint 2 should improve the tracking pipeline without tying recognizers to MediaPipe internals.

## Non-Goals

Do not attempt these in Sprint 2:

- always-on desktop control
- real cursor takeover
- virtual keyboard
- full overlay/control center
- Kinect integration
- ML training
- destructive actions
- a formal user study

The sprint can create enough instrumentation to make those future choices saner.

## Key Decisions

### Sprint 2 Path

Choose a hybrid of Sprint 1's two gate paths:

1. **Tracking robustness first** because `/dev/video0` reported `1920x1080` at only `5.00` FPS.
2. **Dry-run command policy second** once replayed hand-in-frame samples show recognizable open palm, fist, and pinch events.

This keeps momentum toward interaction design while refusing to pretend the live signal is already good.

### Camera Strategy

Prefer lower resolution and higher FPS over high-resolution frames for gesture control.

Target candidates:

- `640x480 @ 30 FPS`
- `1280x720 @ 30 FPS`
- anything stable above `20 FPS`

Sprint 2 should add capture controls or probing utilities that can request width, height, and FPS, then report what the camera actually gives.

### Gesture Policy Strategy

Implement command-mode policy in dry-run only:

- open palm held for ~300 ms enters a listening window
- fist cancels listening
- pinch can produce a dry-run observation event
- no real Hyprland execution from live tracking unless explicitly added later

This lets AirDesk test the clutch/Midas Touch logic before controlling the desktop.

### Measurement Strategy

Every deliberate live sample should produce a JSONL artifact under ignored `data/recordings/`.

Recommended sample set:

- `open-palm-hold`
- `fist-hold`
- `pinch-hold`
- `no-hand`
- `normal-desk-motion`

Each sample should be short, around 5-10 seconds, and replayable.

## Target CLI Shape

By the end of Sprint 2, these commands should exist or be expanded:

```text
airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30
airdesk camera modes --device /dev/video0
airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --label open-palm-hold --duration 5 --out data/recordings/open-palm-hold.jsonl
airdesk replay data/recordings/open-palm-hold.jsonl --recognize
airdesk analyze data/recordings/open-palm-hold.jsonl
airdesk run --backend replay --recording data/recordings/open-palm-hold.jsonl --profile study-safe --dry-run
```

If `camera modes` is awkward through OpenCV alone, it can shell out to `v4l2-ctl` when available and degrade gracefully when not.

## Deliverables

### 1. Camera Controls and Mode Reporting

Acceptance criteria:

- Capture backend accepts requested width, height, and FPS.
- Camera probe can request width/height/FPS and reports both requested and actual values.
- CLI supports `--width`, `--height`, and `--fps` for probe, track, and record.
- If `v4l2-ctl` is installed, `airdesk camera modes` reports available formats/FPS.
- If `v4l2-ctl` is missing, the command explains how to install/use it without failing the rest of AirDesk.
- Tests cover option plumbing and formatting without requiring a webcam.

### 2. Recording Labels and Durations

Acceptance criteria:

- `record` supports `--label` and `--duration`.
- Recording start event includes label, backend, device, requested camera settings, and model path.
- Recording end event includes frame count, duration, and interrupted/completed status.
- Duration-based recording exits cleanly even without `--max-frames`.
- Tests cover metadata events and directory creation.

### 3. Replay Analysis

Acceptance criteria:

- Add an analysis path that reports:
  - frame count
  - hand-present frame count
  - average FPS from timestamps
  - candidate counts for open palm, fist, pinch
  - longest consecutive run of each candidate
  - rough landmark jitter for selected stable landmarks when a hand is present
- Analysis works on checked-in tiny fixtures and ignored local recordings.
- Tests cover analysis on synthetic/replay fixtures.

### 4. Live Gesture Sample Protocol

Acceptance criteria:

- Add `dev/active/cs465-airdesk/tracking-samples.md`.
- Document how to record the five recommended samples.
- Document local observations from at least one deliberate hand-in-frame sample.
- Decide whether current camera settings are acceptable for command-mode experiments.

### 5. Command Mode State Machine

Acceptance criteria:

- Add a mode/policy component that consumes `GestureCandidate` streams.
- Open palm held for configurable duration enters listening mode.
- Fist cancels listening mode.
- Listening mode expires after a short timeout.
- Policy emits typed events such as `ModeChanged`, `GestureConfirmed`, and `ActionRequested`.
- Unit tests cover:
  - open-palm hold enters listening
  - short open palm does not enter listening
  - fist cancels listening
  - timeout exits listening
  - action requests are dry-run only in `study-safe`

### 6. Profile Binding Resolution

Acceptance criteria:

- Given a confirmed gesture and active profile, AirDesk can resolve a matching binding.
- Confidence thresholds and cooldowns are applied.
- Destructive bindings remain blocked unless explicitly allowed by profile and command path.
- Tests cover `study-safe` and `window-manager` bindings without executing Hyprland.

### 7. Dry-Run Runtime Path

Acceptance criteria:

- Add a safe runtime command, likely:

```text
airdesk run --backend replay --recording tests/fixtures/... --profile study-safe --dry-run
```

- Runtime can read replayed frames, recognize primitive gestures, apply command-mode policy, resolve bindings, and send requests to `DryRunActionTarget`.
- Runtime logs events to stdout and optionally JSONL.
- Live backend is allowed only with dry-run defaults.
- No real Hyprland commands are executed by Sprint 2 runtime.

## Recommended Implementation Order

1. Add camera width/height/FPS options to capture, probe, track, and record.
2. Add duration and label metadata to recordings.
3. Add replay analysis utilities and tests.
4. Record deliberate hand-in-frame samples locally.
5. Update tracking notes with observed FPS, detection, and jitter.
6. Implement command-mode state machine over synthetic candidate streams.
7. Add profile binding resolution and cooldown logic.
8. Add dry-run runtime path over replay backend.
9. Optionally test dry-run runtime on a live recording if tracking quality is adequate.
10. Run `ruff`, `pytest`, and update docs/tasks.

## Risks and Mitigations

### Camera FPS Remains Too Low

Risk:

- The camera may stay near 5 FPS at default settings, making gestures laggy.

Mitigation:

- Request lower resolution.
- Report actual capture settings instead of assuming.
- Use replay analysis to measure real frame intervals.
- If needed, spend Sprint 3 on camera/backend robustness before real command control.

### Gesture Rules Fail on Real Landmarks

Risk:

- Synthetic open palm/fist/pinch rules may not match MediaPipe landmarks well enough.

Mitigation:

- Use recorded samples to adjust thresholds.
- Keep threshold changes profile/config-driven where practical.
- Add fixtures generated from real landmark samples after privacy review.

### False Activations

Risk:

- Command-mode policy may trigger from normal hand motion.

Mitigation:

- Require clutch hold.
- Keep all routing dry-run.
- Record `no-hand` and `normal-desk-motion` samples.
- Track false activation counts before real actions.

### Scope Creep Into Desktop Control

Risk:

- It will be tempting to wire gestures to Hyprland once dry-run works.

Mitigation:

- Keep real execution behind a future explicit opt-in.
- Finish replay analysis and live sample notes first.
- Only promote to real actions after reliability data supports it.

## Definition of Done

Sprint 2 is done when:

- camera probe/track/record can request and report width/height/FPS,
- at least one deliberate hand-in-frame sample has been recorded and analyzed,
- replay analysis reports FPS, hand presence, primitive candidate counts, and simple stability metrics,
- command-mode policy is implemented and tested over synthetic/replay inputs,
- profile binding resolution is implemented and tested,
- `airdesk run ... --dry-run` can route replayed gestures to dry-run actions,
- no real desktop action is triggered from live gestures,
- `uv run ruff check .` passes,
- `uv run pytest` passes,
- README/tasks/research notes are updated with the Sprint 2 outcome.

## Sprint 3 Gate

At the end of Sprint 2, choose:

### Path A: Dry-Run Command Mode Is Reliable

Proceed to real opt-in Hyprland command execution:

- explicit `--execute` or profile flag
- safer command subset only
- visible status feedback
- kill switch / pause
- longer live pilot with logging

### Path B: Tracking or Policy Is Not Reliable

Stay in robustness mode:

- camera tuning
- smoothing filters
- threshold calibration
- OpenVINO fallback spike
- real-sample fixtures
- better false-positive analysis

Either path should be chosen from recordings and tests, not from vibes.

