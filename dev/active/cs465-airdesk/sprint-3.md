# AirDesk Sprint 3 Plan

## Purpose

Sprint 3 should move AirDesk from "live tracking and dry-run routing work" to "a pilot-safe command system can be tested on the real desktop."

Sprint 2 created the core live loop:

- camera mode probing and 30 FPS MJPG capture
- MediaPipe Tasks backend with model/threshold/hand-count tuning
- mirrored live preview with landmarks and gesture indicators
- live primitive tuning
- replay analysis
- command-mode policy
- profile binding resolution
- dry-run runtime routing

Sprint 2 did not yet prove:

- landmark stability across deliberate hand-in-frame samples,
- whether the current static thresholds are robust under normal desk motion,
- whether dynamic gestures such as swipes are usable,
- whether live command-mode feedback is clear enough,
- or whether real Hyprland dispatch should be enabled beyond a guarded local pilot.

Sprint 3 should therefore close the measurement loop, add the first temporal gesture recognizer, improve live status feedback, and introduce opt-in real Hyprland execution only after dry-run behavior is observable and logged.

## Sprint Theme

> Make live command mode observable, logged, and pilot-safe.

## Product / Research Stance

AirDesk should still be framed as a secondary input layer for command-like desktop actions.

- Do not claim gestures replace keyboard/mouse.
- Do not make MediaPipe the project identity.
- Keep dry-run as the default behavior.
- Require explicit opt-in before real desktop actions.
- Treat recordings and logs as research data and debugging infrastructure.
- Prefer a small reliable gesture vocabulary over a flashy broad one.
- Build the temporal recognizer as a backend-independent AirDesk layer over normalized landmarks.

## Non-Goals

Do not attempt these in Sprint 3:

- daemon/service installation
- polished control center
- always-on global cursor control
- virtual keyboard
- Kinect integration
- ML training
- two-hand gesture vocabulary
- destructive Hyprland/window actions
- formal user study with outside participants

## Key Decisions

### Sprint 3 Path

Sprint 3 should choose a guarded hybrid of Sprint 2's two gate paths.

Tracking appears promising enough to continue toward command interaction, but reliability still needs measured samples. The sprint should start in robustness mode, then enable a small pilot-safe execution path if the recordings and dry-run logs support it.

### Gesture Vocabulary Strategy

Keep Sprint 3's usable vocabulary small:

- open palm hold: enter listening / command mode
- fist: cancel
- pinch: confirm or safe note action
- swipe left/right: workspace navigation candidate
- point left/right: focus movement candidate if static pointing is stable enough

The implementation should support stateful recognizers because dynamic gestures depend on motion over time. Avoid jumping to ML. Start with interpretable rules and replayable synthetic tests.

### Runtime Logging Strategy

Dry-run and live sessions should be inspectable after the fact.

Add a runtime event log output path, likely:

```text
airdesk run --backend mediapipe --device /dev/video0 --dry-run --events-out data/logs/live-dry-run.jsonl
```

The log should include:

- session start/end metadata
- camera and MediaPipe settings
- gesture candidates
- mode transitions
- gesture confirmations
- action requests
- action results
- safety blocks and errors

Do not record raw video by default.

### Feedback Strategy

Live command mode needs visible state. The preview already shows landmarks and gesture labels; Sprint 3 should add runtime-oriented feedback:

- idle
- activation hold progress
- listening
- confirmed gesture
- action requested
- cooldown
- paused
- blocked/error

This can stay in the OpenCV debug window for now. A polished overlay/control center belongs later.

### Execution Safety Strategy

Real Hyprland execution must remain opt-in.

Proposed rules:

- `airdesk run` remains dry-run by default.
- Real actions require an explicit flag such as `--execute`.
- `--execute` refuses to run when the profile's `dry_run_default` is true unless a second explicit override is provided or the profile is changed.
- Only `hyprland.dispatch` actions from a small allowlist are executable in Sprint 3.
- Destructive actions remain blocked.
- `q`/`esc` in the preview window exits immediately.
- A pause/kill-switch command exists before real execution is considered done.

## Target CLI Shape

By the end of Sprint 3, these commands should exist or be expanded:

```text
airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label open-palm-hold --out data/recordings/open-palm-hold.jsonl
airdesk analyze data/recordings/open-palm-hold.jsonl
airdesk run --backend replay --recording data/recordings/open-palm-hold.jsonl --profile configs/profiles/study-safe.toml --dry-run --events-out data/logs/replay-dry-run.jsonl
airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --dry-run --show --events-out data/logs/live-window-manager-dry-run.jsonl
airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --execute --show
```

The final `--execute` command is a gate item. It should only be used after dry-run logs show acceptable false activation behavior.

## Deliverables

### 1. Live Sample Reliability Pass

Acceptance criteria:

- Record the recommended samples:
  - `open-palm-hold`
  - `fist-hold`
  - `pinch-hold`
  - `no-hand`
  - `normal-desk-motion`
- Analyze each sample with `airdesk analyze`.
- Update `tracking-samples.md` with observed FPS, hand-present frames, false positives, false negatives, and visible jitter notes.
- Decide whether `640x480 @ 30 FPS MJPG` remains the default live command setting.
- Decide whether one-hand tracking remains the default for command mode.

### 2. Runtime Event Logging

Acceptance criteria:

- Add `--events-out` to `airdesk run`.
- Runtime writes JSONL event logs using the existing recording/event schema where possible.
- Session start event includes backend, profile, dry-run/execute mode, camera settings, and MediaPipe tuning settings.
- Session finish event includes frame count, event count, action count, interruption status, and duration.
- Tests cover log creation over replay backend.

### 3. Stateful Gesture Recognizer Foundation

Acceptance criteria:

- Add a recognizer abstraction that can combine stateless pose recognizers and stateful temporal recognizers.
- Add swipe-left and swipe-right recognition over a short landmark history.
- Keep gesture output as `GestureCandidate` so existing policy/profile/runtime code continues to work.
- Add synthetic tests for left/right swipes.
- Add replay analysis counts for new dynamic candidates.
- Do not couple the recognizer to MediaPipe internals.

### 4. Pointing Gesture Spike

Acceptance criteria:

- Add a first rule-based `point_left` and `point_right` candidate, or document why it is too unstable from current landmarks.
- Test pointing rules against synthetic landmarks.
- Keep point gestures lower priority than swipe if false positives are high.
- Update `window-manager.toml` only if the gestures are reliable enough for dry-run.

### 5. Live Command Feedback

Acceptance criteria:

- `airdesk run --show` displays command-mode state in the preview.
- The preview shows at least idle, listening, confirmed gesture, action requested/result, paused, and blocked/error state.
- Gesture indicators remain visible for setup and tuning.
- Tests cover the state model or formatter without requiring a GUI.

### 6. Pause / Kill Switch

Acceptance criteria:

- Add a runtime pause state or immediate kill switch before real execution is enabled.
- `q`/`esc` exits preview-backed live runs.
- A keyboard-controlled pause/resume path is available when `--show` is active, or a CLI-safe alternative is documented.
- Paused runtime should continue previewing if practical but should not execute actions.
- Tests cover paused runtime suppressing action execution.

### 7. Opt-In Hyprland Execution

Acceptance criteria:

- Add an explicit real execution path, likely `airdesk run --execute`.
- Dry-run remains default.
- Real execution refuses unsafe profile/settings combinations.
- Only a small safe Hyprland dispatcher allowlist is enabled:
  - `workspace`
  - `movefocus`
  - possibly `fullscreen` only after manual verification
- Destructive actions remain blocked.
- Tests cover dry-run default, refusal cases, and injected Hyprland action target behavior.
- Live real execution is only considered done after successful replay and live dry-run logs.

### 8. Pilot Protocol

Acceptance criteria:

- Add `studies/pilot-0.md` or an equivalent active planning doc.
- Define a small Caden-only pilot:
  - baseline keyboard/mouse actions
  - AirDesk dry-run actions
  - optional AirDesk execute actions
  - task success/failure notes
  - false activations
  - fatigue/discomfort notes
- Keep it informal but structured enough to guide the later CS465 study.

## Recommended Implementation Order

1. Record and analyze the five deliberate samples.
2. Update tracking notes with actual sample outcomes.
3. Add runtime event logging over replay backend.
4. Add stateful recognizer foundation and synthetic swipe tests.
5. Add swipe recognition to replay analysis and runtime.
6. Add command-mode status feedback for `run --show`.
7. Add pause/kill switch.
8. Add guarded `--execute` path using Hyprland action target.
9. Verify safe Hyprland dispatchers manually.
10. Add pilot protocol and update README/tasks/handoff.
11. Run `ruff`, `pytest`, live dry-run smoke, and one deliberate replay test.

## Risks and Mitigations

### Dynamic Gestures Are Too Noisy

Risk:

- Swipe recognition may fire during normal hand movement.

Mitigation:

- Require listening mode first.
- Require minimum travel, duration bounds, and direction dominance.
- Analyze `normal-desk-motion` before enabling real execution.
- Keep real execution off by default.

### Live Feedback Is Misleading

Risk:

- The user may think a command is armed or canceled when the runtime state says otherwise.

Mitigation:

- Make state text explicit in the preview.
- Log every mode transition and action request.
- Prefer boring clear feedback over polished visuals.

### Real Execution Arrives Too Early

Risk:

- Workspace/focus changes may trigger accidentally.

Mitigation:

- Two-stage opt-in for real execution.
- Small allowlist.
- Pause/kill switch first.
- Dry-run logs must look acceptable before using `--execute`.

### Scope Creep Into Cursor Mode

Risk:

- Pinch and hand-position data make cursor control tempting.

Mitigation:

- Keep Sprint 3 focused on command gestures.
- Cursor mode gets a separate sprint with a controlled test surface before global pointer control.

## Definition of Done

Sprint 3 is done when:

- deliberate sample recordings have been analyzed and documented,
- runtime sessions can write replayable event logs,
- swipe-left/right are implemented or explicitly deferred based on sample data,
- command-mode state is visible during live runs,
- pause/kill-switch behavior exists and is tested,
- real Hyprland execution is available only through explicit opt-in and safety checks,
- a small Caden-only pilot protocol is documented,
- `uv run ruff check .` passes,
- `uv run pytest` passes,
- README/tasks/handoff are updated with the Sprint 3 outcome.

## Sprint 4 Gate

At the end of Sprint 3, choose:

### Path A: Pilot-Safe Command Mode Works

Proceed to a focused study/tooling sprint:

- trial logging
- study task scripts
- CSV export
- baseline timing helpers
- pilot results summary
- small study protocol

### Path B: Gesture Reliability Is Still Weak

Stay in robustness mode:

- better smoothing
- per-user calibration
- stronger temporal gesture filters
- OpenVINO hand tracker spike
- alternative camera settings
- gesture vocabulary reduction

### Path C: Command Mode Works, Cursor Is the Next Product Bet

Start a separate cursor-mode sprint:

- fake cursor overlay
- pinch-to-enter pointer control
- smoothing/gain/dead zone
- controlled click/drag test surface
- no global pointer takeover until the test surface is stable
