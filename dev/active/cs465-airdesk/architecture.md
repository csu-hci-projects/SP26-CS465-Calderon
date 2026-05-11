# AirDesk Architecture

## Vision

AirDesk should be designed as an OS-level spatial input layer, not as a one-off MediaPipe demo.

Long-term product shape:

> A configurable desktop control layer where hands, keyboard, mouse, camera, depth sensors, and desktop state blend into one modeful input system.

Near-term research shape:

> A reliable, instrumented prototype for evaluating whether mid-air gestures can support useful desktop commands under situational impairment.

The architecture should let the product vision grow without forcing the research paper to overclaim.

## Design Principles

- Keep hand tracking replaceable.
- Keep gesture recognition separate from sensor capture.
- Keep desktop actions separate from gesture recognition.
- Treat modes and profiles as first-class concepts.
- Record enough data to debug, replay, tune, and eventually train models.
- Favor explicit activation, clutching, and visible system state over always-on magic.
- Put safety boundaries around destructive or high-risk actions.
- Build the prototype as something Caden may personally use after the class.

## System Shape

```text
camera / sensor
  -> capture backend
  -> tracking backend
  -> normalized hand/body state
  -> smoothing and calibration
  -> primitive pose / motion recognizers
  -> stable event and combo grammar
  -> mode/profile safety state machine
  -> action router
  -> Hyprland / media / cursor / overlay / logs
```

For the crunch-time prototype, the live-control lane should be deterministic:
MediaPipe landmarks become primitive pose and movement facts, those facts become
stable per-hand events, and a small grammar decides whether an OS action is
allowed. Learned recognizers stay in preview/replay/evaluation lanes until
their false activation rate is low enough for guarded actions.

## Runtime Surfaces

### Daemon

The daemon owns the real-time pipeline:

- device capture
- hand/body tracking
- gesture recognition
- mode transitions
- profile selection
- action dispatch
- logging
- recording/replay

It should be able to run without a heavy UI.

### Overlay

The overlay is the small, always-available surface:

- active/inactive state
- current profile and mode
- listening/clutch status
- current gesture candidate
- confidence/debug indicators
- quick pause/resume
- fake cursor
- virtual keyboard or radial menus when active

The overlay should be visible when it helps and quiet when it does not.

### Control Center

The control center is the full configuration and tuning app:

- camera/sensor setup
- profile configuration
- gesture-to-action bindings
- threshold tuning
- calibration
- personal gesture recording
- replay/debug tools
- model training controls
- study-session management
- log review and export

This can be built after the daemon/overlay core exists, but the data model should anticipate it.

## Package Boundaries

Proposed Python package layout:

```text
airdesk/
  capture/          webcam, video file, recorded stream, Kinect stream
  tracking/         MediaPipe, OpenVINO, Kinect/depth, mock/replay, future Ultraleap
  state/            normalized hand/body state, calibration, smoothing
  poses/            direct landmark pose features shared by control and diagnostics
  control/          deterministic live-control grammar, debouncing, combos
  gestures/         legacy/dynamic/learned recognizers, templates, classifiers
  modes/            command, cursor, media, window manager, presentation, keyboard
  profiles/         bindings, thresholds, safety rules, profile selection
  actions/          Hyprland, media, input, shell, dry-run
  overlay/          status UI, fake cursor, radial/menu surfaces, virtual keyboard
  logging/          JSONL/CSV logs, recordings, replay
  study/            tasks, trials, participant/session metadata, exports
  config/           schema, defaults, profile loading
```

Sprint 0 implementation status:

- `src/airdesk/state/` owns typed frame, landmark, tracking, gesture, action, and event records.
- `src/airdesk/profiles/` loads TOML profiles and validates required profile/binding fields.
- `src/airdesk/actions/` contains dry-run and Hyprland action targets.
- `src/airdesk/capture/` and `src/airdesk/tracking/` define backend interfaces.
- `src/airdesk/recording/` stores replayable JSONL tracking/event records.
- `src/airdesk/gestures/` contains the first static rule recognizers for open palm, fist, and pinch.
- `configs/profiles/` contains the initial `study-safe` and `window-manager` profiles.

Near-term logic-control additions should preserve these boundaries:

- `poses` should own primitive landmark facts such as pinch distance, stable
  open palm, fist, sideways open palm, finger count, and palm zone. If adding a
  new top-level package feels too large for the first patch, add the code under
  `control/poses.py` and leave a future extraction note.
- `control` should own deterministic live behavior: pose debouncing, hold
  windows, combo buffers, action grammar, cooldown, and whether an event is
  currently armed.
- `gestures` should be treated as the learned/dynamic diagnostic lane:
  `dtw.py`, `motion.py`, `learned_filter.py`, `decoder.py`, and TCN-facing
  helpers should not become dependencies of the new live-control MVP.
- `actions` should own the OS adapters: Hyprland dispatch, cursor movement, and
  future `uinput` pointer-button/scroll injection.
- `overlay` / live preview should explain what is being seen, what combo is
  pending, which window is targeted, and what was executed or suppressed.

## Pivot Cleanup and Separation

The pivot should be implemented as a side-by-side architecture, not by deleting
or half-rewriting the existing gesture stack under time pressure.

Keep:

- `airdesk gesture ...` commands for replay/evaluation/model diagnostics.
- `airdesk run` as the older profile/command-mode runtime until the new control
  runtime proves itself.
- `airdesk cursor run` as the old pinch-held cursor experiment, but document it
  as legacy once the new control runtime owns pointer behavior.
- Existing recordings, labels, features, IPN imports, and TCN checkpoints as
  evidence and future work.

Add:

- `src/airdesk/control/` for the deterministic control runtime:
  `poses.py` or `features.py`, `debounce.py`, `combos.py`, `grammar.py`,
  `runtime.py`, and possibly `status.py`.
- `airdesk control run` as the new class-demo surface, dry-run by default. This
  keeps the new loop easy to test without disturbing older `airdesk run` and
  `airdesk cursor run` behavior.
- Focused tests such as `tests/test_control_poses.py`,
  `tests/test_control_debounce.py`, `tests/test_control_combos.py`,
  `tests/test_control_grammar.py`, and `tests/test_input_actions.py`.

Do not do in the first implementation slice:

- Do not move large learned/DTW/TCN files just to make the tree look cleaner.
- Do not rename public commands before the demo.
- Do not route learned recognizer outputs into the control grammar.
- Do not let `control` import from `gestures.dtw`, `gestures.motion`,
  `gestures.learned_filter`, or TCN modules.

Allowed compatibility cleanup:

- It is fine to leave `StaticHandPoseRecognizer` in `gestures.primitives` for
  old tests while extracting shared landmark math into `poses` or
  `control/poses`.
- It is fine for old modules to import the new shared pose helpers later, but
  keep that as a small compatibility step after the live-control MVP is green.
- It is fine to add deprecation/legacy notes to docs and help text, but avoid
  breaking CLI behavior during the class crunch.

Potential top-level project layout:

```text
src/airdesk/
tests/
configs/
studies/
scripts/
dev/active/cs465-airdesk/
```

## Core Data Model

Use typed data structures for pipeline boundaries. Early versions can be dataclasses; if configuration and validation become complex, use Pydantic for config models.

Important events:

- `FrameCaptured`
- `HandFrame`
- `BodyFrame`
- `TrackingLost`
- `PoseCandidate`
- `GestureCandidate`
- `GestureConfirmed`
- `ModeChanged`
- `ProfileChanged`
- `ActionRequested`
- `ActionExecuted`
- `ActionFailed`
- `StudyEvent`

Important state objects:

- `HandLandmarks`
- `NormalizedHand`
- `BodyState`
- `GestureWindow`
- `RecognitionResult`
- `ModeState`
- `Profile`
- `ActionBinding`
- `CommandLogEntry`

## Tracking Backends

Tracking backends should expose a common interface. The rest of AirDesk should not care whether landmarks came from MediaPipe, OpenVINO, Kinect, recorded logs, or future hardware.

Candidate interface:

```text
HandTrackerBackend.start()
HandTrackerBackend.stop()
HandTrackerBackend.frames() -> stream[TrackingFrame]
```

`TrackingFrame` should include:

- timestamp
- source id
- frame dimensions
- zero or more hands
- handedness if available
- confidence if available
- optional raw landmarks
- optional world/depth coordinates
- optional debug image reference

### Webcam / MediaPipe

MediaPipe remains the fastest first backend because it provides 21 hand landmarks, handedness, and live-stream tracking options. It should be treated as a backend, not as the identity of the project.

### OpenVINO

OpenVINO may be useful if MediaPipe is unstable on the ThinkPad/Linux setup or if Intel/NVIDIA acceleration experiments produce smoother tracking.

### Kinect v2

Kinect v2 through `libfreenect2` should be considered a depth/body/presence backend first. It can provide depth, RGB, IR, and registration. It is likely more valuable for body distance, arm movement, segmentation, and presence than for precise finger tracking.

### Recorded Replay

Recorded replay is a core backend, not a convenience.

It should allow:

- replaying frame streams
- replaying landmark streams
- testing recognizer changes without repeating gestures
- comparing smoothing/threshold/model variants
- building personal training datasets

## Gesture Recognition Strategy

Do not start with an LSTM. Start with interpretable recognizers and logs, then train when the system has real data.

### Phase 1: Rule-Based Recognition

Use normalized landmarks and temporal windows for:

- open palm
- fist
- pinch
- point direction
- swipe left/right/up/down
- palm push
- hold
- dwell
- cancel

Rules should emit confidence, not just booleans.

### Phase 2: Template / Fallback Recognition

Use template matching or dynamic time warping for custom motion gestures:

- user-defined swipes/flicks
- air-drawn symbols
- repeated command gestures
- personalized motion profiles

This may need less data than neural models and gives better debuggability. For the current roadmap, DTW/template recognition is fallback and calibration support rather than the primary learned-recognition bet.

### Phase 3: Personal ML

Once AirDesk can record and label real sessions:

- train static gesture classifiers over normalized landmarks
- train one primary temporal classifier first: a small causal TCN over normalized AirDesk features
- defer GRU/LSTM comparisons unless the TCN path disappoints
- support per-user calibration and models

### Phase 4: General ML

Later, evaluate public datasets or pretrained models for broader gesture recognition. The goal is to reduce per-user setup without losing reliability.

### Recognition V2 Architecture

Current evidence says the recognition package needs a cleaner continuous-spotting boundary before more model work.

The detailed plan is in `recognition-v2-plan.md`. The short version:

```text
TrackingFrame
  -> per-hand feature stream
  -> motion activity proposal
  -> scorer / model adapter
  -> event decoder
  -> command queue
  -> mode/profile/safety policy
```

Important boundaries:

- feature streams own normalization, masks, and per-hand history;
- recognizers score evidence but do not execute actions;
- event decoding turns noisy frame/window evidence into one-shot command events;
- command queue and mode/profile policy own ordering, chaining, and safety;
- live desktop actions remain opt-in and guarded outside recognition.

Future learned-recognition package cleanup:

```text
airdesk/features/
  stream.py
  normalized.py

airdesk/recognition/
  evidence.py
  motion.py
  decoder.py
  queue.py
  tcn_v2.py
```

This package shape is provisional and belongs to the future model lane. It is
not the next class-demo cleanup. The next implementation should add the
side-by-side deterministic `control` lane first, then revisit learned
recognition package moves only if they remove real maintenance friction.

May 2026 plan review update: keep this package shape as the likely future
direction, but do not start with a broad migration. Current code already has
`FeatureRowStream`, a replayable event decoder, hand-scoped DTW/TCN evaluation,
and separate command/action policy. The first deterministic motion-evidence
scorer now lives at the existing gesture boundary in `airdesk/gestures/motion.py`;
evaluate that boundary before deciding whether a top-level `recognition/`
package is justified.

May 2026 TCN v2 architecture update: the v2 learned scorer is now closer to the
continuous-spotting boundary above. `src/airdesk/ml/tcn_v2_train.py` owns a
schema-2 residual dilated causal TCN with normalization/dropout, weighted/focal
evidence loss, sparse-boundary weighting, calibration metadata, and batched
prediction. `src/airdesk/analysis/tcn_v2.py` maps `start` and `end` evidence
into decoder activation/release behavior, while desktop actions remain outside
the learned recognizer. The same analysis boundary now owns
`diagnose-tcn-v2-events` and early-match evaluation support for causal peaks
that fire just before a hand-labeled event start; this keeps replay evidence
honest without changing live action safety. `airdesk gesture watch-tcn-v2` is
the corresponding no-action live preview: it loads schema-2 evidence checkpoints,
runs the shared model per visible hand stream, defaults to a resizable dashboard
with webcam landmarks, evidence bars, decoded-gesture history, emit/peak delay,
timing summaries, and motion-feature diagnostics, decodes candidates through the
same event decoder without flushing open live events before release evidence, and
can write prediction/candidate JSONL for later review. `airdesk gesture
holdout-tcn-v2` now provides the matching source-level train/test check for v2
so strong same-source replay cannot be mistaken for generalization. The first
schema-2 holdout on `sprint4-swipes-001` scored `2/4` held-out swipes with `5`
false activations, so this remains preview/replay infrastructure rather than an
action recognizer. It still bypasses runtime policy and action targets.

## Modes

Modes describe interaction technique. Profiles describe context and bindings.

### Command Mode

Open palm clutch enters a short listening window. One confirmed gesture triggers one command, then the mode exits.

This is the safest first mode and the easiest to evaluate.

### Cursor Mode

Pinch or another explicit clutch takes over pointer control. Release exits. Start with a fake cursor overlay before moving the real cursor globally.

### Text / Virtual Keyboard Mode

Large on-screen keyboard, pinch/dwell selection, short text only at first. Treat as separate from the first research evaluation unless it becomes reliable enough to study.

### Media Mode

Safe commands like play/pause, next/previous, volume, rewind, mute.

### Presentation Mode

Slide navigation, pointer/highlight, zoom, fullscreen.

### Hybrid Keyboard + Hands Mode

Use keyboard/mouse activity as input context. Examples:

- keyboard modifier plus hand swipe
- hand gesture while mouse remains available
- keyboard activity suppresses accidental gesture execution
- hands provide spatial commands while keyboard handles text

This may be one of the most promising ways to beat normal desktop workflows for specific tasks.

## Profiles

Profiles package bindings, thresholds, safety rules, and UI behavior.

Initial profiles:

- Window Manager
- Media / Kitchen
- Presentation
- Cursor
- Virtual Keyboard
- Hybrid Keyboard + Hands
- Accessibility
- Study Safe
- Experimental

Each profile should define:

- allowed modes
- activation gesture
- gesture bindings
- cooldowns
- confidence thresholds
- destructive-action policy
- overlay behavior
- logging level

## Action System

Actions should be routed through typed adapters:

- `DryRunActionTarget`
- `HyprlandActionTarget`
- `MediaActionTarget`
- `InputActionTarget`
- `OverlayActionTarget`
- `ShellActionTarget`

Hyprland should be the first real desktop target through `hyprctl dispatch`.

Caden's current Hyprland setup supports the core class-demo actions directly:

- launcher: `hyprctl dispatch global caelestia:launcher`
- switch workspace: `hyprctl dispatch workspace -1` / `+1`
- move focused/window-under-cursor to workspace: `hyprctl dispatch movetoworkspace -1` / `+1`
- close active window: `hyprctl dispatch killactive`
- move cursor: `hyprctl dispatch movecursor <x> <y>`

Close-window and move-window commands should remain guarded. The grammar should
show the focused/target window title from `hyprctl activewindow -j` before a
destructive or disruptive action, and execution should remain dry-run-first.

Longer term, AirDesk may use Hyprland IPC sockets for events/state and lower-latency control. The wrapper must handle socket lifecycle carefully because compositor IPC misuse can affect desktop responsiveness.

Cursor control should be isolated behind an input driver because Wayland input injection has multiple possible paths:

- Hyprland cursor dispatchers
- Wayland virtual pointer protocols
- Linux `uinput` / `ydotool`
- fake cursor overlay for safe testing

As of the 2026-05-11 planning pass, `hyprctl` is installed, Caden has write
access to `/dev/uinput`, and no external pointer helper such as `ydotool`,
`dotool`, or `wtype` is installed. The next implementation should therefore add
an internal input target behind tests rather than depending on an unavailable
binary. The Python `evdev` package was not installed during planning; either
add it deliberately or implement the minimal `uinput` path with clear setup
docs.

## Logging and Study Instrumentation

Logging is not optional. It is core infrastructure.

Log:

- timestamp
- profile
- mode
- tracking backend
- gesture candidates
- gesture confirmations
- stable pose events and combo-buffer contents
- armed action and target window/workspace when available
- confidence
- action requested
- action executed/failed
- cooldown/cancel events
- task/trial id when in study mode
- false activation markers when available

Prefer JSONL for raw events and CSV exports for analysis.

Recording levels:

- metadata only
- landmarks only
- landmarks plus debug frames
- full video, only with explicit consent

Study data should be excluded from version control by default.

## Safety Model

Default to safe actions while recognition is immature.

Safety rules:

- destructive commands disabled by default
- dry-run mode available for all actions
- visible feedback before/after command execution
- cooldown after command execution
- universal cancel gesture
- pause/kill switch
- keyboard/mouse override
- no always-on cursor control

## Open Architecture Questions

- Should the daemon and overlay communicate over local WebSocket, Unix socket, DBus, or a simple local HTTP API?
- Should the overlay be Qt, GTK, webview, Tauri, Eww, or a Wayland layer-shell app?
- Should the first UI be a debug OpenCV window, a real desktop overlay, or both?
- Should profiles be YAML/TOML files, a SQLite database, or both?
- How much should AirDesk depend on Hyprland-specific concepts versus a generic action layer?
- How early should Kinect support be added?
- What is the right boundary between a research build and a daily-use build?
