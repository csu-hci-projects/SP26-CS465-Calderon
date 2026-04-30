# AirDesk Tasks

Current next sprint: see `sprint-0.md`.

## Phase 0: Project Setup

- [ ] Initialize new repo in `/home/caden/projects/AirDesk` if desired
- [ ] Choose Python package manager (`uv` recommended if available)
- [ ] Add baseline `src/airdesk/` project structure
- [ ] Add README with project setup and run command
- [ ] Add license if this may become public
- [ ] Add `.gitignore` for Python, logs, local videos, and study data
- [ ] Add config directory for profiles and defaults
- [ ] Add tests directory and baseline test runner
- [ ] Decide initial UI technology for debug window / overlay / control center

## Phase 0.5: Architecture Foundation

- [ ] Define core typed data structures for frames, landmarks, gestures, modes, profiles, and actions
- [ ] Define `HandTrackerBackend` interface
- [ ] Define capture backend interface
- [ ] Define action target interface
- [ ] Define profile/config schema
- [ ] Define event/log schema
- [ ] Add dry-run action target
- [ ] Add recorded replay backend design
- [ ] Add sample config/profile files
- [ ] Add unit tests for config loading and event serialization

## Phase 1: Technical Spike

- [ ] Open webcam reliably
- [ ] Enumerate camera modes and identify stable resolution/FPS options
- [ ] Tune webcam exposure/focus/FPS settings where possible
- [ ] Run MediaPipe hand tracking locally
- [ ] Draw/debug 21 hand landmarks
- [ ] Measure landmark stability under normal desk lighting
- [ ] Record sample landmark streams
- [ ] Replay recorded landmark streams through a mock/replay backend
- [ ] Detect open palm
- [ ] Detect fist
- [ ] Detect swipe left/right
- [ ] Detect pointing direction
- [ ] Detect pinch start/hold/release
- [ ] Add gesture confidence and cooldown
- [ ] Compare at least one alternative hand tracking backend if MediaPipe is unstable

## Phase 1.5: Sensor Backend Experiments

- [ ] Investigate OpenVINO hand tracking viability
- [ ] Install/test Kinect v2 through `libfreenect2`
- [ ] Capture Kinect RGB/depth/IR stream
- [ ] Evaluate Kinect depth for distance-from-desk and body/presence context
- [ ] Decide whether Kinect enters the main architecture now or remains experimental
- [ ] Document hardware setup and reliability notes

## Phase 2: Hyprland Integration

- [ ] Verify `hyprctl dispatch workspace ...`
- [ ] Verify `hyprctl dispatch movefocus ...`
- [ ] Verify fullscreen/floating dispatchers
- [ ] Create `hyprland.py` wrapper
- [ ] Add dry-run mode so commands can be tested safely
- [ ] Add command log output
- [ ] Investigate Hyprland IPC events for active workspace/window context
- [ ] Keep Hyprland-specific behavior behind an action adapter

## Phase 3: Interaction Design

- [ ] Implement clutch mode: open palm held for ~300 ms
- [ ] Add visible feedback for "listening"
- [ ] Add gesture-to-command mappings
- [ ] Add cancel gesture
- [ ] Add cooldown after command execution
- [ ] Define mode model: command, cursor, text, media, window-manager, presentation, accessibility
- [ ] Define profile model: bindings, thresholds, safety rules, overlay behavior
- [ ] Avoid destructive commands until recognition is reliable
- [ ] Add pause/kill switch behavior
- [ ] Add keyboard/mouse override or suppression behavior
- [ ] Add Study Safe profile
- [ ] Add Hybrid Keyboard + Hands profile concept

## Phase 3.5: Cursor Mode

- [ ] Prototype fake cursor overlay controlled by hand position
- [ ] Add pinch-to-enter cursor mode
- [ ] Add release-to-exit cursor mode
- [ ] Add smoothing / gain / dead-zone settings
- [ ] Add visible cursor-mode indicator
- [ ] Investigate safe real-cursor control on Wayland/Hyprland
- [ ] Add controlled test surface for click/drag before global cursor control
- [ ] Evaluate Hyprland cursor dispatchers
- [ ] Evaluate Wayland virtual pointer or `uinput`/`ydotool` path

## Phase 3.6: Text Mode / Virtual Keyboard

- [ ] Design large-key virtual keyboard
- [ ] Add pinch-to-press on fake keyboard
- [ ] Consider dwell-select fallback
- [ ] Keep text mode separate from first research evaluation unless it becomes stable
- [ ] Consider command-palette style text alternatives before full keyboard complexity

## Phase 3.7: Control Center / Configuration

- [ ] Decide control center technology
- [ ] Create profile editor concept
- [ ] Create gesture binding editor concept
- [ ] Create threshold tuning UI concept
- [ ] Create calibration flow concept
- [ ] Create replay/log review concept
- [ ] Keep daemon usable without full control center

## Phase 4: Study Tooling

- [ ] Add trial/task logging
- [ ] Add participant/session ID support
- [ ] Add manual task start/end controls
- [ ] Export CSV or JSONL logs
- [ ] Create `studies/protocol.md`
- [ ] Create `studies/trial_tasks.md`
- [ ] Create `studies/survey.md`
- [ ] Add false activation annotation flow
- [ ] Add replayable study-session artifact format
- [ ] Decide whether raw video is collected or avoided

## Phase 5: Experiment

- [ ] Pilot test with Caden only
- [ ] Revise gestures and tasks after pilot
- [ ] Recruit small sample if allowed by course constraints
- [ ] Counterbalance task order
- [ ] Collect objective logs
- [ ] Collect SUS / NASA-TLX / fatigue ratings
- [ ] Conduct short interview

## Phase 6: Paper

- [ ] Write introduction and motivation
- [ ] Write related work section
- [ ] Describe AirDesk prototype
- [ ] Describe gesture vocabulary
- [ ] Describe experiment design
- [ ] Present pilot/user-study results or expected evaluation if no full study is required
- [ ] Discuss limitations
- [ ] Clearly distinguish accessibility motivation from validated accessibility claims
- [ ] Discuss cursor/text modes as product extensions or separate evaluation tracks if not studied
- [ ] Discuss future work
- [ ] Format references

## Phase 7: Learning System

- [ ] Add gesture labeling workflow
- [ ] Add template/DTW recognizer baseline for dynamic gestures
- [ ] Train simple static gesture classifier from recorded landmarks
- [ ] Evaluate temporal classifier options: GRU, LSTM, TCN
- [ ] Compare ML recognizers against rule/template baselines
- [ ] Add per-user model/profile concept
- [ ] Investigate public dataset usefulness, including HaGRID/HaGRIDv2

## Nice-to-Haves

- [ ] Small desktop overlay showing current gesture
- [ ] Audio feedback for recognized command
- [ ] Waybar module integration
- [ ] Configurable YAML gesture mappings
- [ ] Context profiles: kitchen/media, window manager, accessibility, presentation
- [ ] Mode for media-only controls
- [ ] Mode for presentation controls
- [ ] Modeful cursor takeover
- [ ] Virtual keyboard
- [ ] Optional two-hand resize/volume gesture
- [ ] Sensor fusion: webcam hand landmarks plus Kinect depth/body context
- [ ] Profile auto-switching from desktop context
- [ ] Gesture-controlled radial menu
- [ ] Personal gesture training wizard
