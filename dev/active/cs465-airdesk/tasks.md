# AirDesk Tasks

Current next sprint: shift recognition work toward continuous gesture spotting, then scope the Sprint 5 pilot around whichever recognizer has event-level replay evidence.

Current context-reset task: evaluate the new deterministic motion-event baseline on existing replay data before any live preview. Do not collect broad combo data or keep sweeping current TCN thresholds.

## Phase 0: Project Setup

- [ ] Initialize new repo in `/home/caden/projects/AirDesk` if desired
- [x] Choose Python package manager (`uv` recommended if available)
- [x] Add baseline `src/airdesk/` project structure
- [x] Add README with project setup and run command
- [ ] Add license if this may become public
- [x] Add `.gitignore` for Python, logs, local videos, and study data
- [x] Add config directory for profiles and defaults
- [x] Add tests directory and baseline test runner
- [ ] Decide initial UI technology for debug window / overlay / control center

## Phase 0.5: Architecture Foundation

- [x] Define core typed data structures for frames, landmarks, gestures, modes, profiles, and actions
- [x] Define `HandTrackerBackend` interface
- [x] Define capture backend interface
- [x] Define action target interface
- [x] Define profile/config schema
- [x] Define event/log schema
- [x] Add dry-run action target
- [x] Add recorded replay backend design
- [x] Add sample config/profile files
- [x] Add unit tests for config loading and event serialization

## Phase 1: Technical Spike

- [x] Open webcam reliably
- [x] Enumerate camera modes and identify stable resolution/FPS options
- [x] Tune webcam exposure/focus/FPS settings where possible
- [x] Expose MediaPipe model/threshold/hand-count tuning knobs
- [x] Add live backend benchmark command for FPS and hand-present frames
- [x] Add live MediaPipe timing breakdown for capture, inference, preview, and total loop cost
- [x] Add opt-in NVIDIA EGL launcher for T550 MediaPipe GPU testing on Hyprland/Arch
- [x] Run MediaPipe hand tracking locally
- [x] Draw/debug 21 hand landmarks
- [ ] Measure landmark stability under normal desk lighting
- [x] Record sample landmark streams
- [x] Replay recorded landmark streams through a mock/replay backend
- [x] Detect open palm
- [x] Detect fist
- [ ] Detect swipe left/right
- [ ] Detect pointing direction
- [x] Detect pinch start/hold/release
- [x] Add gesture confidence and cooldown
- [ ] Compare at least one alternative hand tracking backend if MediaPipe is unstable

## Sprint 3: Pilot-Safe Live Command Mode

- [ ] Record deliberate open palm, fist, pinch, no-hand, and normal desk motion samples
- [ ] Analyze deliberate samples and update tracking observations
- [x] Add runtime `--events-out` JSONL logging
- [x] Add session start/finish runtime events
- [x] Research dynamic gesture model options and document AirDesk strategy
- [x] Add intent-gated phrase recognizer foundation for temporal gestures
- [x] Detect flick/swipe left/right from normalized landmark motion
- [x] Add continuous positive and negative recording protocol for dynamic gestures
- [x] Add prompted collection workflow with countdown and keep/redo/skip
- [x] Decide whether point left/right is reliable enough for Sprint 3
- [x] Show command-mode state in live `run --show` preview
- [x] Add pause/kill-switch behavior before real execution
- [x] Add guarded opt-in Hyprland execution path
- [ ] Verify safe Hyprland dispatchers manually
- [x] Add Caden-only pilot protocol
- [x] Run `ruff`, `pytest`, replay smoke, and live dry-run smoke

## Sprint 4: Gesture Dataset, Labeling, and Causal TCN Recognition

- [x] Define continuous gesture label schema
- [x] Add `airdesk label init`
- [x] Add `airdesk label validate`
- [x] Add `airdesk label suggest` to bootstrap stroke timestamps from motion
- [x] Add deterministic feature extraction from tracking frames
- [x] Add feature export command
- [x] Add rule/DTW fallback support for personalized dynamic gestures
- [x] Add gesture evaluation metrics for continuous sessions
- [x] Add `airdesk gesture evaluate`
- [x] Add deterministic `airdesk gesture holdout-dtw` train/test evaluation
- [x] Run DTW holdout on `sprint4-swipes-001` and document mixed evidence
- [x] Add DTW holdout diagnostics for closest rejected windows
- [x] Diagnose left-swipe holdout misses as a feature/separation issue rather than a simple threshold issue
- [x] Add optional calibrated palm-displacement gate for DTW swipe separation
- [x] Evaluate gated DTW variant on `sprint4-swipes-001` holdout
- [x] Record and spot-check a fresh 60-90 second chained continuous recording
- [x] Add `airdesk gesture spot-dtw` for unlabeled continuous DTW candidate review
- [x] Add `airdesk gesture score-sequence` for remembered R/L order checks
- [x] Score structured chained-session DTW order against `R L R R L L R R L L`
- [x] Decide whether to collect one more timestamp-aware stream or start causal TCN prototype
- [x] Define the first causal TCN training target and dataset manifest format
- [x] Add deterministic dataset/window builder over exported AirDesk features
- [x] Add dependency-free `airdesk gesture build-tcn-dataset` manifest command
- [x] Add tests for TCN manifest/window construction and label assignment
- [x] Add optional PyTorch extra for offline ML training without changing base runtime
- [x] Add optional `airdesk gesture train-tcn` training scaffold and checkpoint metadata
- [x] Add `airdesk gesture evaluate-tcn` for TCN metrics comparable with rule/DTW
- [x] Add live/replay `airdesk gesture watch-tcn` classifier preview without desktop actions
- [x] Run first same-batch TCN training/evaluation pass over the local Sprint 4 manifest
- [x] Add deterministic `airdesk gesture holdout-tcn` evaluation split matching the DTW holdout split
- [x] Evaluate causal TCN against rule/DTW fallback on the same `sprint4-swipes-001` split
- [x] Add feature-diagnostics report for TCN/DTW left-swipe holdout misses
- [x] Run feature diagnostics on `sprint4-swipes-001` and document the evidence
- [x] Add causal trailing-window displacement/velocity/direction features to feature export
- [x] Rerun DTW/TCN holdouts with the new feature representation
- [x] Validate the best DTW window-feature variant on fresh or timestamp-labeled continuous streams
- [x] Explicitly defer LSTM/GRU unless the causal TCN/spotting path fails
- [x] Document the May 2026 research pivot from window classification to continuous gesture spotting
- [x] Add a position-invariant TCN/stream-model feature preset
- [x] Add stream/phase labels for stroke vs recovery/reset
- [x] Add probability/candidate event decoder with hysteresis, confidence peaks, cooldown, and repeated-fire suppression
- [x] Add ordered-sequence labeling helper for chained streams when exact timestamps are unavailable
- [x] Add chart-record collection workflow with timed prompts and chart-derived weak labels
- [x] Delete one-hand structured combo recordings from `sprint4-gpu-swipes-002-structured`
- [x] Document two-hand combo data pivot and pause combo collection
- [x] Make chart collection default/recommended path use `--max-num-hands 2` for combo sessions
- [x] Export per-hand feature rows with independent motion history instead of only `frame.hands[0]`
- [x] Update DTW/TCN evaluation to score per-hand streams and merge decoded events across hands
- [x] Add tests for two-hand feature rows, per-hand motion history, per-hand DTW candidates, and merged event ordering
- [x] Run replay checks on the two-hand feature/DTW/TCN/event-decoder path before collecting new combo data
- [x] Add motion-gated TCN target assignment so stationary visible hands are not trained as prompted strokes
- [x] Train/evaluate shared per-hand TCN on 003-to-004 two-hand chart split and document non-live-ready evidence
- [x] Add decoded TCN event diagnostics for misses, false activations, repeated fires, hand ids, and nearest-event timing
- [x] Add non-destructive motion-peak chart label refinement and document that first refined-label TCN checks were worse than prompt labels
- [x] Make live `watch-tcn` default to two tracked hands for the two-hand diagnostic path
- [x] Hide `recovery` by default in live TCN preview because it is an internal phase, not a gesture
- [x] Add `phase-stroke` TCN target mode to train without the recovery class after live recovery collapse
- [x] Stabilize live TCN HUD so both hand streams are shown together and static fist/pinch overlays are hidden
- [x] Train/evaluate more sensitive `phase-stroke` TCN with 0.20 motion gate and document higher false-activation tradeoff
- [x] Create Recognition V2 architecture plan after deep research report and live TCN misses
- [x] Review/refine Recognition V2 plan before implementation
- [x] Add a narrow deterministic motion-evidence module at the existing gesture boundary
- [x] Add replay-first `spot-motion` / `evaluate-motion` JSON surfaces before live preview
- [x] Implement deterministic per-hand motion-event baseline before TCN v2
- [x] Evaluate motion-event baseline on existing replay data before any live diagnostic preview
- [x] Inspect motion-baseline false activations and weak-left misses before live preview
- [x] Add label-aware motion diagnostic rows for accepted/rejected replay evidence
- [ ] Add the next replay-only negative-motion/intent rejection check after comparing diagnostics across a few labeled recordings
- [ ] Revisit broader `airdesk/recognition/` package split only after the baseline proves the boundary
- [ ] Decide whether to collect a tiny targeted live calibration slice
- [ ] Design TCN v2 only after baseline evidence, with boundary/intent heads rather than window argmax only
- [ ] Tighten active-hand weak labels and timestamp alignment using 003+004 before collecting more broad combo data
- [ ] Turn refinement diagnostics into a manual label-review pass instead of automatically adopting motion-peak labels
- [ ] Recheck decoder thresholds after label/timing diagnostics, not before
- [ ] Recollect targeted combo/chained swipe charts after the weak-label/decoder pass
- [x] Run first event-decoder replay smokes on isolated holdout and chained DTW candidates
- [ ] Document Sprint 5 recognizer decision after event-decoder replay evidence
- [x] Update dynamic gesture protocol and research notes
- [ ] Run `ruff`, `pytest`, and replay evaluation smoke

## Sprint 5: Study Tooling, Pilot, and Paper Evidence

- [ ] Add `studies/pilot-0.md`
- [ ] Define study/trial event schema
- [ ] Add study session/task logging CLI
- [ ] Add study CSV/summary export
- [ ] Integrate runtime logs with study session/task IDs
- [ ] Document keyboard/mouse baseline workflow
- [ ] Run Caden-only baseline pilot
- [ ] Run Caden-only AirDesk dry-run pilot
- [ ] Optionally run guarded execute-mode pilot if safe
- [ ] Summarize pilot results and design failures
- [ ] Add paper outline with results placeholders
- [ ] Update README/tasks/handoff
- [ ] Run `ruff`, `pytest`, replay smoke, and study CLI smoke

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
- [x] Create `hyprland.py` wrapper
- [x] Add dry-run mode so commands can be tested safely
- [x] Add command log output
- [ ] Investigate Hyprland IPC events for active workspace/window context
- [ ] Keep Hyprland-specific behavior behind an action adapter

## Phase 3: Interaction Design

- [x] Implement clutch mode: open palm held for ~300 ms
- [ ] Add visible feedback for "listening"
- [x] Add gesture-to-command mappings
- [x] Add cancel gesture
- [x] Add cooldown after command execution
- [x] Define mode model: command, cursor, text, media, window-manager, presentation, accessibility
- [x] Define profile model: bindings, thresholds, safety rules, overlay behavior
- [x] Avoid destructive commands until recognition is reliable
- [ ] Add pause/kill switch behavior
- [ ] Add keyboard/mouse override or suppression behavior
- [x] Add Study Safe profile
- [ ] Add Hybrid Keyboard + Hands profile concept

## Phase 3.5: Cursor Mode

- [ ] Prototype fake cursor overlay controlled by hand position
- [x] Add pinch-to-enter cursor mode
- [x] Add release-to-exit cursor mode
- [x] Add smoothing / gain / dead-zone settings
- [x] Add visible cursor-mode indicator
- [x] Investigate safe real-cursor control on Wayland/Hyprland
- [x] Add guarded real cursor movement through Hyprland `movecursor`
- [x] Add dry-run cursor movement and JSONL cursor session logging
- [ ] Add controlled test surface for click/drag before global cursor control
- [x] Evaluate Hyprland cursor dispatchers
- [ ] Add pointer-button injection for click and drag
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
- [x] Add template/DTW fallback for dynamic gestures
- [ ] Train simple static gesture classifier from recorded landmarks
- [ ] Train and harden the causal TCN recognizer for temporal gestures
- [x] Train/evaluate a continuous spotting variant of the TCN with position-invariant features and phase/event labels
- [ ] Improve continuous spotting TCN target quality with active-hand assignment and class-balance handling
- [ ] Evaluate alternate temporal classifiers only if the TCN path disappoints
- [ ] Investigate graph/transformer memory only after event decoding and dataset labels are in place
- [ ] Add per-user model/profile concept
- [ ] Investigate public dataset usefulness, including HaGRID/HaGRIDv2

## Nice-to-Haves

- [ ] Small desktop overlay showing current gesture
- [ ] Hyprland/overlay frame-boundary feedback when the hand is near leaving the camera region
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
