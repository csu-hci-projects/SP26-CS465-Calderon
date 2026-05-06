# AirDesk Handoff Prompt

Use this prompt for a fresh agent:

---

You are working with Caden on **AirDesk**, a new CS465 HCI / 3DUI final research project. The project lives at `/home/caden/projects/AirDesk`.

AirDesk explores webcam-based mid-air hand gestures as a secondary control layer for a Hyprland Linux desktop. The motivation is **situationally impaired interaction**: moments when keyboard and mouse are inconvenient, unavailable, dirty, or physically costly, such as cooking, painting, repairing hardware, presenting away from a desk, wearing gloves, or managing temporary wrist strain.

The broader product ambition is now larger than a small gesture demo. AirDesk should become a pluggable, profile-driven OS spatial input layer where webcam, optional depth sensors, hand gestures, keyboard, mouse, and desktop context can blend into command, cursor, media, presentation, accessibility, virtual keyboard, and hybrid interaction modes. The research paper should stay focused on a clean evaluatable slice while the prototype can grow beyond that.

Important stance:

- Do not frame gestures as a full replacement for keyboard/mouse.
- Frame them as opportunistic secondary controls for coarse desktop actions.
- Cursor control should be modeful, e.g. pinch to take over pointer movement and release to stop. Never assume always-on cursor replacement.
- Text entry / virtual keyboard is a separate research problem unless explicitly scoped into the study.
- The HCI contribution is the interaction technique and evaluation, not the novelty of hand tracking itself.
- Avoid unsupported claims about permanent disability populations unless the study actually includes those participants.
- Use "accessibility-motivated" and "situationally impaired interaction" language. Consider arthritis, RSI, chronic hand pain, gloves, limited reach, dirty hands, and distance from keyboard during design, but do not claim validated benefit without testing those populations.
- Do not make MediaPipe the identity of the project. Treat it as one replaceable tracking backend.
- Keep recording/replay and logging as first-class architecture because they support debugging, study analysis, and future ML training.

Read these files first:

1. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/context.md`
2. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/plan.md`
3. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/architecture.md`
4. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/research-notes.md`
5. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/dynamic-gesture-research.md`
6. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-0.md`
7. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-1.md`
8. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-2.md`
9. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-3.md`
10. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-4.md`
11. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/sprint-5.md`
12. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tracking-samples.md`
13. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tasks.md`
14. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/context-reset-prompt.md`

Current preferred research question:

> Can a small, clutch-based mid-air gesture vocabulary provide usable secondary control for common desktop tasks under situational impairment, and how does it compare to keyboard/mouse interaction in speed, error rate, fatigue, and user preference?

Prototype idea:

- Python daemon
- webcam capture first, with optional depth/sensor backends later
- pluggable hand/body tracking backend
- MediaPipe Hand Landmarker as likely backend zero
- OpenVINO/Kinect/recorded replay as important alternatives
- normalized hand/body state model
- rule-based gesture recognizers first
- intent-gated phrase recognizers for dynamic commands
- template/DTW fallback for calibration and safety
- causal TCN as the first learned temporal recognizer after collecting phase-labeled real data
- clutch gesture: open palm held for ~300 ms enters listening mode
- modes: command, cursor, text/virtual-keyboard, media, window-manager, presentation, accessibility
- profiles: window manager, media/kitchen, presentation, cursor, virtual keyboard, hybrid keyboard+hands, accessibility, study safe, experimental
- cursor mode: pinch to take over cursor, hand movement moves pointer, release exits
- Hyprland integration through `hyprctl dispatch`
- daemon + minimal overlay first, larger control center later
- JSONL/CSV logging and replayable landmark/session recordings
- study logs exported as CSV or JSONL

Initial gestures:

- open palm: activate gesture/listening mode
- swipe left/right: switch workspace
- point left/right/up/down: move focus
- palm push: play/pause media
- fist: cancel
- pinch/flick: move window to adjacent workspace or enter cursor mode depending active profile
- two-hand gestures as stretch goals

Initial study:

- within-subjects comparison
- keyboard/mouse baseline vs AirDesk gestures
- optional situational-impairment condition: standing away from keyboard, wearing gloves, or holding an object
- tasks: pause media, switch workspace, focus neighboring window, toggle fullscreen, move active window to workspace, cancel/recover
- metrics: completion time, success rate, false activations, correction actions, fatigue, SUS, NASA-TLX, preference interview

Useful sources:

- MediaPipe Hands: https://arxiv.org/abs/2006.10214
- MediaPipe Hand Landmarker docs: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- OpenVINO hand tracker: https://github.com/geaxgx/openvino_hand_tracker
- OpenKinect libfreenect2: https://github.com/OpenKinect/libfreenect2
- Ultraleap hand tracking: https://docs.ultraleap.com/hand-tracking/index.html
- HaGRID dataset: https://github.com/hukenovs/hagrid
- HaGRIDv2: https://arxiv.org/abs/2412.01508
- Understanding Mid-Air Hand Gestures: https://www.microsoft.com/en-us/research/?p=163789
- Gesture Elicitation Studies for Mid-Air Interaction: https://www.mdpi.com/2414-4088/2/4/65
- SIID workshop: https://arxiv.org/abs/1904.05382
- Situational impairment research: https://arxiv.org/abs/1904.06128
- Ability-Based Design: https://cacm.acm.org/research/ability-based-design/
- Hyprland IPC: https://wiki.hypr.land/IPC/
- Hyprland Dispatchers: https://wiki.hypr.land/Configuring/Dispatchers/

Your job is to help Caden turn this into both:

1. a working prototype he may actually use on Hyprland
2. a credible CS465 HCI research paper/project

Plan first, keep scope tight, and make the research question shape the implementation.

Important nuance: keep the implementation ambitious enough to be personally useful, but keep the paper claims narrow and evidence-based.

Current hardware context:

- generic ThinkPad webcam, likely around 720p60
- Intel i7 CPU
- NVIDIA T550 laptop GPU
- Hyprland Linux desktop
- Kinect v2 available for experiments

Current architecture direction:

- package around replaceable backends and typed pipeline boundaries
- start with recording/replay, normalized hand state, rule-based gestures, Hyprland dry-run/dispatch, overlay feedback, and study logging
- design for future control center, Kinect/depth input, cursor control, virtual keyboard, hybrid keyboard+hand workflows, and personal ML training

Current implementation state:

- uv Python project with ruff/pytest
- typed frame, landmark, gesture, profile, action, and event data structures
- OpenCV camera backend and MediaPipe Tasks backend
- camera probing/mode reporting and benchmark/tuning commands
- JSONL recording/replay/analyze
- static open-palm/fist/pinch/point recognizers
- intent-gated swipe phrase recognizer foundation
- command-mode policy, profile binding resolver, dry-run runtime path
- runtime `--events-out` logging with session start/finish events
- preview-driven collection with countdown, keep, redo, skip, and quit
- guarded opt-in Hyprland execution for allowlisted commands
- modeful cursor command where pinch-hold moves the Hyprland cursor through `movecursor`
- continuous label schema and label CLI, including `label suggest`
- deterministic feature export
- rule and DTW evaluation paths
- dependency-free personalized DTW/template calibration baseline

Current hardware/evidence findings:

- `/dev/video0` works reliably at `640x480 @ 30 FPS MJPG`; requesting 60 FPS falls back to 30 FPS.
- Hyprland supports `hyprctl dispatch movecursor x y`.
- `ydotool`/`wtype` were not installed during the cursor spike, so click/drag injection is still pending.
- Raw data and generated labels/features/evaluations live under ignored `data/` and should not be committed unless Caden explicitly asks.

Current Sprint 3 direction:

- make live command mode observable, logged, and pilot-safe
- start by recording/analyzing the recommended deliberate samples and documenting observed FPS, false positives, false negatives, and jitter
- add runtime `--events-out` JSONL logs with session start/end metadata and gesture/mode/action events
- dynamic gesture research conclusion: AirDesk's primary learned-model bet is intent-gated gesture phrases plus a small causal TCN trained on phase-labeled continuous data; rule/DTW is scaffolding/fallback; LSTM/GRU is deferred unless TCN disappoints
- add a stateful phrase recognizer foundation for temporal gestures
- implement or explicitly defer flick/swipe-left/right and point-left/right based on replayable sample behavior
- show command-mode state in live `run --show` preview
- add pause/kill-switch behavior before any real action execution
- keep dry-run as the default and expose real Hyprland dispatch only through guarded explicit opt-in
- keep Sprint 3 focused on command gestures; cursor mode remains a later separate sprint

Current Sprint 4 direction:

- build the dataset, labeling, feature pipeline, and first causal TCN recognizer
- define labels for continuous sessions, including event labels and phase labels
- export deterministic landmark-derived features from replayable JSONL recordings
- use rule recognizer only as diagnostic scaffolding
- use DTW/template matching as a personalized fallback/calibration baseline
- train/evaluate a small causal TCN on continuous sessions
- do not spend Sprint 4 comparing many model families
- select whether TCN or the rule/DTW fallback is safe enough for Sprint 5

Current Sprint 4 dataset/evidence:

- Caden recorded `data/recordings/sprint4-swipes-001`: 8 left swipes, 8 right swipes, 8 normal desk-motion negatives.
- Rule recognizer failed this batch: 0/16 positive swipe matches and high static-pose false activations.
- DTW model `data/models/gestures/caden-dtw-sprint4-swipes-001.json` matched 16/16 on the same batch, missed 0, produced 18 candidates, 2 false activations, 0 repeated fires, about 0.44 s mean latency, and 0 candidates on negative recordings.
- This is promising but optimistic because calibration and evaluation used the same batch.
- `airdesk gesture holdout-dtw` now runs deterministic DTW train/test evaluation. On `sprint4-swipes-001`, training on takes 001-006 and testing on takes 007-008 matched 2/4 held-out swipes, missed both held-out left swipes, produced 0 false activations on held-out negatives, and had about 0.40 s mean latency on matched events.
- Holdout diagnostics now record the closest rejected DTW windows. The left-swipe misses are not safely fixed by loosening thresholds because the margin that recovers both left swipes introduces false activations. Treat this as a left-swipe/negative feature-separation issue.
- An optional calibrated horizontal-displacement gate is implemented for DTW. With `--negative-distance-margin 1.3 --min-palm-dx-fraction 0.65`, the same holdout matched 4/4 held-out swipes, missed 0, produced 4 candidates, 0 false activations, and about 0.36 s mean latency.
- Next best task: ask Caden for a 60-90 second chained continuous gesture session, then evaluate the gated DTW variant on that fresh recording. Do not treat the gated holdout as proof because its parameters were chosen after inspecting this batch.

Current Sprint 5 direction:

- create study tooling and paper-ready evidence
- define a Caden-only pilot protocol
- add study/trial JSONL logging and CSV/summary export
- integrate runtime logs with study sessions/tasks
- run keyboard/mouse baseline and AirDesk dry-run pilot conditions
- optionally run guarded execute mode only if Sprint 3/4 evidence supports it
- scaffold the CS465 paper with limitations and narrow evidence-based claims

---
