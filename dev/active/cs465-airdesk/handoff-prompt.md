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
10. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tracking-samples.md`
11. `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/tasks.md`

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
- template/DTW recognizers second
- personal ML models later after collecting real data
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

Current first sprint direction:

- implement project skeleton and tooling
- define core typed data structures and event schema
- define profile/config schema
- add dry-run action target and Hyprland action wrapper
- add capture/tracking interfaces
- add JSONL recording/replay format
- add mock/replay backend
- add first synthetic gesture primitive tests
- avoid polished UI, cursor takeover, Kinect, and ML training until the foundation is stable

Current second sprint direction:

- test Python/OpenCV/MediaPipe dependency viability
- constrain Python to 3.12 if live tracking packages do not support 3.14
- add OpenCV camera capture/probe backend
- implement MediaPipe as backend zero if packaging and runtime behavior are viable
- add bounded `track` and `record` CLI commands
- record normalized landmark/event JSONL by default, not raw video
- replay recorded tracking streams through static recognizers
- document real-camera tracking quality before deciding whether Sprint 2 should build command-mode policy or tracking robustness

Sprint 2 outcome and carryover:

- improve camera probing/control with requested width, height, FPS, and FOURCC
- carry over deliberate hand-in-frame samples for open palm, fist, pinch, no-hand, and normal desk motion
- analyze replayed recordings for FPS, hand presence, primitive counts, candidate runs, and simple landmark jitter
- implement command-mode state policy in dry-run only
- resolve profile bindings with confidence thresholds and cooldowns
- add a safe `run` path over replay/live backends that routes gestures to `DryRunActionTarget`
- do not execute real Hyprland commands from live gestures until reliability data supports it
- current camera finding: `/dev/video0` needs OpenCV index normalization plus `--fourcc MJPG` to honor `640x480 @ 30 FPS`
- MediaPipe Tasks tuning is now exposed through `--model-path`, `--max-num-hands`, `--min-detection-confidence`, `--min-presence-confidence`, and `--min-tracking-confidence`; use `airdesk benchmark` to compare configurations instead of guessing
- current CLI default is one tracked hand for latency; test `--max-num-hands 2` only when the interaction needs two hands
- next best task: benchmark/tune the mirrored live view with deliberate hand motion, then record the recommended open palm, fist, pinch, no-hand, and normal desk motion samples

Current Sprint 3 direction:

- make live command mode observable, logged, and pilot-safe
- start by recording/analyzing the recommended deliberate samples and documenting observed FPS, false positives, false negatives, and jitter
- add runtime `--events-out` JSONL logs with session start/end metadata and gesture/mode/action events
- dynamic gesture research conclusion: do not bet Sprint 3 on a standalone LSTM; use intent-gated gesture phrases with rule/DTW baselines first, then compare LSTM/TCN once phase-labeled continuous data exists
- add a stateful phrase recognizer foundation for temporal gestures
- implement or explicitly defer flick/swipe-left/right and point-left/right based on replayable sample behavior
- show command-mode state in live `run --show` preview
- add pause/kill-switch behavior before any real action execution
- keep dry-run as the default and expose real Hyprland dispatch only through guarded explicit opt-in
- keep Sprint 3 focused on command gestures; cursor mode remains a later separate sprint

---
