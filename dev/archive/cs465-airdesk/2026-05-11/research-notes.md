# AirDesk Research Notes

## Updated Direction

Date: 2026-04-30

The project should stay ambitious as a personal computing system while keeping the class research claims narrow and evidence-based.

Product ambition:

> Control the desktop through a blend of hand gestures, keyboard, mouse, webcam, depth sensors, and context-aware profiles.

Research ambition:

> Evaluate where mid-air gestures are actually useful for desktop control, especially under situational impairment, and identify where they fail compared with keyboard/mouse.

## Current Hardware Context

Caden's current likely setup:

- generic ThinkPad webcam, probably around 720p60
- Intel i7 CPU
- NVIDIA T550 laptop GPU
- Hyprland Linux desktop
- Kinect v2 available for experimentation

Implications:

- Webcam should remain the main baseline because it is universal.
- NVIDIA/i7 should be sufficient for local inference experiments.
- Kinect v2 should be treated as an optional depth/body/presence backend, not as the first precise finger tracker.
- Tracking backend abstraction is important because MediaPipe has been unreliable or less smooth on this machine.

## Hand Tracking Research

### MediaPipe

MediaPipe Hand Landmarker remains the fastest first implementation path:

- 21 hand landmarks
- handedness
- confidence values
- video/live-stream modes
- widely used examples and documentation

Risk:

- Linux webcam performance may be inconsistent.
- GPU acceleration and packaging may be more frustrating than expected.
- Tracking quality depends heavily on camera exposure, motion blur, lighting, and frame rate.

Use MediaPipe as one backend, not as the project's core identity.

Sprint 1 implementation note:

- `mediapipe==0.10.35` and `opencv-python==4.13.0.92` install and import under Python 3.14 in this repo's `uv` environment.
- The installed MediaPipe package exposes the Tasks API, not the older `mp.solutions.hands` namespace.
- AirDesk therefore uses `mediapipe.tasks.python.vision.HandLandmarker` with a local `.task` model in ignored `data/models/`.
- The model URL used is Google's public Hand Landmarker bundle:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- A two-frame smoke test on `/dev/video0` ran the pipeline and produced replayable JSONL, but detected zero hands because no hand was intentionally placed in frame during the run.

Sprint 2 implementation note:

- The Tasks API does not expose the older `model_complexity` flag for Hand Landmarker.
- AirDesk now exposes the knobs that Tasks supports: `--model-path`, `--max-num-hands`, `--min-detection-confidence`, `--min-presence-confidence`, and `--min-tracking-confidence`.
- CLI live commands default to one hand because lower latency matters more than two-hand recognition until the gesture vocabulary requires both hands.
- Use `airdesk benchmark` to compare model bundles, hand count, confidence thresholds, camera modes, hand-present frames, and average FPS before changing defaults.

May 2026 two-hand gesture note:

- Combo/chained swipe collection now requires both hands to be tracked when both are visible.
- The old one-hand default is still useful for latency-sensitive cursor/static primitive work, but it is the wrong assumption for alternating-hand combos.
- `--max-num-hands 2` must be paired with per-hand feature export and per-hand recognizer scoring. The current feature path uses the first visible hand, so changing the collection flag alone would still lose information downstream.
- The `sprint4-gpu-swipes-002-structured` combo takes were removed because they were collected under the one-hand assumption. The `sprint4-gpu-swipes-002-singles` takes remain local but should be considered legacy/single-hand-only until reviewed or recollected under two-hand background/rest conditions.

May 2026 T550/Arch implementation note:

- MediaPipe GPU support on Linux is EGL/OpenGL ES based, not CUDA-first for the Python Tasks Hand Landmarker path.
- On Caden's Hyprland hybrid-graphics setup, `prime-run glxinfo` can report the T550 while MediaPipe still initializes EGL on Intel/Mesa.
- A working local T550 path is now captured by `scripts/airdesk-nvidia-mediapipe-wayland`, which sets the NVIDIA GLVND EGL vendor file and `EGL_PLATFORM=wayland` before Python starts.
- Confirm the path from MediaPipe's startup log: it should include `OpenGL ES 3.2 NVIDIA` and `NVIDIA T550 Laptop GPU`.
- Short local smoke timings showed the T550 GPU delegate reducing MediaPipe inference to about 4 ms per frame, while capture remains camera-paced around 30 FPS. This is promising for fast swipes, but tracking quality still needs hand-in-frame benchmark runs under real movement and lighting.

References:

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- https://arxiv.org/abs/2006.10214

### OpenVINO

OpenVINO is worth exploring if MediaPipe performs poorly. It may be useful for model acceleration or alternative hand-pose pipelines.

Reference:

- https://github.com/geaxgx/openvino_hand_tracker

### Kinect v2 / libfreenect2

Kinect v2 through `libfreenect2` can provide:

- RGB
- IR
- depth
- RGB/depth registration
- OpenCL/CUDA/OpenGL processing paths

It does not, by itself, solve precise finger tracking. Its best AirDesk use may be:

- body presence
- distance from desk
- standing/away-from-keyboard context
- coarse arm gestures
- segmentation
- depth-aware gesture disambiguation

Reference:

- https://github.com/OpenKinect/libfreenect2

### Ultraleap / Leap Motion

Potential future hardware option for high-quality hand/finger tracking. This should remain optional because AirDesk's baseline identity should be commodity webcam control.

Reference:

- https://docs.ultraleap.com/hand-tracking/index.html

### Public Gesture Datasets

HaGRID and HaGRIDv2 may be useful for static gesture classifiers or later fine-tuning. HaGRIDv2 is notable because it includes many gesture images and "no gesture" examples, which matter for false-positive control.

References:

- https://github.com/hukenovs/hagrid
- https://arxiv.org/abs/2412.01508

## Gesture Recognition Research

See also:

- `dynamic-gesture-research.md` for the Sprint 3 research spike and the May 2026 update. The current direction is no longer "make the window TCN good enough"; it is continuous gesture spotting with position-invariant features, phase/event labels, event decoding, and a small hybrid recognizer that can grow toward graph/transformer memory later.

### Main Insight

The first hard problem is not just classification. It is gesture spotting:

- when a gesture starts
- when it ends
- whether it was intentional
- whether it should execute a command
- whether the system should ignore it due to mode/profile/safety state

### Recommended Progression

1. Rule-based recognizers for interpretable primitives.
2. Intent-gated phrase recognizers for dynamic commands.
3. Template/DTW fallback for personalized wrist flicks and conductor-like motions.
4. A small causal TCN trained toward stream/phase labels rather than only whole-window labels.
5. Event decoding over model probabilities: confidence rise/peak/fall, hysteresis, recovery, cooldown, and repeated-fire suppression.
6. Weak sequence labeling / CTC-style alignment once Caden can provide ordered streams but not exact timestamps.
7. LSTM/GRU only if the TCN/spotting path disappoints or a later comparison is worth the time.
8. ST-GCN / graph transformer models after the system has enough labeled skeleton data.

### Why Causal TCN, Not LSTM First

A learned temporal model may become useful, but using a classifier without an intent/spotting layer creates problems:

- requires labeled temporal data
- makes false activations harder to debug
- complicates the research prototype
- may classify clean clips while failing in continuous real-time use

The stronger path is to build recording, replay, labeling, phase-aware gesture phrases, and rule/DTW fallback first. Then train one primary learned model: a causal TCN over normalized AirDesk features. LSTM/GRU should stay deferred unless the TCN path fails.

### May 2026 Continuous-Spotting Update

Caden's live TCN test exposed the core failure mode: consecutive fast swipes require a reset, right/left performance is asymmetric, and fixed windows can capture only part of a gesture. This matches the gesture-spotting literature. The next research/implementation target should be a hybrid continuous recognizer:

```text
normalized landmark stream
  -> motion/activity proposal
  -> causal phase model
  -> optional DTW/template score
  -> event decoder
  -> dry-run action log
```

Representation should be position and distance tolerant. Absolute `palm_x`, `palm_y`, and `palm_z` are useful for diagnostics, but the learned gesture identity should rely on wrist/palm-centered landmarks, hand-scale-normalized displacement, velocities, acceleration, direction consistency, confidence, and tracking continuity.

Important research anchors now in `dynamic-gesture-research.md`:

- continuous hand gesture recognition needs segmentation, latency, and false-detection metrics;
- dynamic gestures have preparation, stroke/nucleus, and retraction/recovery phases;
- non-gesture/background handling should be explicit, not an afterthought;
- CTC-style training may help when Caden knows the gesture order but cannot provide exact timestamps;
- transformers are interesting mainly for cross-window memory and skeleton relations, not as a replacement for event decoding.

## Hyprland / Wayland Research

Hyprland is a good first desktop target because it exposes many useful actions through dispatchers:

- workspace switching
- focus movement
- fullscreen/floating
- move active window
- cursor movement
- monitor/workspace/window operations

References:

- https://wiki.hypr.land/Configuring/Dispatchers/
- https://wiki.hypr.land/IPC/

AirDesk should start with `hyprctl dispatch` through a typed wrapper and dry-run mode.

Longer term:

- use Hyprland IPC for state/events
- add profile changes based on active window/workspace
- isolate real pointer control behind an input driver

Wayland cursor/input paths to evaluate:

- Hyprland cursor dispatchers
- Wayland virtual pointer protocol
- Linux `uinput`
- `ydotool`
- fake cursor overlay before global pointer control

Reference:

- https://wayland.app/protocols/wlr-virtual-pointer-unstable-v1
- https://github.com/ReimuNotMoe/ydotool

## Research Opportunities

### Track 1: Command Gestures Under Situational Impairment

Most class-friendly study:

- keyboard/mouse baseline
- AirDesk command gestures
- optional simulated impairment condition

Tasks:

- switch workspace
- focus neighboring window
- toggle fullscreen
- play/pause media
- move active window to adjacent workspace
- cancel/recover

### Track 2: Hybrid Keyboard + Hand Interaction

Potentially the most interesting personal-computing research direction.

Question:

> Can mid-air gestures improve desktop workflows when combined with keyboard/mouse rather than replacing them?

Examples:

- keyboard modifier plus hand swipe
- hand gestures for window/workspace commands while keyboard handles text
- mouse movement temporarily suppresses gesture execution
- hand pose chooses command target, keyboard confirms

### Track 3: Mode and Clutch Design

Compare:

- always-on recognition
- open-palm clutch
- pinch-hold clutch
- keyboard-assisted clutch
- profile-specific clutch gestures

Measures:

- false activations
- perceived control
- speed
- fatigue
- preference

### Track 4: Sensor Comparison

Compare:

- webcam-only
- Kinect/depth-only for coarse interactions
- webcam plus Kinect context

This is likely a future track because it adds hardware and setup complexity.

### Track 5: Personalized Recognition

Compare:

- hand-authored rules
- template/DTW personal gestures
- trained personal classifier

Useful if AirDesk logs enough labeled data.

## Product Research Questions

- Which gestures are fast enough to feel better than a keyboard shortcut?
- Which gestures feel natural but perform poorly?
- How much visible feedback is enough?
- Does clutching feel reassuring or annoying?
- Does hybrid keyboard plus gesture control create genuinely faster workflows?
- Can a user safely operate the computer from 1-2 meters away?
- Which tasks should never be gesture-controlled?
- How should destructive commands be gated?
- How should AirDesk recover when tracking quality drops?

## Current Working Position

AirDesk should be designed as a pluggable, profile-driven spatial input daemon.

## Sprint 1 Live Tracking Notes

Date: 2026-04-30

Local smoke-test observations:

- `uv sync --dev --extra live` resolves OpenCV and MediaPipe.
- `uv run airdesk camera list` reports `/dev/video0` through `/dev/video3`.
- `uv run airdesk camera probe --device /dev/video0` opened the camera and read a frame at `1920x1080`, reporting `5.00` FPS.
- OpenCV emitted `ioctl(VIDIOC_QBUF): Bad file descriptor` on release, but the probe and recording still succeeded.
- `uv run airdesk track --backend mediapipe --device /dev/video0 --max-frames 2 --no-show` ran MediaPipe Hand Landmarker and printed frame summaries.
- `uv run airdesk record --backend mediapipe --device /dev/video0 --max-frames 2 --no-show --out data/recordings/sprint1-smoke.jsonl` produced a replayable file.
- `uv run airdesk replay data/recordings/sprint1-smoke.jsonl` reported `frames=2 events=2 hands=0 open_palm=0 fist=0 pinch=0`.

Interpretation:

- The live signal pipeline works mechanically.
- The observed 5 FPS at 1920x1080 is likely too slow for comfortable gesture interaction.
- Before moving to command-mode policy, run a deliberate hand-in-frame sample and investigate camera resolution/FPS controls.

## Sprint 2 Tracking Notes

Date: 2026-04-30

Local implementation observations:

- `airdesk camera modes --device /dev/video0` reports MJPG modes up to `1920x1080 @ 30 FPS` and YUYV modes where `1920x1080` is limited to `5 FPS`.
- Opening `/dev/video0` as a literal path through OpenCV did not honor requested `640x480 @ 30 FPS`.
- AirDesk now normalizes `/dev/videoN` to numeric index `N` before passing it to OpenCV.
- With `--width 640 --height 480 --fps 30 --fourcc MJPG`, `airdesk camera probe --device /dev/video0` reports `640x480 @ 30 FPS`.
- A short `sprint2-smoke` recording at `640x480/MJPG` produced replayable JSONL and analyzed successfully with no hand present.
- The next useful local artifact should be a deliberate hand-in-frame sample, not another empty-frame smoke test.

Initial implementation should prioritize:

1. recording/replay
2. normalized hand-state model
3. rule-based command gestures
4. Hyprland dry-run and real dispatch
5. overlay feedback
6. study logging

ML, Kinect, cursor control, and virtual keyboard are part of the serious roadmap, not discarded stretch fantasies.
