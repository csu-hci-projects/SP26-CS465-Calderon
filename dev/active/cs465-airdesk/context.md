# AirDesk Context

## Date

2026-05-01

## Project Summary

AirDesk is a proposed CS465 final research project for an HCI / 3D user interfaces course.

The system uses a webcam and hand tracking to recognize mid-air gestures, then maps those gestures to desktop actions in Hyprland. The goal is not to replace keyboard and mouse for all work. The goal is to evaluate whether gestures are useful as a **secondary interaction channel** when traditional input is inconvenient, unavailable, undesirable, or physically costly.

The prototype is now being framed as a broader desktop-control system with command gestures, modeful cursor control, virtual keyboard/text entry, media controls, presentation controls, accessibility profiles, hybrid keyboard-plus-hand interaction, and optional depth/body sensing. The research paper should stay more focused than the product: evaluate a clean slice of the interaction rather than claiming every possible feature is solved.

## Updated Product Vision

AirDesk should become an OS-level spatial input layer:

> A configurable desktop control system where hands, keyboard, mouse, webcam, depth sensors, and desktop context blend into one profile-driven interaction layer.

This is closer to a browser extension for the desktop than a standalone demo:

- a background daemon owns tracking, recognition, modes, bindings, and logs
- a small overlay shows state and quick controls
- a larger control center configures profiles, thresholds, bindings, calibration, replay, and training

AirDesk should be pluggable from the beginning. MediaPipe may be the first hand-tracking backend, but the architecture should support alternatives such as OpenVINO, Kinect/depth input, recorded replay streams, and future hardware like Ultraleap.

## Current Hardware Context

Caden's current target setup:

- generic ThinkPad webcam, likely around 720p60
- Intel i7 CPU
- NVIDIA T550 laptop GPU
- Hyprland Linux desktop
- Kinect v2 available for experimentation

Implications:

- webcam remains the universal baseline
- NVIDIA/i7 makes local inference and model experiments realistic
- Kinect v2 is promising for depth, body presence, distance from desk, and coarse arm context, but should not be assumed to solve precise finger tracking
- recorded replay should be a first-class backend so tracking and recognition can be tuned without repeating gestures manually

## Course Fit

CS465 focuses on input devices, interaction techniques, multimodal interaction, and 3D user interfaces. AirDesk fits because:

- the input modality is spatial, mid-air, hand-based interaction
- the system communicates with a computer through embodied gestures rather than mouse/keyboard only
- it can be evaluated using HCI methods: task time, error rate, false activations, workload, fatigue, usability, and preference
- it touches accessibility and situationally induced impairments without requiring the project to make unsupported claims about permanent disability populations

## Motivation

Keyboard and mouse are excellent for precision tasks, but many desktop actions are coarse and command-like:

- switch workspace
- pause or rewind media
- move focus between windows
- toggle fullscreen
- dismiss a notification
- move a window to another workspace
- launch an application

These tasks may be useful to trigger without touching the computer when the user is:

- cooking with dirty hands
- painting or working with materials
- repairing hardware or a car
- wearing gloves
- standing away from the desk
- presenting
- dealing with temporary wrist strain
- dealing with chronic or recurring hand pain, arthritis, RSI, or limited reach
- otherwise situationally limited

This is the HCI framing: **situationally induced impairments and disabilities** (SIIDs), also called situational impairments. The project can also cite **ability-based design**, which argues that systems should adapt to users' available abilities instead of assuming one ideal interaction channel.

AirDesk is also accessibility-motivated. Users with arthritis, repetitive strain, hand pain, or limited reach should be considered during design. However, unless the study actually recruits those users, the paper should not claim validated accessibility benefits for those populations. Safer phrasing:

> AirDesk explores an alternate input channel that may reduce reliance on repetitive keyboard/mouse actions and may be relevant to users with temporary, situational, or chronic limitations.

## Working Title Options

- AirDesk: Mid-Air Gestures for Situationally Impaired Desktop Interaction
- AirDesk: Webcam-Based Gesture Control for a Tiling Linux Desktop
- Evaluating Mid-Air Gestures as a Secondary Control Layer for Desktop Window Management
- Hands Off the Keyboard: Gesture-Based Desktop Control Under Situational Impairment

Current preferred title:

**AirDesk: Mid-Air Gestures for Situationally Impaired Desktop Interaction**

## Important Design Stance

Do not claim that hand gestures are universally better than keyboard/mouse.

The more defensible claim is:

> Mid-air gestures may be useful for command-like, spatial, low-text desktop actions when traditional input is inconvenient, unavailable, painful, or undesirable.

This avoids the trap of trying to make hand tracking beat a keyboard shortcut at keyboard-shortcut things.

## Product Direction

Think of AirDesk as a modeful input system, not one giant always-on gesture recognizer.

Potential modes:

- **Command Mode**: open palm clutch, then one discrete command gesture. Best for workspace, focus, media, fullscreen, and cancel actions.
- **Cursor Mode**: pinch to take control of the cursor, move the hand to move the pointer, release to stop. Best for coarse pointing and occasional click/drag.
- **Mouse + Keyboard Mode**: cursor mode plus an on-screen virtual keyboard for limited text entry.
- **Media Mode**: a small safe vocabulary for pause/play, rewind, volume, and next/previous.
- **Window Manager Mode**: workspace switching, focus movement, fullscreen/floating, send-to-workspace, and later resize/move.
- **Accessibility Profile**: lower movement gain, larger dwell targets, longer confirmation windows, reduced pinch force, fewer destructive mappings.
- **Presentation Mode**: next/previous slide, pointer/highlight, zoom, fullscreen.
- **Hybrid Keyboard + Hands Mode**: keyboard/mouse handle precision and text while hand gestures handle spatial commands, window movement, mode switches, and large desktop actions.

The early prototype should prioritize Command Mode because it is safest and easiest to evaluate. Cursor Mode should be implemented as modeful pinch takeover, not as an always-on replacement for the mouse.

The broader product should not artificially narrow scope. The class study can evaluate a focused slice while the implementation is designed to grow into full hands-first or hybrid desktop control.

## Key Research Terms

- HCI
- 3D user interfaces
- mid-air interaction
- gesture interaction
- situational impairments / SIIDs
- ability-based design
- desktop window management
- tiling window manager
- multimodal input
- clutching / mode activation
- modeful interaction
- cursor takeover
- virtual keyboard
- Midas Touch problem
- gesture vocabulary
- false activation
- fatigue / "gorilla arm"

## Related Work Seeds

Use these as starting points, not as a final bibliography:

- MediaPipe Hands: On-device Real-time Hand Tracking  
  https://arxiv.org/abs/2006.10214

- Understanding Mid-Air Hand Gestures: A Study of Human Preferences in Usage of Gesture Types for HCI  
  https://www.microsoft.com/en-us/research/?p=163789

- Gesture Elicitation Studies for Mid-Air Interaction: A Review  
  https://www.mdpi.com/2414-4088/2/4/65

- Proceedings of the CHI 2019 Workshop: Addressing the Challenges of Situationally-Induced Impairments and Disabilities in Mobile Interaction  
  https://arxiv.org/abs/1904.05382

- Situationally-Induced Impairments and Disabilities Research  
  https://arxiv.org/abs/1904.06128

- Ability-Based Design  
  https://cacm.acm.org/research/ability-based-design/

- Hyprland IPC  
  https://wiki.hypr.land/IPC/

- Hyprland Dispatchers  
  https://wiki.hypr.land/Configuring/Dispatchers/

## Current Decision

Proceed with the Hyprland hand-control project rather than forcing the existing portfolio / point-engine work into the class assignment.

Reason:

- the point engine ideas were mostly output/art-style driven
- AirDesk is directly about input, interaction technique, user performance, and situational usability
- the prototype may remain useful after the class

## Current Implementation State

Sprint 2 established a working live and replay foundation:

- OpenCV camera probing can request width, height, FPS, and FOURCC.
- `/dev/video0` works at `640x480 @ 30 FPS` with MJPG when normalized to OpenCV camera index `0`.
- MediaPipe Tasks Hand Landmarker is backend zero and remains replaceable behind the tracking interface.
- Live preview is mirrored by default and shows landmarks, skeleton, bounding boxes, handedness/confidence, hand count, and gesture indicators.
- MediaPipe model path, hand count, and confidence thresholds are CLI-tunable.
- JSONL recording/replay and replay analysis are available.
- Command-mode policy, profile binding resolution, and dry-run runtime routing are implemented.

Current next step:

> Sprint 3 should make live command mode observable, logged, and pilot-safe before real desktop actions are used beyond explicit local testing.

## Current Roadmap

### Sprint 3: Pilot-Safe Live Command Mode

Build the live command loop:

- runtime event logs,
- continuous positive/negative recordings,
- intent-gated phrase recognizer foundation,
- rule/DTW dynamic gesture baselines,
- live command-state feedback,
- pause/kill switch,
- guarded opt-in Hyprland execution only if dry-run behavior supports it.

### Sprint 4: Gesture Dataset, Labeling, and Model Evaluation

Turn recordings into evidence:

- label schema and CLI,
- feature extraction,
- DTW/template recognizer,
- optional LSTM/GRU and causal TCN baselines,
- continuous-stream evaluation metrics,
- model-selection decision for the pilot.

### Sprint 5: Study Tooling, Pilot, and Paper Evidence

Turn the prototype into class-ready evidence:

- pilot protocol,
- study event logging,
- baseline task workflow,
- CSV/summary export,
- Caden-only pilot,
- paper outline and limitations.
