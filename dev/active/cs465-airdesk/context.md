# AirDesk Context

## Date

2026-05-06

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
- Cursor mode now has an explicit `airdesk cursor run` command. Dry-run is default; `--execute` uses Hyprland `movecursor` for real cursor movement while pinch is held. Release exits cursor movement, `p` pauses/resumes, and `q`/`esc` exits. Click/drag injection is still pending because no pointer-button injector is installed locally.
- `airdesk label suggest` can bootstrap swipe labels by finding the strongest palm-motion window and applying phase/event labels for review. This is a labeling accelerator, not a final recognizer.
- `airdesk gesture calibrate --kind dtw` and `airdesk gesture evaluate --recognizer dtw --model ...` now provide a dependency-free personalized DTW/template baseline for replay evaluation.
- Sprint 4 batch `data/recordings/sprint4-swipes-001` has 24 local takes: 8 left swipes, 8 right swipes, and 8 normal desk-motion negatives. Generated labels/features/evaluations live under ignored `data/labels`, `data/features`, and `data/evaluations`.
- Rule recognizer result on that batch was poor: 0/16 positive swipe matches and high static-pose false activations.
- DTW calibrated on the same batch matched all 16 intended swipes, missed 0, produced 18 candidates total, 2 false activations, 0 repeated fires, and about 0.44 s mean latency. Negative recordings produced 0 DTW candidates.
- DTW holdout evaluation now exists via `airdesk gesture holdout-dtw`. On a deterministic `sprint4-swipes-001` split using takes 001-006 for train and 007-008 for test, DTW matched 2/4 held-out swipes, missed both held-out left swipes, produced 0 false activations on two held-out negatives, and had about 0.40 s mean latency on matched events.
- Holdout diagnostics show that loosening the left threshold enough to catch both held-out left swipes also introduces false activations. Treat this as a left-swipe/negative feature-separation problem, not a threshold-tuning win.
- An optional calibrated horizontal-displacement gate now exists for DTW. With `--negative-distance-margin 1.3 --min-palm-dx-fraction 0.65`, the same deterministic holdout matched 4/4 held-out swipes with 0 false activations and about 0.36 s mean latency. Because this was tuned after viewing the holdout, it needs fresh chained-session validation before live control.
- `airdesk gesture spot-dtw` now spots DTW candidates in unlabeled continuous recordings. On fresh chained recording `data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl`, gated DTW found 16 candidates, roughly matching Caden's "15-ish" swipe count and including back-to-back swipe clusters.
- `airdesk gesture score-sequence` now compares spotted DTW candidates with a remembered R/L order. On structured chained recording `data/recordings/sprint4-chained-002/chained-structured-swipes-001.jsonl`, expected sequence `R L R R L L R R L L` scored against detected sequence `R L R R L R R L` as 8/10 matched in order, 2 missed-or-wrong-order, and 0 extra-or-wrong-order.
- The causal TCN scaffold now has deterministic manifest/window building, optional PyTorch training, same-batch evaluation, and holdout evaluation. Same-batch TCN was optimistic: 16/16 matched with 1 false activation. Holdout TCN reproduced the plain-DTW weakness: 2/4 held-out swipes matched, both held-out left swipes missed, and 0 false activations.
- `airdesk gesture diagnose-features` now compares feature, timing, and tracking-quality summaries across that same split. On `sprint4-swipes-001`, held-out left swipes are weaker/slower than train-left examples (`palm_dx` about `0.181` vs `0.235`, normalized `palm_dx_per_hand_scale` about `1.387` vs `1.857`, max palm speed about `3.230` vs `5.163`), while label/frame alignment looks consistent at about one frame inside the event interval.
- Feature export now includes causal trailing-window swipe features: signed horizontal palm displacement, hand-scale-normalized displacement, peak absolute horizontal velocity, and direction consistency. DTW models remain backward-compatible with older saved feature vectors.
- With the new features, DTW on the same deterministic holdout can match 4/4 held-out swipes with 0 false activations using `--negative-distance-margin 0.75`, but this was tuned on existing evidence. TCN holdout still misses both held-out left swipes and adds 1 false activation. On the structured chained session, the looser `1.3` gated DTW variant with new features scored 9/10 in order with 1 extra-or-wrong-order detection, while the conservative `0.75` variant under-detected badly.
- Caden recorded `data/recordings/sprint4-chained-003/chained-structured-swipes-001.jsonl` with a 10-seconds-active / 10-seconds-rest protocol and coarse half-window event labels for sequence `R L R R L L R R L L`. Old gated DTW matched 7/10 event windows with 0 false activations and scored 8/10 in order. The looser window-feature gated DTW matched 8/10 event windows, scored 10/10 in order, but produced 3 extra-or-wrong-order sequence detections and 2 repeated fires. Conservative `0.75` DTW matched only 1/10 in the continuous stream. TCN matched 3/10.
- `airdesk gesture watch-tcn` now provides a live/replay TCN classifier preview. It displays the latest target/probabilities in the webcam preview and prints non-background predictions by default. This is an observation/debug tool only and does not trigger desktop actions.
- `airdesk gesture chart-record` now provides a structured "Guitar Hero for swipes" collection path. A compact chart such as `RR | rest | RL | rest | RRR` expands into get-ready, stroke, reset, and rest prompt windows in the live preview, waits for space before countdown by default, records the replayable landmark stream, and writes coarse chart-derived stroke/recovery/event labels beside the recording. These labels capture prompt timing and should still be treated as weak labels until reviewed.
- `airdesk benchmark` now reports timing slices for live MediaPipe runs: camera read, color conversion, MediaPipe inference, normalization, preview draw, and total loop time.
- The T550 GPU path is now opt-in through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`. Plain `--hand-delegate gpu` can still land on Intel/Mesa EGL under Hyprland; the launcher selects the NVIDIA GLVND EGL vendor and Wayland EGL platform before Python starts. The success signal is a MediaPipe log line containing `OpenGL ES 3.2 NVIDIA` and `NVIDIA T550 Laptop GPU`.
- Short bounded smoke evidence on 2026-05-06: CPU delegate inference averaged about `16.84 ms`; plain GPU-on-Intel averaged about `13.60 ms`; T550 GPU via the launcher averaged about `4.17 ms`. Capture is still camera-paced around 30 FPS, so the practical benefit should be judged with hand-in-frame fast-swipe tracking continuity, not FPS alone.

Current next step:

> Use the chart recorder to collect a larger GPU-tracked swipe dataset with singles, right-heavy chains, mixed chains, alternating chains, and normal background/rest windows. Then rebuild labels/features and compare DTW, live-optimized DTW, and TCN/event-decoded paths in replay before considering any guarded desktop action wiring.

## Current Research Direction Update

Caden's live `watch-tcn` test showed that `swipe_left` works better than `swipe_right`, fast consecutive swipes are weak, and the model often needs the hand to reset before another gesture. This is consistent with the continuous-gesture literature: the hard problem is spotting gesture events inside an untrimmed stream, not classifying a clean fixed window.

The first continuous-spotting implementation pass is now in place:

- TCN manifests support `--feature-preset stream-invariant`, which excludes absolute `palm_x`, `palm_y`, and `palm_z`.
- TCN manifests support `--target-mode phase`, with default targets `background`, `stroke_left`, `stroke_right`, and `recovery`.
- Label files accept `recovery` / `reset` phases, and `airdesk label add-sequence` can create coarse ordered L/R stroke+recovery labels for chained sessions when exact timestamps are unavailable.
- `airdesk gesture evaluate-tcn --event-decoder` and `airdesk gesture decode-candidates` add a replayable hysteresis/peak/cooldown decoder over probability or candidate streams.

Initial replay evidence is still mixed:

- Current gated DTW with window features remains the best isolated holdout result: 4/4 matched, 0 false activations, 0 repeated fires.
- Current TCN with window features remains weak on the same holdout: 2/4 matched, 1 false activation, both held-out left swipes missed.
- TCN plus the new event decoder on that holdout matched 3/4 but introduced 2 false activations at permissive thresholds, so the decoder is not a free reliability win.
- A new stream-invariant phase TCN/event-decoder holdout smoke matched 2/4 with 1 false activation. The representation/target plumbing works, but the current local data is not enough to improve the model yet.
- Decoding the looser chained-session DTW candidates reduced repeated fires from 2 to 0 on `sprint4-chained-003`, but also reduced matches from 8/10 to 6/10. This is useful filtering behavior, not enough evidence for live control.

Updated stance:

- treat the current TCN as a scaffold, not the destination;
- make learned features less dependent on absolute frame position and distance from camera;
- train toward phase/event stream labels such as `background`, `stroke_left`, `stroke_right`, and `recovery`;
- add an event decoder with hysteresis, confidence peaks, cooldown, and repeated-fire suppression;
- use DTW/template and motion-energy gates as low-data personalization and candidate-filtering tools;
- consider graph/transformer memory only after labels and event decoding are in place.

See `dynamic-gesture-research.md` for the deeper research notes and source anchors.

## Current Roadmap

### Sprint 3: Pilot-Safe Live Command Mode

Build the live command loop:

- runtime event logs,
- continuous positive/negative recordings,
- intent-gated phrase recognizer foundation,
- rule/DTW dynamic gesture scaffolding,
- live command-state feedback,
- pause/kill switch,
- guarded opt-in Hyprland execution only if dry-run behavior supports it.

### Sprint 4: Gesture Dataset, Labeling, and Continuous Gesture Spotting

Turn recordings into evidence:

- label schema and CLI,
- feature extraction,
- DTW/template baseline and holdout evaluation,
- train and evaluate one small causal stream model over AirDesk features,
- rule/DTW fallback for safety, calibration, and debugging,
- LSTM/GRU deferred unless the TCN path fails,
- continuous-stream evaluation metrics,
- Sprint 5 recognizer decision for the pilot,
- position-invariant features, phase labels, and event decoding before live actions.

### Sprint 5: Study Tooling, Pilot, and Paper Evidence

Turn the prototype into class-ready evidence:

- pilot protocol,
- study event logging,
- baseline task workflow,
- CSV/summary export,
- Caden-only pilot,
- paper outline and limitations.
