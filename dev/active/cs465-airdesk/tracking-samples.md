# AirDesk Tracking Samples

## Purpose

These samples are short local recordings used to decide whether live tracking is good enough for command-mode work.

Use live tuning first when you want immediate feedback. Record after the live numbers look plausible so threshold changes can be checked against the same landmark stream later.

Record landmarks/events by default. Do not record raw video unless there is a deliberate reason and the file is kept out of Git.

## Setup

Install live dependencies:

```bash
uv sync --dev --extra live
```

Start by probing lower-latency camera settings:

```bash
uv run airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG
uv run airdesk camera probe --device /dev/video0 --width 1280 --height 720 --fps 30 --fourcc MJPG
uv run airdesk camera modes --device /dev/video0
```

Prefer the lowest resolution that keeps the hand landmarks stable.

## Live Tuning

Use the live view first to position your hand and inspect MediaPipe overlays:

```bash
uv run airdesk view --device /dev/video0
```

The preview shows the camera frame, hand landmarks, hand skeleton, bounding box, handedness/confidence label, and hand count. Press `q` or `esc` in the preview window to quit.

The preview is mirrored by default so setup feels like looking into a mirror. Use `--no-mirror` if you want the raw camera orientation.

Then use numeric tuning while moving through open palm, fist, and pinch:

```bash
uv run airdesk tune --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --show
```

The live output prints:

- frame FPS
- hand count and confidence
- extended/folded finger counts
- finger spread
- thumb/index pinch distance
- recognized primitive candidates

Useful threshold experiments:

```bash
uv run airdesk tune --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --pinch-threshold 0.08
uv run airdesk tune --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --extended-threshold 0.06
```

## MediaPipe Runtime Tuning

The current backend uses MediaPipe Tasks Hand Landmarker. The old `mp.solutions.hands` style `model_complexity` knob is not exposed here; model selection is done by swapping the `.task` bundle with `--model-path`, while runtime behavior is tuned with hand count and confidence thresholds.

Start with the fast single-hand path:

```bash
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --max-num-hands 1
```

Compare against two-hand tracking before enabling it by default:

```bash
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --max-num-hands 2
```

Then test threshold trade-offs:

```bash
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --min-detection-confidence 0.4 --min-presence-confidence 0.4 --min-tracking-confidence 0.4
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --min-detection-confidence 0.7 --min-presence-confidence 0.7 --min-tracking-confidence 0.7
```

Lower thresholds may keep hands present through motion but can admit shakier detections. Higher thresholds may reduce false positives but can drop hands during fast movement. Record the FPS, hand-present frames, lighting, and visible jitter for each setup.

If a heavier or alternate Hand Landmarker `.task` bundle is available, keep it in ignored `data/models/` and compare it directly:

```bash
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --model-path data/models/hand_landmarker.task
```

## Recommended Samples

Each sample should be 5-10 seconds:

```bash
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label open-palm-hold --out data/recordings/open-palm-hold.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label fist-hold --out data/recordings/fist-hold.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label pinch-hold --out data/recordings/pinch-hold.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label no-hand --out data/recordings/no-hand.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 5 --label normal-desk-motion --out data/recordings/normal-desk-motion.jsonl
```

Analyze each recording:

```bash
uv run airdesk analyze data/recordings/open-palm-hold.jsonl
uv run airdesk replay data/recordings/open-palm-hold.jsonl
```

## Sprint 3 Dynamic Gesture Samples

Collect continuous positive and negative streams before training or trusting a learned recognizer:

```bash
uv run airdesk collect --out-dir data/recordings/sprint4-smoke --label swipe-left-positive --label swipe-right-positive --label normal-desk-motion-negative --reps 5 --duration 6 --countdown 3 --show
```

The collect command shows the webcam preview, prints the next take, starts a countdown, displays a recording status while writing JSONL, and then prompts to keep, redo, skip, or quit. Use `--auto-keep` for scripted/replay checks.

Lower-level one-off commands are still useful when you want explicit filenames:

```bash
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 10 --label swipe-left-positive --out data/recordings/swipe-left-positive.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 10 --label swipe-right-positive --out data/recordings/swipe-right-positive.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 15 --label normal-desk-motion-negative --out data/recordings/normal-desk-motion-negative.jsonl
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 15 --label aborted-gestures-negative --out data/recordings/aborted-gestures-negative.jsonl
```

These are continuous frame-over-frame landmark streams. Do not try to squeeze a gesture into exactly 30 frames. Sprint 4 feature extraction and causal TCN training can use sliding windows internally, but collection should include lead-in, arming, stroke, release, aborted motion, and background.

Analyze both static and dynamic candidates:

```bash
uv run airdesk analyze data/recordings/swipe-left-positive.jsonl
uv run airdesk replay data/recordings/swipe-left-positive.jsonl
uv run airdesk run --backend replay --recording data/recordings/swipe-left-positive.jsonl --profile configs/profiles/window-manager.toml --dry-run --events-out data/logs/swipe-left-replay.jsonl
```

Record negative streams where hands enter/leave the frame, reach for the keyboard or mouse, and rest near the desk without intending a command. These are more important for false-activation evidence than clean isolated clips.

## Notes Template

For each sample, write:

- camera device:
- requested settings:
- actual average FPS:
- lighting:
- approximate hand distance:
- candidate counts:
- false positives:
- false negatives:
- visible jitter/failure notes:

## Current Observations

Initial Sprint 1 smoke tests:

- `/dev/video0` opened at `1920x1080`.
- OpenCV reported `5.00` FPS at default settings.
- MediaPipe Hand Landmarker ran through the AirDesk pipeline.
- A two-frame smoke recording replayed successfully with zero hands detected because no deliberate hand sample was recorded.

Sprint 2 should replace this with deliberate hand-in-frame samples before real desktop execution is considered.

Sprint 3 implementation notes:

- `airdesk analyze` and `airdesk replay` now count `swipe_left`, `swipe_right`, `point_left`, and `point_right` candidates in addition to static primitives.
- `point_left` and `point_right` are implemented as rule-based dry-run candidates and should remain lower priority than swipes until real sample logs show low false positives.
- A bounded smoke on `/dev/video0` opened `640x480 @ 30 FPS MJPG` and ran one MediaPipe dry-run frame with `frames=1 events=2 actions=0`.
- Real local sample results are still pending in this environment. Do not mark live reliability proven until Caden records and analyzes the recommended samples above.
