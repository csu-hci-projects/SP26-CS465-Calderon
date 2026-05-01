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

Use this while moving through open palm, fist, and pinch:

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
