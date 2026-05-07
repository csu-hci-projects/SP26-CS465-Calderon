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

## NVIDIA T550 / Hyprland GPU Path

On Caden's current Arch/Hyprland hybrid-graphics setup, plain `--hand-delegate gpu` can initialize MediaPipe on the Intel/Mesa EGL renderer even though `prime-run glxinfo` uses the T550. MediaPipe's GPU delegate uses EGL/OpenGL ES, so the useful local check is the MediaPipe startup log, not just `glxinfo`.

The working opt-in launcher is:

```bash
scripts/airdesk-nvidia-mediapipe-wayland benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --max-frames 120
```

Use the same launcher for live previews:

```bash
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-dtw --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-window-features-gated.json --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-tcn --model data/models/gestures/tcn-sprint4-swipes-001-holdout-window-features.pt --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
```

A successful T550 MediaPipe run prints a startup line like:

```text
GL version: 3.2 (OpenGL ES 3.2 NVIDIA ...), renderer: NVIDIA T550 Laptop GPU/PCIe/SSE2
```

If it prints `Mesa Intel(R) Iris(R) Xe Graphics`, MediaPipe is using the integrated GPU path. If it fails with `eglGetDisplay`, check that `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists and that the launcher is being used from the repo root. The launcher sets `__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, and `EGL_PLATFORM=wayland` before Python starts; setting these inside the Python backend was not reliable because GLVND/EGL selection can happen before backend initialization. It also sets `QT_QPA_PLATFORM=xcb` so OpenCV's Qt preview can use Xwayland instead of looking for the missing Qt Wayland plugin in the `cv2` wheel, and `QT_QPA_FONTDIR=/usr/share/fonts/TTF` to point Qt at system fonts.

Use timing diagnostics when the live recognizer feels laggy:

```bash
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-dtw --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-window-features-gated.json --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show --profile-timing
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-tcn --model data/models/gestures/tcn-sprint4-swipes-001-holdout-window-features.pt --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show --profile-timing
```

`watch-dtw` uses a live-optimized latest-window scan. Offline DTW evaluation still scans all candidate windows, but live preview only scores windows ending at the newest usable hand frame so it does not repeatedly rescan the whole rolling buffer.

Short smoke evidence from 2026-05-06 on `/dev/video0` at `640x480 @ 30 FPS MJPG`:

- CPU delegate on Intel/Mesa EGL: MediaPipe inference mean about `16.84 ms`, p95 about `22.39 ms`.
- Plain GPU delegate on Intel/Mesa EGL: MediaPipe inference mean about `13.60 ms`, p95 about `19.89 ms`.
- NVIDIA launcher plus GPU delegate on T550: MediaPipe inference mean about `4.17 ms`, p95 about `3.91 ms` in a 20-frame smoke.

Treat those as bounded startup smokes, not final tracking-quality evidence. The benchmark now reports timing slices for capture read, color conversion, MediaPipe inference, normalization, preview draw, and total loop time. Capture read is camera-paced and often dominates at 30 FPS; the key T550 win is inference headroom and lower tracking latency under real hand motion.

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
scripts/airdesk-nvidia-mediapipe-wayland collect --out-dir data/recordings/sprint4-gpu-swipes-001 --label swipe-left-positive --label swipe-right-positive --label normal-desk-motion-negative --reps 5 --duration 6 --countdown 3 --hand-delegate gpu --show
```

The collect command shows the webcam preview before recording so you can position your hand in frame. In the preview window:

- `space` starts the countdown.
- `k` keeps a finished take.
- `r` redoes a finished take.
- `s` skips the current take.
- `q` quits collection.

The preview displays countdown and recording status while writing JSONL. Use `--auto-keep` for scripted/replay checks.

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

## Continuous Spotting Samples

The next recognizer direction is continuous gesture spotting rather than fixed-window classification. Future recordings should make the failure modes visible:

- fast chained swipes without a full hand reset,
- slow swipes with long preparation/recovery,
- same direction repeated twice, such as `R R` or `L L`,
- alternating directions, such as `R L R L`,
- starts from left, center, and right side of the preview,
- near, normal, and farther hand distance from the camera,
- normal desk motion between gestures,
- aborted half-swipes and hand repositions.

Use ordered-sequence notes when exact timestamps are too much to track. For example:

```text
0-10s active: R then L
10-20s rest/background
20-30s active: R then R
30-40s rest/background
40-50s active: L then L
```

This is not as precise as frame labels, but it is useful for weak sequence scoring and later CTC-style alignment. Keep the direction meaning user-facing: `R` means palm motion toward the right side of the preview/screen and `L` means palm motion toward the left side, regardless of raw camera coordinate sign.

For structured chained takes, `record --show` can display a countdown and prompt timeline in the preview:

```bash
scripts/airdesk-nvidia-mediapipe-wayland record --out data/recordings/sprint4-gpu-swipes-002-structured/structured-a-right-heavy.jsonl --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 82 --countdown 3 --wait-for-space --label structured-a-right-heavy --hand-delegate gpu --show --segment "0:10:R R" --segment "10:20:rest" --segment "20:30:R L" --segment "30:40:rest" --segment "40:50:R R R" --segment "50:60:rest" --segment "60:70:R L R" --segment "70:80:rest"
```

The faster path is now the chart recorder. It expands a compact pattern into an on-screen colored timing lane with get-ready, stroke, reset, and rest windows; waits for space by default; and writes coarse stroke/recovery/event labels beside the recording. A combo block such as `RRR` stays grouped as one active prompt (`SWIPE R R R`) so the individual swipes can happen at a natural pace inside that block instead of being flashed one at a time:

```bash
scripts/airdesk-nvidia-mediapipe-wayland gesture chart-record --out data/recordings/sprint4-gpu-swipes-002-structured/chart-a-right-heavy.jsonl --chart "RR | rest | RL | rest | RRR | rest | RLR | rest" --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
scripts/airdesk-nvidia-mediapipe-wayland gesture chart-record --out data/recordings/sprint4-gpu-swipes-002-structured/chart-b-mixed.jsonl --chart "LR | rest | RR | rest | LRR | rest | RRL | rest" --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
scripts/airdesk-nvidia-mediapipe-wayland gesture chart-record --out data/recordings/sprint4-gpu-swipes-002-structured/chart-c-alternating.jsonl --chart "RLR | rest | LRL | rest | RRL | rest | LRR | rest" --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
```

Default chart timing is `1.5s` cue, `0.75s` stroke, `0.75s` recovery, and `10s` rest. Adjust with `--cue-seconds`, `--gesture-seconds`, `--recovery-seconds`, and `--rest-seconds` if the prompts feel too tight or too slow. If a recording was made without labels, rebuild the same coarse labels later:

```bash
uv run airdesk gesture chart-label data/recordings/sprint4-gpu-swipes-002-structured/chart-a-right-heavy.jsonl --chart "RR | rest | RL | rest | RRR | rest | RLR | rest" --out data/labels/sprint4-gpu-swipes-002-structured/chart-a-right-heavy.labels.json
```

For a coarse active window with a known order, bootstrap weak labels with:

```bash
uv run airdesk label add-sequence data/labels/sprint4-chained-003/chained-structured-swipes-001.labels.json --sequence "R L R R L L R R L L" --start 0 --end 100
```

This creates evenly spaced stroke/recovery phases and gesture events. It is intended for quick replay scoring and later weak-alignment experiments; refine timestamps manually before treating the labels as final training truth.

Position/distance should not become the gesture identity. When collecting model data, deliberately vary setup after a few takes:

- hand starts left/center/right in frame,
- camera sees the hand close/far,
- elbow/arm posture changes,
- lighting and background remain normal rather than lab-perfect.

Do not wire any recognizer into live desktop actions based on these recordings alone. Use them to evaluate false activations, missed gestures, repeated fires, and latency in replay.

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
- `/dev/video0` advertises MJPG modes up to 1920x1080 at 30 FPS. Requesting `640x480 @ 60 FPS MJPG` falls back to 30 FPS, so 60 FPS does not appear to be supported by this webcam.
- Real local sample results are still pending in this environment. Do not mark live reliability proven until Caden records and analyzes the recommended samples above.

Sprint 4 smoke collection notes:

- Run `uv run airdesk collection-summary data/recordings/sprint4-smoke` to summarize the prompted collection batch.
- Run `uv run airdesk label init data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --out data/labels/swipe-left-positive-001.labels.json` to create a starter label file.
- Add rough relative-time labels with commands such as `uv run airdesk label add-phase data/labels/swipe-left-positive-001.labels.json --phase stroke_left --start 2.4 --end 3.1 --gesture swipe_left` and `uv run airdesk label add-event data/labels/swipe-left-positive-001.labels.json --gesture swipe_left --start 2.4 --end 3.1`.
- You can still edit `event_labels` and `phase_labels` manually, then run `uv run airdesk label validate data/labels/swipe-left-positive-001.labels.json`.
- Run `uv run airdesk features export data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/features/swipe-left-positive-001.csv` to export deterministic frame features for inspection/model work.
- Run `uv run airdesk gesture evaluate --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-rule.json` to evaluate the current rule recognizer against event labels.
- Caden's first natural swipe/negative batch recorded 15 takes at about 29.6 FPS.
- Current swipe phrase rules produced zero `swipe_left` / `swipe_right` candidates because they depend on a brittle static `open_palm` arm.
- Natural negative motion produced substantial crude `pinch` / `fist` counts, which reinforces the need for phase labels, negative data, and a causal temporal recognizer rather than a static-pose-only command path.

Sprint 4 swipe batch `data/recordings/sprint4-swipes-001`:

- Caden recorded 24 prompted takes: 8 `swipe-left-positive`, 8 `swipe-right-positive`, and 8 `normal-desk-motion-negative`.
- Each take has 238 frames at about 29.65 FPS.
- Labels were generated under ignored `data/labels/sprint4-swipes-001`:
  - positive swipes use `airdesk label suggest` phase/event labels,
  - negative takes use background-only starter labels.
- Features were exported under ignored `data/features/sprint4-swipes-001`.
- Rule evaluations were exported under ignored `data/evaluations/sprint4-swipes-001`.
- Current rule recognizer result on 16 positive swipe takes: 16 intended, 0 matched, 16 missed, 1707 total candidates, 1543 false activations.
- Current rule recognizer result on 8 negative takes: 1221 candidates, all counted as false activations, mainly `fist` and `pinch`.
- The `label suggest` observed direction is opposite the intended left/right labels in this batch, likely because user-facing mirrored motion and raw camera coordinates differ. Do not hard-code camera-left/camera-right as the final semantic direction without calibration.
- This batch is enough to start a DTW/template baseline or first causal TCN prototype; it is not evidence that the current rule recognizer is acceptable for live swipe control.
- DTW baseline model `data/models/gestures/caden-dtw-sprint4-swipes-001.json` was calibrated from the 16 positive swipe labels plus 8 negative/background labels.
- DTW evaluation over all 24 takes: 16 intended, 16 matched, 0 missed, 18 candidates, 2 false activations, 0 repeated fires, about 0.44 s mean latency.
- Per gesture DTW result: `swipe_left` 8/8 matched with 0 false activations; `swipe_right` 8/8 matched with 2 extra candidates on positive takes; negative recordings produced 0 candidates.
- DTW is now useful as a personalized replay baseline and calibration tool. It is still not ready for live desktop actions until latency, runtime cost, and chained/background recordings are tested.

DTW holdout evaluation:

```bash
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout.json --train-per-gesture 6 --test-per-gesture 2 --train-negatives 6 --test-negatives 2
```

- The deterministic split trains on takes 001-006 for each positive gesture and negative/background group, then tests on takes 007-008.
- Holdout result: 4 intended held-out swipes, 2 matched, 2 missed, 2 total candidates, 0 false activations, 0 repeated fires, and about 0.40 s mean latency on matched events.
- Per gesture: held-out `swipe_right` matched 2/2; held-out `swipe_left` matched 0/2.
- Held-out negative recordings `normal-desk-motion-negative-007` and `normal-desk-motion-negative-008` produced 0 candidates.
- Interpretation: DTW remains promising as a low-data personalized baseline, but the holdout split exposes left-swipe generalization weakness. Do not wire DTW swipes into live desktop actions or start making reliability claims from the same-batch score.

DTW left-miss diagnostic notes:

- The holdout JSON now includes `diagnostics` with the closest DTW window per gesture even when the recognizer rejects it.
- The train-only model threshold for `swipe_left` is `0.417` because negative calibration clamps it from a raw template threshold of about `2.78`; the closest held-out left distances were about `0.618` and `0.485`.
- Raising the negative margin enough to match both held-out left swipes also introduced false activations on held-out positive/negative streams, so this should not be solved by simply loosening thresholds.
- The likely issue is weak separation between current left-swipe features/templates and natural desk-motion negatives. Next, inspect label quality and feature separation for left swipes before collecting a longer chained continuous session.

Optional gated DTW variant:

```bash
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary-gated.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --train-per-gesture 6 --test-per-gesture 2 --train-negatives 6 --test-negatives 2 --negative-distance-margin 1.3 --min-palm-dx-fraction 0.65
```

- This variant loosens the negative-distance margin but adds a calibrated horizontal-displacement gate. The gate requires an accepted swipe window to move in the same horizontal direction as its calibrated templates and by at least a fraction of the smallest calibrated swipe displacement.
- Result on the same held-out split: 4 intended swipes, 4 matched, 0 missed, 4 candidates, 0 false activations, 0 repeated fires, and about 0.36 s mean latency.
- Interpretation: this is a promising fix candidate because it addresses the observed false-activation tradeoff directly, but it was tuned after looking at the holdout. Treat it as a hypothesis to validate on a fresh chained continuous recording, not as proof of reliability.

Sprint 4 chained session `data/recordings/sprint4-chained-001`:

- Caden recorded one continuous take, `chained-left-right-swipes-001.jsonl`, with roughly 15 swipes plus natural movement and some back-to-back swipes.
- Recording health: 2669 frames, 1384 hand-present frames, about 29.66 FPS, and about 90 seconds of data.
- Rule recognizer again found 0 swipe candidates, so dynamic swipe evaluation should use DTW/TCN paths rather than static pose rules.
- Gated DTW spotting command:

```bash
uv run airdesk gesture spot-dtw --recording data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --out data/evaluations/sprint4-chained-001/gated-dtw-candidates.json
```

- Gated DTW produced 16 candidates: 10 `swipe_right` and 6 `swipe_left`.
- Candidate timestamps in seconds from recording start: right at 2.36, 24.58, 27.75, 50.24, 52.67, 59.89, 75.20, 77.29, 86.43, 87.94; left at 5.80, 8.76, 58.10, 60.29, 81.74, 83.76.
- This aligns with Caden's rough "15-ish" swipe count and includes back-to-back clusters. It still needs human timestamp review or event labels before reporting matched/missed/false-activation metrics.

Sprint 4 structured chained session `data/recordings/sprint4-chained-002`:

- Caden recorded one structured continuous take, `chained-structured-swipes-001.jsonl`.
- Intended movement-direction sequence: `R L R R L L R R L L`. Here `R` means palm motion toward the right side of the preview/screen and `L` means palm motion toward the left side, regardless of which hand was used.
- Recording health: 2670 frames, 1220 hand-present frames, about 29.66 FPS, and about 90 seconds of data.
- Gated DTW spotting produced 8 candidates: detected sequence `R L R R L R R L`.
- Order-level score:

```bash
uv run airdesk gesture score-sequence --candidates data/evaluations/sprint4-chained-002/gated-dtw-candidates.json --expected-sequence "R L R R L L R R L L" --out data/evaluations/sprint4-chained-002/gated-dtw-sequence-score.json
```

- Result: 8/10 matched in order, 2 missed-or-wrong-order gestures, 0 extra-or-wrong-order detections.
- Interpretation: gated DTW is now finding plausible continuous-session swipes without extra order-level detections, but still misses some gestures in a structured stream. That is good enough evidence to continue model work, not enough evidence for live desktop actions.
- Next model step: use these recordings as evidence for a causal TCN scaffold, starting with deterministic dataset/window construction over exported feature rows. Keep gated DTW as the baseline and keep live desktop actions disabled.

TCN dataset scaffold:

```bash
uv run airdesk gesture build-tcn-dataset --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/models/gestures/tcn-sprint4-swipes-001-manifest.json
uv run airdesk gesture train-tcn --manifest data/models/gestures/tcn-sprint4-swipes-001-manifest.json --out data/models/gestures/tcn-sprint4-swipes-001.pt --epochs 25
uv run airdesk gesture evaluate-tcn --manifest data/models/gestures/tcn-sprint4-swipes-001-manifest.json --model data/models/gestures/tcn-sprint4-swipes-001.pt --out data/evaluations/sprint4-swipes-001-tcn/summary.json
uv run airdesk gesture holdout-tcn --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-tcn-holdout/summary.json --model-out data/models/gestures/tcn-sprint4-swipes-001-holdout.pt
```

- The first TCN target is narrow by design: `background`, `swipe_left`, and `swipe_right`.
- The manifest is dependency-free JSON. It points to exported feature CSV files and records deterministic sliding-window row ranges, target labels, target indexes, feature columns, and per-target counts.
- TCN training is optional and requires `uv sync --dev --extra ml`. It saves a Torch checkpoint with model weights, target mapping, feature columns, normalization stats, window settings, and training metrics.
- This remains an offline training/evaluation path. It must not be wired into live desktop actions until a later evaluation beats the current DTW baseline on held-out continuous sessions.
- Live/replay TCN classifier preview is available for observation only:

```bash
uv run airdesk gesture watch-tcn --model data/models/gestures/tcn-sprint4-swipes-001-holdout-window-features.pt --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --show
```

- The preview overlay shows the latest target and probabilities. The terminal prints non-background predictions by default; add `--include-background` if you want every background window printed too.
- First same-batch TCN run on `sprint4-swipes-001`: `samples=821`, `train_accuracy=0.989`, `validation_accuracy=0.957`; replay evaluation over the same manifest reported 16 intended, 16 matched, 0 missed, 17 candidates, 1 false activation, and about 0.407 s mean latency.
- Interpretation: useful smoke evidence that the model path is functioning, but optimistic because windows come from the same labeled batch. Next required evidence is a deterministic TCN holdout split comparable to `gesture holdout-dtw`.
- TCN holdout using the same 6 train / 2 test per gesture and negative split shape: 4 intended, 2 matched, 2 missed, 2 candidates, 0 false activations, 0 repeated fires, about 0.502 s mean latency. Per gesture: `swipe_right` matched 2/2; `swipe_left` matched 0/2.
- Interpretation: the first causal TCN scaffold works but does not yet beat gated DTW. It reproduces the same left-swipe generalization weakness as plain DTW, so the next useful work is feature/label diagnosis rather than live action wiring.

Feature-diagnostics pass:

```bash
uv run airdesk gesture diagnose-features --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-feature-diagnostics/summary.json --train-per-gesture 6 --test-per-gesture 2 --train-negatives 6 --test-negatives 2
```

- The report uses the same filename-ordered train/test split as DTW/TCN holdout and writes per-file plus aggregate feature summaries.
- Held-out `swipe_left` examples are weaker than train-left examples: mean raw `palm_dx` drops from about `0.235` to `0.181`, mean normalized `palm_dx_per_hand_scale` drops from about `1.857` to `1.387`, and mean max palm speed drops from about `5.163` to `3.230`.
- Held-out `swipe_right` examples also weaken slightly, but less disruptively for the current recognizers: mean raw `palm_dx` changes from about `-0.296` to `-0.269`, normalized displacement from about `-1.622` to `-1.435`, and mean max speed from about `3.234` to `2.553`.
- Label/window alignment does not look like the primary failure: the first feature row inside positive event labels is about `0.012 s` after label start, and the last row is about `0.012 s` before label end for both train and test positives.
- Interpretation: the left-swipe misses are more consistent with feature representation and weak-motion separation than with obviously late/early labels. The next code change should be targeted, likely explicit signed horizontal displacement over windows, hand-scale-normalized displacement, peak horizontal velocity, and a direction-consistency feature, followed by rerunning DTW and TCN holdouts. This is still offline evidence only; do not wire swipes into live actions.

Window-feature rerun:

- Feature export now includes causal trailing-window motion summaries: signed horizontal palm displacement, displacement normalized by hand scale, peak absolute horizontal velocity, and horizontal direction consistency.
- After regenerating `data/features/sprint4-swipes-001`, plain DTW with the new features and the default `negative_distance_margin=0.85` matched 4/4 held-out swipes but produced 1 false activation.
- A more conservative DTW variant with `--negative-distance-margin 0.75` matched 4/4 held-out swipes with 0 false activations and about 0.382 s mean latency on this split. The same result held with `--min-palm-dx-fraction 0.65`.
- TCN holdout with the new features still matched only 2/4 held-out swipes, missed both held-out left swipes, and produced 1 false activation. So the feature addition currently helps the DTW/template path more than the small TCN path.
- Continuous sanity check: the conservative `0.75` gated DTW variant under-detected the structured chained session, scoring only 3/10 in order. The looser `1.3` gated DTW variant with the new window features detected `R L R L L R R L L R`, scoring 9/10 matched in order with 1 extra-or-wrong-order detection.
- Interpretation: the new representation is promising for DTW recall and may improve the structured stream, but the threshold/gate setting is still tuned on existing data and remains unstable across isolated vs chained recordings. Treat this as a hypothesis for the next labeled continuous pass, not live-control evidence.

Sprint 4 structured chained session `data/recordings/sprint4-chained-003`:

- Caden recorded one timestamp-aware structured continuous take, `chained-structured-swipes-001.jsonl`.
- Protocol: 10 seconds active, 10 seconds rest, two swipes per active window. Intended sequence was `R L R R L L R R L L`.
- Recording health: 2670 frames, 592 hand-present frames, about 29.66 FPS, and about 90 seconds of data.
- Coarse labels were generated as half-window event intervals: `0-5`, `5-10`, `20-25`, `25-30`, `40-45`, `45-50`, `60-65`, `65-70`, `80-85`, and `85-89.8` seconds.
- Old gated DTW (`negative_distance_margin=1.3`, `min_palm_dx_fraction=0.65`, old feature set) matched 7/10 coarse event windows, missed 3, produced 8 candidates, 0 false activations, 1 repeated fire, and about 2.14 s mean latency. Order-level score was detected `R L R R L R R L`, or 8/10 matched in order with 0 extra-or-wrong-order detections.
- Window-feature gated DTW with margin `1.3` matched 8/10 coarse event windows, missed 2, produced 13 candidates, 0 false activations, 2 repeated fires, and about 2.05 s mean latency. Order-level score was detected `R L L R R L R L R L R L L`, or 10/10 matched in order with 3 extra-or-wrong-order detections.
- Conservative window-feature gated DTW with margin `0.75` matched only 1/10, so it is too strict for continuous streams despite its clean isolated holdout result.
- TCN with the window-feature holdout checkpoint matched 3/10, missed 7, produced 5 candidates, 0 false activations, and 1 repeated fire.
- Interpretation: this validates the direction of travel but not live readiness. The best current candidate for Sprint 5 is still a DTW/template recognizer, likely the looser window-feature gated variant if extra detections can be filtered with better cooldown/sequence handling. TCN remains behind and should stay offline unless retrained with more continuous labels.
