# AirDesk

AirDesk is a CS465 HCI / 3DUI research project and personal computing prototype exploring webcam-based mid-air hand gestures as an OS-level spatial input layer for a Hyprland Linux desktop.

The project is motivated by **situationally impaired interaction**: moments when keyboard and mouse are inconvenient, unavailable, dirty, or physically costly, such as cooking, painting, repairing hardware, presenting away from a desk, wearing gloves, or managing wrist strain.

The long-term vision is broader than a small gesture demo: AirDesk should become a pluggable, profile-driven desktop control system where webcam, depth sensors, hand gestures, keyboard, mouse, and desktop context can blend into practical command, cursor, media, presentation, accessibility, and hybrid interaction modes.

Start here:

- `dev/active/cs465-airdesk/context.md` - current state and project framing
- `dev/active/cs465-airdesk/plan.md` - research plan, prototype scope, study design
- `dev/active/cs465-airdesk/architecture.md` - proposed system architecture and package boundaries
- `dev/active/cs465-airdesk/research-notes.md` - technical research notes and current working positions
- `dev/active/cs465-airdesk/dynamic-gesture-research.md` - dynamic gesture recognition research and model strategy
- `dev/active/cs465-airdesk/sprint-0.md` - first implementation sprint plan and acceptance criteria
- `dev/active/cs465-airdesk/sprint-1.md` - live camera/tracking/recording sprint plan
- `dev/active/cs465-airdesk/sprint-2.md` - tracking-quality and dry-run command-mode plan
- `dev/active/cs465-airdesk/sprint-3.md` - pilot-safe live command-mode plan
- `dev/active/cs465-airdesk/sprint-4.md` - gesture dataset, labeling, and model-evaluation plan
- `dev/active/cs465-airdesk/sprint-5.md` - study tooling, pilot, and paper-evidence plan
- `dev/active/cs465-airdesk/tracking-samples.md` - local tracking sample protocol
- `dev/active/cs465-airdesk/tasks.md` - implementation and paper checklist
- `dev/active/cs465-airdesk/handoff-prompt.md` - prompt for a fresh agent
- `dev/active/cs465-airdesk/context-reset-prompt.md` - concise prompt for clearing context and restarting

## Development

AirDesk currently uses Python, `uv`, `ruff`, and `pytest`.

```bash
uv sync --dev
uv run airdesk --help
uv run pytest
uv run ruff check .
```

Live camera/tracking support is optional:

```bash
uv sync --dev --extra live
uv run airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG
uv run airdesk view --device /dev/video0
uv run airdesk tune --device /dev/video0 --max-frames 300 --show
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120
uv run airdesk track --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --no-show
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --out data/recordings/sample.jsonl
uv run airdesk replay data/recordings/sample.jsonl
uv run airdesk analyze data/recordings/sample.jsonl
```

The MediaPipe backend uses the Tasks Hand Landmarker API and downloads the model bundle into ignored `data/models/` on first use.
MediaPipe tuning flags include `--model-path`, `--max-num-hands`, `--min-detection-confidence`, `--min-presence-confidence`, and `--min-tracking-confidence`.
The CLI defaults to one hand for lower latency; use `--max-num-hands 2` when comparing two-hand tracking.

Offline ML training is also optional:

```bash
uv sync --dev --extra ml
```

Useful safe commands:

```bash
uv run airdesk doctor
uv run airdesk camera list
uv run airdesk camera probe --device /dev/video0
uv run airdesk profile validate configs/profiles/study-safe.toml
uv run airdesk replay tests/fixtures/replay-one-frame.jsonl
uv run airdesk run --backend replay --recording tests/fixtures/replay-one-frame.jsonl --profile configs/profiles/study-safe.toml --dry-run --events-out data/logs/replay-dry-run.jsonl
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --dry-run --show --events-out data/logs/live-window-manager-dry-run.jsonl
uv run airdesk collect --out-dir data/recordings/sprint4-smoke --label swipe-left-positive --label swipe-right-positive --reps 5 --duration 6 --countdown 3 --show
uv run airdesk collection-summary data/recordings/sprint4-smoke
uv run airdesk label init data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --out data/labels/swipe-left-positive-001.labels.json
uv run airdesk label suggest data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --gesture swipe_left --out data/labels/swipe-left-positive-001.labels.json --apply
uv run airdesk label add-phase data/labels/swipe-left-positive-001.labels.json --phase stroke_left --start 2.4 --end 3.1 --gesture swipe_left
uv run airdesk label add-event data/labels/swipe-left-positive-001.labels.json --gesture swipe_left --start 2.4 --end 3.1
uv run airdesk label validate data/labels/swipe-left-positive-001.labels.json
uv run airdesk features export data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/features/swipe-left-positive-001.csv
uv run airdesk gesture evaluate --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-rule.json
uv run airdesk gesture calibrate --kind dtw --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/models/gestures/caden-dtw.json
uv run airdesk gesture evaluate --recognizer dtw --model data/models/gestures/caden-dtw.json --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-dtw.json
uv run airdesk gesture build-tcn-dataset --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/models/gestures/tcn-sprint4-swipes-001-manifest.json
uv run airdesk gesture train-tcn --manifest data/models/gestures/tcn-sprint4-swipes-001-manifest.json --out data/models/gestures/tcn-sprint4-swipes-001.pt --epochs 25
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout.json
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary-gated.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --negative-distance-margin 1.3 --min-palm-dx-fraction 0.65
uv run airdesk gesture spot-dtw --recording data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --out data/evaluations/sprint4-chained-001/gated-dtw-candidates.json
uv run airdesk gesture score-sequence --candidates data/evaluations/sprint4-chained-002/gated-dtw-candidates.json --expected-sequence "R L R R L L R R L L" --out data/evaluations/sprint4-chained-002/gated-dtw-sequence-score.json
uv run airdesk hyprland dry-run workspace r+1
uv run airdesk cursor run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --events-out data/logs/cursor-dry-run.jsonl
```

In `airdesk collect --show`, use the webcam preview itself: `space` starts the countdown, then `k` keeps, `r` redoes, `s` skips, and `q` quits.

Sprint 3 guarded real execution is opt-in and allowlisted:

```bash
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --execute --allow-profile-execute --show --events-out data/logs/live-window-manager-execute.jsonl
```

Dry-run remains the default. Use `--pause-on-start` or press `p` in the live preview to suppress actions while tracking continues.

Cursor control is also dry-run by default. Real cursor movement is opt-in and uses Hyprland's `movecursor` dispatcher:

```bash
uv run airdesk cursor run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --execute --events-out data/logs/cursor-execute.jsonl
```

In cursor mode, pinch-hold activates relative cursor movement, releasing the pinch exits cursor movement, `p` pauses/resumes, and `q`/`esc` exits. Mouse click/drag injection is intentionally not enabled yet because this machine does not currently have a pointer-button injector installed.

`airdesk label suggest` is a bootstrap helper for dynamic gestures. It finds the strongest palm-motion window in a recording, applies a phase/event label, and should still be reviewed before training or evaluation.
`airdesk gesture build-tcn-dataset` builds a dependency-free JSON manifest of sliding windows over exported CSV features. The first target is intentionally narrow: `background`, `swipe_left`, and `swipe_right`. The manifest stores feature-file paths, row ranges, target labels, and frame-count summaries; it does not train a model and does not add PyTorch to the base runtime.
`airdesk gesture train-tcn` is an optional offline PyTorch training scaffold for that manifest. It saves a checkpoint with model weights, target mapping, feature columns, normalization stats, window settings, and training metrics. Keep it in replay/evaluation workflows until a later TCN evaluation beats gated DTW on held-out continuous sessions.
`airdesk gesture calibrate --kind dtw` builds a dependency-free personalized template model for replay evaluation; keep it in dry-run/evaluation workflows until false activations are low on negative recordings.
`airdesk gesture holdout-dtw` runs a deterministic train/test replay evaluation for a collection batch and writes closest-window diagnostics for rejected DTW matches. The first `sprint4-swipes-001` holdout matched 2/4 held-out swipes, missed both held-out left swipes, and produced 0 false activations on two held-out negative recordings, so the same-batch DTW result should still be treated as optimistic. An optional calibrated horizontal-displacement gate is available through `--min-palm-dx-fraction`; the first gated variant matched 4/4 held-out swipes with 0 held-out false activations, but it still needs a fresh chained recording before live-control use.
`airdesk gesture spot-dtw` runs a DTW model over an unlabeled continuous recording and exports candidate timestamps for review.
`airdesk gesture score-sequence` compares spotted candidates with a remembered R/L order when exact timestamps are not available.

Tests and replay do not require webcam, Hyprland, or MediaPipe access.
