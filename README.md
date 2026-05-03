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
uv run airdesk label add-phase data/labels/swipe-left-positive-001.labels.json --phase stroke_left --start 2.4 --end 3.1 --gesture swipe_left
uv run airdesk label add-event data/labels/swipe-left-positive-001.labels.json --gesture swipe_left --start 2.4 --end 3.1
uv run airdesk label validate data/labels/swipe-left-positive-001.labels.json
uv run airdesk features export data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/features/swipe-left-positive-001.csv
uv run airdesk hyprland dry-run workspace r+1
```

In `airdesk collect --show`, use the webcam preview itself: `space` starts the countdown, then `k` keeps, `r` redoes, `s` skips, and `q` quits.

Sprint 3 guarded real execution is opt-in and allowlisted:

```bash
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --execute --allow-profile-execute --show --events-out data/logs/live-window-manager-execute.jsonl
```

Dry-run remains the default. Use `--pause-on-start` or press `p` in the live preview to suppress actions while tracking continues.

Tests and replay do not require webcam, Hyprland, or MediaPipe access.
