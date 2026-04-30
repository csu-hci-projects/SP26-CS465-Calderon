# AirDesk

AirDesk is a CS465 HCI / 3DUI research project and personal computing prototype exploring webcam-based mid-air hand gestures as an OS-level spatial input layer for a Hyprland Linux desktop.

The project is motivated by **situationally impaired interaction**: moments when keyboard and mouse are inconvenient, unavailable, dirty, or physically costly, such as cooking, painting, repairing hardware, presenting away from a desk, wearing gloves, or managing wrist strain.

The long-term vision is broader than a small gesture demo: AirDesk should become a pluggable, profile-driven desktop control system where webcam, depth sensors, hand gestures, keyboard, mouse, and desktop context can blend into practical command, cursor, media, presentation, accessibility, and hybrid interaction modes.

Start here:

- `dev/active/cs465-airdesk/context.md` - current state and project framing
- `dev/active/cs465-airdesk/plan.md` - research plan, prototype scope, study design
- `dev/active/cs465-airdesk/architecture.md` - proposed system architecture and package boundaries
- `dev/active/cs465-airdesk/research-notes.md` - technical research notes and current working positions
- `dev/active/cs465-airdesk/sprint-0.md` - first implementation sprint plan and acceptance criteria
- `dev/active/cs465-airdesk/sprint-1.md` - live camera/tracking/recording sprint plan
- `dev/active/cs465-airdesk/sprint-2.md` - tracking-quality and dry-run command-mode plan
- `dev/active/cs465-airdesk/tracking-samples.md` - local tracking sample protocol
- `dev/active/cs465-airdesk/tasks.md` - implementation and paper checklist
- `dev/active/cs465-airdesk/handoff-prompt.md` - prompt for a fresh agent

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
uv run airdesk track --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --no-show
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --out data/recordings/sample.jsonl
uv run airdesk replay data/recordings/sample.jsonl
uv run airdesk analyze data/recordings/sample.jsonl
```

The MediaPipe backend uses the Tasks Hand Landmarker API and downloads the model bundle into ignored `data/models/` on first use.

Useful safe commands:

```bash
uv run airdesk doctor
uv run airdesk camera list
uv run airdesk camera probe --device /dev/video0
uv run airdesk profile validate configs/profiles/study-safe.toml
uv run airdesk replay tests/fixtures/replay-one-frame.jsonl
uv run airdesk hyprland dry-run workspace r+1
```

Tests and replay do not require webcam, Hyprland, or MediaPipe access.
