# AirDesk

## Submission Links

Final videos: TBD

Project overview video: TBD

Latex file in repo, link: [paper/latex-source/main.tex](paper/latex-source/main.tex)

Overleaf link: TBD

PDF Literature Survey Link: [paper/PDFs-LiteratureSurvey/](paper/PDFs-LiteratureSurvey/)

Work Allocation: I worked solo, with AI assistance for parts of the code.

Technical Demo/Videos: TBD

Research Paper: [paper/latex-source/main.tex](paper/latex-source/main.tex)

Other notes: AirDesk's current live demo path is deterministic MediaPipe control through `airdesk control run`; learned TCN/IPN/DTW work is retained as research and diagnostic infrastructure, not the live action recognizer.

AirDesk is a webcam-based mid-air desktop control prototype for Hyprland Linux. It uses MediaPipe hand landmarks to turn a small, deliberate hand vocabulary into pointer, click, scroll, launcher, workspace, and window-management actions, with dry-run logging as the default safety mode. The project is designed with situationally impaired interaction in mind: dirty hands, gloves, limited reach, standing away from the desk, temporary pain, presentation contexts, or other moments when touching the keyboard and mouse is inconvenient.

## What to Grade / What Works

- The current live demo path is deterministic MediaPipe control: `airdesk control run`.
- Live actions are dry-run by default. Commands can be demonstrated safely without moving windows or clicking the real desktop.
- Optional real execution exists for the tested Hyprland/Linux setup. Pointer movement, click, and scroll should use `--execute --pointer-execute` so `/dev/uinput` sends normal mouse events with hover/click behavior.
- The implemented control loop includes open-hand cursor movement, index-pinch left click/drag, middle-pinch right click/scroll, fist workspace switching, fist window movement, launcher combo, close-window combo, live preview feedback, and JSONL event logs.
- Learned TCN/IPN/DTW recognizers are retained as research and diagnostic infrastructure. They are not the live global desktop action recognizer because live preview produced too many false activations for safe OS commands.

## Quick Setup

AirDesk uses Python 3.12 and `uv`.

```bash
uv sync --dev
uv run airdesk --help
```

For live webcam control, install the live extras:

```bash
uv sync --dev --extra live
```

For optional offline ML/research commands, install the ML extras too:

```bash
uv sync --dev --extra live --extra ml
```

## Safe Dry-Run Demo

This is the recommended grading/demo command. It shows the webcam overlay, logs what AirDesk sees, and routes actions to dry-run targets.

```bash
uv run airdesk control run \
  --backend mediapipe \
  --device /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --fourcc MJPG \
  --events-out data/logs/control-demo-dry-run.jsonl \
  --show
```

Useful smoke test without a webcam:

```bash
uv run airdesk control run \
  --backend replay \
  --recording tests/fixtures/replay-one-frame.jsonl \
  --events-out data/logs/control-replay-dry-run.jsonl \
  --max-frames 1 \
  --no-show
```

Press `p` in the preview to pause/resume actions while tracking continues. Press `q` or `esc` to exit the preview.

## Optional Live Execute

Real execution is Linux/Hyprland/uinput-specific and should only be used on a machine where `hyprctl` works and `/dev/uinput` is writable by the user.

```bash
uv run airdesk control run \
  --backend mediapipe \
  --device /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --fourcc MJPG \
  --execute \
  --pointer-execute \
  --events-out data/logs/control-demo-execute.jsonl \
  --show
```

`--execute` enables guarded Hyprland dispatches. `--pointer-execute` enables real relative pointer movement, button presses, and scroll through `/dev/uinput`. Without `--pointer-execute`, pointer button/scroll actions remain dry-run and cursor movement falls back to Hyprland cursor dispatch.

## Gesture / Control Cheat Sheet

| Input | Current behavior |
| --- | --- |
| Open or relaxed tracked hand | Move the pointer. |
| Index pinch tap | Left click. |
| Index pinch hold or drag | Hold left button for selection/drag. |
| Middle pinch tap | Right click on clean release. |
| Middle pinch hold plus vertical movement | Scroll; cursor locks while scrolling. |
| Stable fist, then up/down motion | Switch workspace with Hyprland `workspace r-1` / `r+1`. |
| Stable fist, then left/right motion or side zone | Move active/window-under-cursor with `movetoworkspace r-1` / `r+1`. |
| Open palm -> sideways open palm | Open the configured launcher. |
| Open palm -> fist -> open palm | Close the active window with `killactive`. |

The live preview/status line reports what the system is seeing, combo state, armed state, target window, executed action, and suppression reason.

## Repo Map

- `src/airdesk/control/` - deterministic live-control poses, debouncing, combos, grammar, and runtime.
- `src/airdesk/actions/` - dry-run targets, guarded Hyprland dispatch, cursor targets, and uinput pointer injection.
- `src/airdesk/tracking/` - MediaPipe and replay tracking backends.
- `src/airdesk/gestures/` - DTW, motion, learned-filter, and event-decoder diagnostic/research code.
- `src/airdesk/ml/` and `src/airdesk/public_datasets/` - optional TCN/IPN training and evaluation infrastructure.
- `configs/profiles/` - older profile-driven runtime configs.
- `tests/` - unit and CLI coverage, including deterministic control tests.
- `dev/active/cs465-airdesk/` - internal planning/provenance docs for development continuity, not the primary grader entrypoint.

## Architecture Summary

The live demo path is intentionally simple and dry-run-first:

```text
webcam or replay
  -> MediaPipe/replay hand tracking
  -> normalized hand landmarks
  -> deterministic pose facts
  -> stable pose events and short combo buffer
  -> control grammar with cooldowns/suppression
  -> dry-run, Hyprland, and uinput action targets
  -> preview status and JSONL logs
```

This separates the class-demo control path from the research recognizer path. Learned models can still be trained, replayed, and evaluated, but they do not trigger desktop actions.

## Testing / Verification

Run the standard checks:

```bash
uv run ruff check .
uv run pytest
```

Useful CLI smoke checks:

```bash
uv run airdesk --help
uv run airdesk control run --help
uv run airdesk gesture --help
uv run airdesk cursor run --help
```

Optional environment checks:

```bash
uv run airdesk doctor
uv run airdesk camera list
uv run airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG
uv run airdesk profile validate configs/profiles/study-safe.toml
```

## Limitations and Known Issues

- AirDesk has not been validated with accessibility populations. It is designed with situational and temporary input constraints in mind and may be relevant to broader accessibility use cases, but the repo should not be read as evidence of validated accessibility benefit.
- The current live path is tuned for Caden's Hyprland/Linux setup. Real execution depends on `hyprctl` and `/dev/uinput`.
- Webcam hand tracking is sensitive to lighting, camera angle, occlusion, and hand pose ambiguity.
- Learned TCN/IPN/DTW models are useful for evaluation and future work, but the current checkpoints are not safe enough for live global desktop actions.
- Raw recordings, logs, models, and public-dataset imports are ignored under `data/` and are not required for the core grader demo.
- Internal planning docs under `dev/active/` and `dev/archive/` are tracked for provenance. For a clean grading submission, prefer a generated export/zip or submission branch that keeps this README, source, configs, tests, scripts, and selected study docs while excluding internal agent/session planning docs and local artifacts.

## Grader Notes

For a fast grading pass, start with `airdesk control run` and the safe dry-run command above. That is the current implemented demo surface. The older `airdesk run`, `airdesk cursor run`, and `airdesk gesture ...` commands remain available for compatibility and diagnostics, but they are not the primary live-control story.
