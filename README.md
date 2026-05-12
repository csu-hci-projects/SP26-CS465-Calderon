# AirDesk

AirDesk is a CS465 HCI / 3DUI project about using webcam-based mid-air hand gestures as a secondary desktop control layer for Hyprland Linux. The project is not trying to replace the keyboard and mouse. It focuses on moments where ordinary input is temporarily inconvenient, unavailable, dirty, painful, or physically costly.

The final prototype uses MediaPipe hand landmarks and deterministic landmark math to control a small desktop vocabulary: pointer movement, click, drag/select, scroll, right click, workspace switching, moving windows between workspaces, opening a launcher, and closing a window with a deliberate combo.

## Grader Note

I worked on this project solo as a group of one. If anything is unclear or you have questions while grading, feel free to message me on Teams or email me at cadencc@colostate.edu.

## Start Here

For grading or review:

1. Read the final paper PDF from the submitted zip, or the paper source at `paper/latex-source/main.tex`.
2. Read `Final_Report.pdf` in this repository.
3. Check the literature PDFs in `paper/PDFs-LiteratureSurvey/`.
4. Watch the presentation and short demo videos linked below.
5. Run the safe replay smoke test below.
6. If reviewing on a Linux laptop with webcam support, run the live practice-mode demo.

Project links:

- GitHub: https://github.com/csu-hci-projects/SP26-CS465-Calderon
- Overleaf: https://www.overleaf.com/read/ttdzxcqmcknp#c89490
- Project / presentation video: https://youtu.be/3thM46mDVHU
- Short demo video: https://youtu.be/k68HIlHOwME

## What Is Implemented

The main live-control path is:

```text
webcam or replay
  -> MediaPipe/replay hand landmarks
  -> deterministic landmark features
  -> stable gesture events
  -> action rules
  -> practice logs, Hyprland commands, or Linux pointer events
```

Implemented controls:

| Gesture | Behavior |
| --- | --- |
| Open or relaxed hand | Move the pointer. |
| Index pinch tap | Left click. |
| Index pinch hold or drag | Hold left button for selection/drag. |
| Middle pinch tap | Right click on clean release. |
| Middle pinch hold plus vertical movement | Scroll while the cursor is locked. |
| Stable fist, then up/down motion | Switch workspaces. |
| Stable fist, then left/right motion | Move the active or targeted window between workspaces. |
| Open palm -> sideways palm | Open the configured launcher. |
| Open palm -> fist -> open palm | Close the active window. |

Earlier DTW, TCN, IPN, and motion-recognition code remains in the repository as research infrastructure because it shaped the project direction. It is not the final live-control path. The final demo uses deterministic MediaPipe landmark logic because it was more reliable for live OS control within the course timeline.

## Setup

AirDesk uses Python 3.12 and `uv`.

```bash
uv sync --dev
uv run airdesk --help
```

For live webcam tracking:

```bash
uv sync --dev --extra live
```

For optional ML/research commands:

```bash
uv sync --dev --extra live --extra ml
```

## Safe Replay Smoke Test

This command does not require a webcam, Hyprland, or live desktop control. It checks that the CLI and deterministic control runtime can read a recorded frame and write an event log.

```bash
uv run airdesk control run \
  --backend replay \
  --recording tests/fixtures/replay-one-frame.jsonl \
  --events-out /tmp/airdesk-control-replay.jsonl \
  --max-frames 1 \
  --no-show
```

## Live Practice Demo

This is the recommended live demo command. It opens the webcam preview and records what the system would do. It does not click, scroll, move the pointer, or send Hyprland commands unless execution flags are added.

```bash
uv run airdesk control run \
  --backend mediapipe \
  --device /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --fourcc MJPG \
  --events-out /tmp/airdesk-control-demo.jsonl \
  --show
```

Preview controls:

- `p`: pause/resume actions while tracking continues.
- `q` or `esc`: exit the preview.

## Optional Live Execution

Real execution is Linux/Hyprland/uinput-specific. Use it only on a machine where `hyprctl` works and `/dev/uinput` is writable by the user.

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
  --events-out /tmp/airdesk-control-execute.jsonl \
  --show
```

`--execute` enables guarded Hyprland commands. `--pointer-execute` enables real pointer movement, click, drag, and scroll through `/dev/uinput`.

## Pilot Summary

The pilot used a Google Docs workspace task that required workspace navigation, cursor movement, clicking, scrolling, text selection/copying, and pasting. Two people participated: the author and one roommate participant.

Normal desktop condition:

- Keyboard/mouse mean: 22.0 seconds.
- AirDesk hand-control mean: 41.5 seconds.
- Result: keyboard/mouse was clearly faster under ideal desk conditions.

Dirty-hands condition:

- Participants started with hands covered in olive oil, flour, and honey.
- Keyboard/mouse runs required washing hands first.
- AirDesk runs skipped washing and used gestures directly.
- Keyboard/mouse mean: 43.8 seconds.
- AirDesk hand-control mean: 40.0 seconds.
- Result: AirDesk became competitive and slightly faster because it avoided the cleanup interruption.

This is a small formative pilot. It supports the situational-impairment framing, not a broad claim that mid-air gestures replace keyboard and mouse input.

## Repo Map

- `src/airdesk/control/` - deterministic live-control poses, debouncing, combos, grammar, and runtime.
- `src/airdesk/actions/` - practice targets, guarded Hyprland dispatch, cursor targets, and uinput pointer injection.
- `src/airdesk/tracking/` - MediaPipe and replay tracking backends.
- `src/airdesk/gestures/` - DTW, motion, learned-filter, and event-decoder research code.
- `src/airdesk/ml/` and `src/airdesk/public_datasets/` - optional TCN/IPN training and evaluation infrastructure.
- `tests/` - unit and CLI coverage.
- `configs/profiles/` - profile examples from runtime work.
- `Final_Report.pdf` - final paper PDF.
- `paper/latex-source/` - ACM LaTeX source and template files.
- `paper/PDFs-LiteratureSurvey/` - literature PDFs used in the related-work section.
- `studies/pilot-0.md` - small pilot task summary and timings.

## Verification

Standard checks:

```bash
uv run ruff check .
uv run pytest
```

Useful CLI checks:

```bash
uv run airdesk --help
uv run airdesk control run --help
uv run airdesk doctor
```

The final PDF is included as `Final_Report.pdf`. To rebuild locally, compile `paper/latex-source/main.tex` with a LaTeX distribution such as `pdflatex` or `latexmk`, or use the Overleaf link above.

## Known Limitations

- The pilot has only two participants, including the author.
- The live prototype is tuned for a Linux/Hyprland laptop setup.
- Webcam tracking depends heavily on lighting, distance from the camera, hand angle, and occlusion.
- The deterministic landmark thresholds are understandable and tunable, but not robust across every camera setup.
- The project is accessibility-motivated, but it is not validated with a disability population.
- The learned recognizer work is future/research infrastructure, not the final live-control recognizer.
