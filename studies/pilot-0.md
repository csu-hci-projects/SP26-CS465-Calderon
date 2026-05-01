# AirDesk Pilot 0 Protocol

## Purpose

Pilot 0 is a Caden-only dry-run-first check of Sprint 3 command mode before any broader CS465 study work.

The goal is not to prove that gestures replace keyboard and mouse. The goal is to decide whether AirDesk's clutch-based mid-air command layer is observable, safe, and reliable enough for a narrow desktop-control evaluation.

## Research Question

Can a small open-palm-clutched gesture vocabulary support common Hyprland command actions with acceptable false activation, latency, and perceived control during a short local pilot?

## Conditions

- Baseline keyboard/mouse.
- AirDesk live dry-run.
- AirDesk guarded execute mode only if dry-run logs show no concerning false activations.

## Task Set

- Switch workspace left.
- Switch workspace right.
- Move focus left.
- Move focus right.
- Cancel or recover from listening mode.
- Pause/resume the runtime using the preview `p` key.

Do not include destructive actions in Pilot 0.

## AirDesk Setup

Use dry-run first:

```bash
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --dry-run --show --events-out data/studies/pilot-0/live-dry-run.jsonl
```

If dry-run behavior is acceptable, guarded execute mode can be tried locally:

```bash
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --execute --allow-profile-execute --show --events-out data/studies/pilot-0/live-execute.jsonl
```

Use `--pause-on-start` when setting up the camera or when testing pause behavior.

## Procedure

1. Run a 2-minute normal desk motion dry-run with hands occasionally entering frame.
2. Record false activations and any unexpected listening/action events.
3. Perform five repetitions of each task in dry-run.
4. Rest for at least one minute.
5. Repeat the task set with keyboard/mouse baseline timing.
6. Only if dry-run is stable, perform guarded execute mode for workspace and focus commands.

## Measures

- Task success or failure.
- False activations per minute.
- Missed intended gestures.
- Repeated-fire events.
- Gesture-to-action latency from the event log.
- Tracking loss or unstable landmark notes.
- Pause/resume success.
- Fatigue, discomfort, frustration, and perceived control notes.

## Abort Rules

- Stop immediately if real execution controls the wrong workspace or focus target twice.
- Stop if pause does not suppress actions.
- Stop if false activations occur during normal desk motion.
- Stop if wrist, shoulder, or arm fatigue becomes noticeable.

## Data Files

- Runtime event logs in ignored `data/studies/pilot-0/`.
- Optional replay recordings in ignored `data/recordings/`.
- Notes in this file or a copied local notes file.

Raw video is not collected by default.

## Results Notes

Pilot not yet run in this repository session. Caden should fill in:

- camera settings:
- lighting:
- hand distance:
- dry-run false activations:
- dry-run missed gestures:
- execute-mode result, if attempted:
- fatigue/discomfort:
- design changes before Sprint 4:
