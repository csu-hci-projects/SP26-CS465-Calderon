# AirDesk Pilot Summary

This file summarizes the final small pilot used in the CS465 paper. The more
complete paper-facing evidence log is [paper/findings/evidence-log.md](../paper/findings/evidence-log.md).

## Purpose

The pilot was a formative check of whether the current AirDesk prototype could
complete a realistic desktop workflow with only mid-air hand gestures.

The goal was not to prove that gestures replace keyboard and mouse. The goal was
to test the situational-impairment framing: gestures may be useful when ordinary
input is temporarily inconvenient or costly.

## Participants

- Author/developer.
- One roommate participant.

## Apparatus

- Linux laptop.
- Built-in laptop webcam at approximately 640x480 and 30 FPS.
- Good indoor lighting.
- Hyprland desktop.
- AirDesk deterministic MediaPipe landmark-control runtime.

## Task

Participants started on workspace 1. A Google Docs tab was open on workspace 8.
The task was:

1. Navigate to workspace 8.
2. Click/open the correct document.
3. Scroll to page 4.
4. Select/highlight/copy the keyword/code string.
5. Navigate back to workspace 4.
6. Click into another Google Docs tab.
7. Paste the code into the target "paste here" area.

This task was chosen because it requires workspace navigation, pointer movement,
clicking, scrolling, text selection, copy/paste behavior, and returning to a
previous context.

## Conditions

Normal condition:

- Keyboard/mouse available immediately.
- AirDesk hand control using the deterministic control runtime.

Dirty-hands condition:

- Participants started with hands covered in olive oil, flour, and honey.
- Keyboard/mouse runs required washing hands first.
- AirDesk runs skipped washing and used gestures directly.

## Results

Normal condition:

| Participant | Keyboard/mouse runs | AirDesk runs | Mean difference |
| --- | --- | --- | --- |
| Author | 22s, 20s | 41s, 39s | AirDesk 19.0s slower |
| Roommate | 24s, 22s | 44s, 42s | AirDesk 20.0s slower |
| Overall | 22.0s mean | 41.5s mean | AirDesk 19.5s slower |

Dirty-hands condition:

| Participant | Keyboard/mouse runs | AirDesk runs | Mean difference |
| --- | --- | --- | --- |
| Author | 46s, 43s | 38s, 40s | AirDesk 5.5s faster |
| Roommate | 44s, 42s | 40s, 42s | AirDesk 2.0s faster |
| Overall | 43.8s mean | 40.0s mean | AirDesk 3.8s faster |

## Qualitative Notes

- Cursor movement and pinch-to-click felt best.
- Scrolling was harder and less familiar.
- Pinch-to-click sometimes missed on the first attempt and needed a second pinch.
- Good lighting and a front-facing hand position helped tracking.
- Camera angle and hand distance mattered. Farther hands compressed landmarks
  together and made threshold-based pinches easier to trigger unintentionally.

## Interpretation

Keyboard/mouse was clearly faster in normal desktop conditions. In the
dirty-hands condition, AirDesk became competitive and slightly faster because it
avoided the cleanup interruption. This supports the paper's main framing:
mid-air gestures are useful as an optional input layer when ordinary input has a
real access cost.
