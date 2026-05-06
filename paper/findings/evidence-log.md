# AirDesk Evidence Log

This file tracks paper-usable evidence. Keep the wording cautious and tied to actual artifacts.

## Current Prototype Evidence

Source docs:

- `/home/caden/projects/AirDesk/README.md`
- `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/context.md`
- `/home/caden/projects/AirDesk/dev/active/cs465-airdesk/context-reset-prompt.md`

Current implementation supports:

- OpenCV webcam capture.
- MediaPipe Tasks Hand Landmarker backend.
- Replay backend for JSONL recordings.
- Typed normalized hand state.
- Static pose recognizers for primitives such as open palm, fist, and pinch.
- Intent-gated phrase recognizer foundation for dynamic gestures.
- Dry-run action target.
- Guarded Hyprland dispatch target.
- Runtime `--events-out` JSONL logs with session start and finish events.
- Preview-first recording collection.
- Label init/validate/add/suggest workflow.
- Deterministic feature export.
- Rule and DTW gesture evaluation.
- Cursor-mode experiment with pinch-hold movement and guarded Hyprland `movecursor`.

## Sprint 4 Swipe Batch

Artifact group:

- `data/recordings/sprint4-swipes-001`
- `data/labels/sprint4-swipes-001`
- `data/features/sprint4-swipes-001`
- `data/evaluations/sprint4-swipes-001`
- `data/evaluations/sprint4-swipes-001-dtw`
- `data/evaluations/sprint4-swipes-001-dtw-holdout`

Observed local data:

- 24 recordings total.
- 8 left-swipe positive takes.
- 8 right-swipe positive takes.
- 8 normal-desk-motion negative takes.
- Matching label and feature files exist for all 24 recordings.

## Recognition Evidence

Rule recognizer:

- 16 intended positive swipe events.
- 0 matched.
- 16 missed.
- High false activations from static-pose candidates during natural motion.
- Interpretation: rule recognition is not adequate for natural dynamic swipes.

Same-batch DTW baseline:

- 16 intended swipes.
- 16 matched.
- 0 missed.
- 18 candidates.
- 2 false activations.
- 0 repeated fires.
- Mean latency about 0.44 seconds.
- Interpretation: promising but optimistic because calibration and evaluation used the same small batch.

Deterministic DTW holdout:

- Train on takes 001-006.
- Test on takes 007-008.
- 4 intended held-out swipes.
- 2 matched.
- 2 missed.
- 0 held-out false activations.
- Right swipes matched 2/2.
- Left swipes matched 0/2.
- Interpretation: left-swipe/negative separation was not good enough.

Gated DTW holdout variant:

- Uses calibrated horizontal displacement gate.
- 4 intended held-out swipes.
- 4 matched.
- 0 missed.
- 0 held-out false activations.
- Mean latency about 0.36 seconds.
- Interpretation: promising, but tuned after viewing the holdout. Needs fresh chained continuous validation.

## Evidence Needed Next

- Fresh 60-90 second chained recording with multiple left/right swipes and normal motion between them.
- Gated DTW evaluation on that fresh recording.
- Caden pilot notes.
- Roommate pilot notes.
- Keyboard/mouse baseline timings for the same task set.
- At least one table comparing task success, misses, false activations, and subjective notes.

