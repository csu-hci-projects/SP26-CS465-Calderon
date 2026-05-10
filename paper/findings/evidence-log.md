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

Recognizer motivation:

- Prior LSTM/sliding-window smart-home gesture work exposed a timing problem that AirDesk should name in the paper.
- Fixed windows can capture partial gestures, reset motion, adjacent gestures, or background motion.
- A longer window improves context but increases latency and can merge repeated commands.
- A shorter window improves responsiveness but misses gesture boundaries.
- TCN v2 is motivated by treating windows as causal compute context, while event boundaries and decoder logic define commands.

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

TCN v2 quick smoke:

- TCN v2 manifest/model/evaluation surface exists for replay/evaluation.
- Uses frame-level evidence heads: intentional motion, left/right stroke, start, and end.
- Old-data smoke trained cleanly, but event-level replay was weak.
- Permissive replay pass matched 1/16 intended swipes.
- Produced 2 false activations.
- Applying the same old-data model to a chained session matched 0/10.
- Interpretation: useful architecture direction and regression harness, but not pilot-ready without fresh continuous training data and better event decoding.

## Evidence Needed Next

- Fresh 60-90 second chained recording with multiple left/right swipes and normal motion between them.
- Gated DTW, motion-baseline, and TCN v2 evaluation on that fresh recording.
- A recognizer decision for the pilot based on event-level behavior, not window accuracy alone.
- Caden pilot notes.
- Roommate pilot notes.
- Keyboard/mouse baseline timings for the same task set.
- At least one table comparing task success, misses, false activations, and subjective notes.
