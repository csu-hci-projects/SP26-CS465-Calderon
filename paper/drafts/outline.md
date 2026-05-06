# AirDesk Paper Outline

## Target Paper Type

System paper with pilot evidence.

The paper should read as:

1. Here is a real HCI problem.
2. Here is the AirDesk system built to explore it.
3. Here is what the prototype can currently do.
4. Here is the pilot evidence from Caden and one roommate.
5. Here is what worked, what failed, and what should happen next.

## Proposed Section Flow

1. Abstract
2. Introduction
3. Related Work
4. System Overview
5. Interaction Design and Gesture Vocabulary
6. Implementation
7. Pilot Method
8. Prototype Evidence and Findings
9. Discussion
10. Limitations
11. Future Work
12. Conclusion

## Core Argument

AirDesk should not argue that mid-air gestures are faster than keyboard shortcuts in normal desktop use.

AirDesk should argue that gestures may be valuable as a secondary command layer for moments when traditional input is inconvenient, unavailable, dirty, painful, or physically costly.

## Claims We Can Probably Support

- AirDesk implements a modular webcam-to-desktop gesture pipeline.
- Dry-run-first execution and replayable logs make the system safer to evaluate.
- Naive rule-based dynamic swipe recognition fails under natural desktop motion.
- Personalized DTW recognition is a more promising low-data baseline, but still needs fresh continuous-session validation.
- Small command vocabularies and explicit clutching are more defensible than broad always-on gesture control.

## Claims To Avoid

- Gestures replace keyboard and mouse.
- AirDesk is validated as an accessibility tool.
- The current recognizer is live-control reliable.
- The pilot generalizes to a broad population.
- MediaPipe is the contribution.

