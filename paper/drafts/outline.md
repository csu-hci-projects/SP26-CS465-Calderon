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
4. Design Goals and Research Questions
5. System Overview
6. Interaction Design
7. Implementation and Reproducibility
8. Pilot Method
9. Prototype Evidence and Findings
10. Discussion
11. Limitations
12. Future Work
13. Conclusion

## Why This Order

This order fits a system paper with pilot evidence.

The early sections explain why the problem matters and what prior work says. The
middle sections describe the design choices and prototype before asking the
reader to interpret results. The later sections report evidence, then honestly
separate what the current prototype supports from what still needs future work.

The ACM template does not require this specific section order. It mainly defines
the formatting shell: document class, title, abstract, CCS concepts, keywords,
body sections, and bibliography. The course rubric is the real content guide, so
the paper needs strong state-of-the-art, methods/evaluation, prototype
description, reproducibility, limitations, future work, and conclusion sections.

## Core Argument

AirDesk should not argue that mid-air gestures are faster than keyboard shortcuts in normal desktop use.

AirDesk should argue that gestures may be valuable as a secondary command layer for moments when traditional input is inconvenient, unavailable, dirty, painful, or physically costly.

The tone should stay clear and human. The paper can be formal, but it should not
read like inflated system-paper boilerplate. Prefer direct sentences, concrete
examples, and evidence-bounded claims.

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
