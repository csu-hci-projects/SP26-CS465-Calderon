# AirDesk Dynamic Gesture Recognition Research

## Purpose

This note answers the Sprint 3 question:

> What recognition architecture should AirDesk use for dynamic gestures such as swipes, wrist flicks, and "conducting" motions?

The important distinction is that AirDesk is not solving isolated clip classification. It is solving continuous OS-level input:

- hands may be visible while doing non-command motion,
- the user may reposition, think, talk, or reach for the keyboard,
- gestures may chain together,
- gesture boundaries are ambiguous,
- the system must reject almost everything,
- and false activation is worse than a missed optional command.

Therefore the core problem is **gesture spotting plus intent detection**, not simply LSTM-vs-TCN-vs-Transformer classification.

## Bottom Line

Do not make Sprint 3's answer "train an LSTM."

Recommended architecture:

```text
tracking
  -> normalized landmark features
  -> smoothing / quality gates
  -> intent gate / gesture phrase state machine
  -> candidate proposal
  -> recognizer or ranker
  -> policy / profile / safety checks
  -> action
```

The first serious AirDesk dynamic recognizer should be a **gesture phrase recognizer**:

- explicit arming gesture or mode,
- short rolling history,
- background/non-gesture rejection,
- phase-aware recognition,
- release/commit semantics,
- cooldown and cancellation,
- replayable logs.

Then compare recognizers behind that same phrase interface:

1. **Rule + features baseline** for immediate inspectability.
2. **Template / DTW baseline** for personalized wrist flicks and conductor-like motions.
3. **Small causal TCN** as the first learned model if enough labeled continuous data exists.
4. **LSTM/GRU baseline** for comparison because it is familiar and lightweight.
5. **ST-GCN / graph transformer** only after AirDesk has a larger labeled dataset and a stable training workflow.

## Why LSTM Alone Is Not Enough

LSTMs can work for gesture clips, especially personal datasets, but they do not automatically solve:

- onset detection,
- offset detection,
- rejection of non-gesture motion,
- chained gestures,
- partially captured gestures,
- aborted gestures,
- "I was just moving my hand" cases,
- or commit timing.

This matches Caden's prior LSTM experience. The rolling-buffer headache is not incidental; it is the primary interaction problem.

A model can be part of the answer, but AirDesk needs a stateful intent layer around it.

## What VR / XR Systems Suggest

### Apple Vision Pro

Apple's spatial input design is not a large arbitrary gesture vocabulary. It combines:

- eye targeting,
- subtle hand confirmation,
- pinch and drag,
- pinch/flick scrolling,
- palm-facing system gestures,
- direct touch for nearby content,
- keyboard/trackpad/voice as complementary modalities.

Design lessons for AirDesk:

- Use subtle low-fatigue gestures.
- Combine a gesture with context or target state.
- Prefer familiar patterns over abstract poses.
- Give immediate feedback.
- Avoid custom gestures that conflict with common movements.
- Preserve keyboard/mouse as complementary inputs.

AirDesk does not have eye tracking, but it can use analogs:

- active profile,
- focused Hyprland window/workspace,
- explicit clutch mode,
- hand pose state,
- pointer/keyboard activity,
- and optional future gaze/depth context.

### Meta Quest

Meta's Direct Touch move is also instructive. The system started with abstract pinch-based interaction, then added direct finger tapping/swiping on panels because abstract air pinches are less intuitive for many UI tasks.

Design lessons for AirDesk:

- Raw gesture vocabularies are brittle even on headset hardware.
- Direct manipulation and contextual UI affordances reduce ambiguity.
- A future AirDesk overlay/control surface may be easier to make reliable than many global abstract gestures.

### Ultraleap

Ultraleap's hand-tracking guidance strongly favors tangible, physical interactions and says abstract poses/gestures should be used sparingly.

Design lessons for AirDesk:

- Keep the command vocabulary small.
- Keep exertion low.
- Design around an interaction zone.
- Signal tracking loss and state clearly.
- Prefer physical metaphors when possible.

### Google Soli / Microgestures

Soli work is relevant to the "conducting" feeling because it targets small, fine-grained dynamic gestures. The important caveat is that radar directly captures motion signatures that a normal webcam may not recover as cleanly.

Design lessons for AirDesk:

- Small wrist/finger gestures are a good interaction direction.
- Velocity and temporal shape matter.
- Personal calibration can improve recognition.
- Gesture spotting still remains a separate hard problem.

## Model Family Comparison

### Rule-Based Temporal Recognizers

Best for:

- first implementation,
- safety-critical gating,
- interpretable failure analysis,
- limited data,
- unit tests.

Weaknesses:

- brittle if used as the whole system,
- hard to scale to many gestures,
- thresholds drift across camera placement and users.

AirDesk role:

- baseline,
- candidate proposal,
- safety/quality gates,
- explainable logging.

### Template Matching / DTW

Best for:

- personalized conductor-like gestures,
- low-data setups,
- variable speed gestures,
- quick calibration,
- comparing whole trajectory shapes.

Weaknesses:

- still needs spotting/onset,
- can be sensitive to feature normalization,
- may not generalize across users without per-user templates.

AirDesk role:

- strong near-term fit for Caden's personal gesture vocabulary.
- likely better than LSTM for early "record my flick once or a few times" workflows.

### LSTM / GRU

Best for:

- lightweight sequence classification,
- personal models with moderate data,
- known fixed vocabularies,
- comparison against prior work.

Weaknesses:

- sliding windows can fire late, early, or repeatedly,
- output confidence may be overconfident on unknown motion,
- clean clip accuracy can hide poor continuous behavior,
- online segmentation still needs a separate design.

AirDesk role:

- baseline learned model, not the main architecture.

### TCN

Best for:

- causal streaming sequence modeling,
- framewise phase/state prediction,
- lower-latency inference than many recurrent designs,
- easier batching/training than RNNs,
- segmentation-style outputs.

Weaknesses:

- fixed receptive field must be designed,
- still needs background/negative data,
- less intuitive than DTW for one-off personalization.

AirDesk role:

- best first learned model candidate once logs exist.
- train to output frame-level states:
  - background,
  - preparation,
  - stroke_left,
  - stroke_right,
  - release,
  - cooldown.

### Transformer

Best for:

- larger datasets,
- long-range dependencies,
- multimodal context,
- attention over many joints/features.

Weaknesses:

- data hungry,
- easier to overbuild,
- may add latency/complexity,
- not automatically better for tiny personal datasets.

AirDesk role:

- later research comparison, not Sprint 3 implementation.

### ST-GCN / Graph Models

Best for:

- skeleton data,
- modeling hand joint topology,
- larger gesture datasets,
- gestures where hand shape and motion both matter.

Weaknesses:

- implementation/training complexity,
- likely overkill before AirDesk has labels,
- still does not solve intent by itself.

AirDesk role:

- serious future ML path after logging/labeling.
- more promising than plain LSTM for mature skeleton-based recognition.

### HMM / Probabilistic Spotting

Best for:

- explicit onset/offset modeling,
- non-gesture rejection,
- low-data sequential modeling,
- combining learned encoders with interpretable temporal states.

Weaknesses:

- old-school implementation complexity,
- feature engineering matters,
- not as fashionable as end-to-end deep learning.

AirDesk role:

- conceptually important.
- a hybrid "encoder + HMM/threshold" approach is attractive for continuous command gestures.

## Feature Strategy

Avoid recognizing dynamic gestures from raw wrist translation alone. That encourages big arm motion.

For conductor-like gestures, AirDesk should compute features in a hand-centered coordinate system:

- palm center from wrist/index MCP/pinky MCP,
- palm velocity and acceleration,
- index tip relative to palm,
- thumb-index pinch distance and derivative,
- palm normal / roll estimate,
- hand bounding-box size as distance proxy,
- finger curl/open features,
- short trajectory shape,
- confidence and tracking continuity,
- tracking loss flags.

Potential gesture phrase examples:

### Pinch + Wrist Flick

```text
pinch_down -> armed
short palm/index velocity impulse left/right/up/down
pinch_release -> commit
```

Why this is good:

- small movement,
- clear start/end,
- less vulnerable to random hand motion,
- maps well to Apple-style pinch/flick,
- avoids dragging the whole arm.

### Open Palm + Beat

```text
open_palm_hold -> listening
short downward/upward palm impulse -> command proposal
return to low motion -> commit
```

Why this is good:

- feels like conducting,
- but needs stronger false-positive protection because open hands move naturally.

### Palm Tilt

```text
listening
palm roll/yaw crosses threshold and returns
commit on return
```

Why this is good:

- small wrist motion,
- less screen-space dependent,
- potentially robust if MediaPipe landmarks preserve hand orientation.

Risk:

- webcam viewpoint and occlusion may make orientation noisy.

## Data Strategy

AirDesk should collect data for spotting, not just classification.

Record sessions containing:

- intended flick left/right/up/down,
- intended slow swipes,
- intended palm tilts,
- intended pinch holds/releases,
- aborted gestures,
- chained gestures,
- normal desk motion,
- talking-with-hands motion,
- reaching for keyboard/mouse,
- hand entering/leaving frame,
- idle hand visible,
- no hand visible.

Label at two levels:

### Event Labels

- gesture name,
- intended command,
- start timestamp,
- end timestamp,
- commit timestamp,
- success/failure notes.

### Phase Labels

- background,
- preparation,
- stroke,
- hold,
- release,
- cooldown,
- aborted.

The phase labels are what make TCN/HMM/CTC-style approaches practical later.

## Evaluation Strategy

Do not optimize only isolated accuracy.

Primary metrics:

- false activations per minute/hour,
- missed intended gestures,
- command latency,
- segmental F1,
- word error rate / edit distance for gesture streams,
- repeated-fire count,
- cancellation success,
- user-rated control/confidence,
- fatigue/discomfort.

Model comparisons should run on the same continuous sessions:

- rule baseline,
- DTW/template,
- LSTM/GRU,
- TCN,
- later ST-GCN/Transformer.

## Recommended AirDesk Roadmap

### Sprint 3

- Add runtime event logging.
- Record deliberate and negative continuous sessions.
- Add a phrase-recognizer abstraction.
- Implement rule and DTW baselines for pinch/flick and palm impulse gestures.
- Keep execution dry-run while evaluating false activations.

### Sprint 4

- Add labeling tooling.
- Export training datasets from JSONL.
- Train LSTM/GRU and TCN baselines.
- Compare against rule/DTW on continuous sessions.
- Pick one learned model only if it beats baselines on false activations and latency.

### Later

- Explore ST-GCN / graph transformer if dataset size justifies it.
- Add context signals:
  - Hyprland active window,
  - keyboard/mouse activity,
  - hand distance/zone,
  - optional depth/Kinect,
  - optional gaze if hardware ever supports it.

## Current Recommendation

For AirDesk's near-term goal, the best recognition architecture is:

> **Intent-gated gesture phrases with rule/DTW baselines now, then a causal TCN trained on phase-labeled continuous data.**

LSTM should be included as a baseline because it is easy to compare and Caden has prior experience with it, but it should not be the primary bet.

The likely strongest mature path is not a single classifier. It is a hybrid:

```text
state machine / clutch
  + quality gates
  + phrase segmentation
  + DTW or TCN recognizer
  + background rejection
  + profile policy
```

That architecture best matches the "conducting a choir for your computer" goal: small expressive gestures, clear phrasing, strong intent, and minimal accidental commands.

## Sources

- Apple, "Design for spatial input", WWDC23: https://developer.apple.com/videos/play/wwdc2023/10073/
- Apple Support, "Use gestures with Apple Vision Pro": https://support.apple.com/en-lk/117741
- Apple, "Design great visionOS apps", WWDC24: https://developer.apple.com/videos/play/wwdc2024/10086/
- Meta, "Use Your Fingers (Not Controllers) to Swipe Through the VR Interface on Meta Quest": https://about.fb.com/news/2023/02/meta-quest-direct-touch-use-your-fingers-in-vr/
- Ultraleap, "6 VR Design Principles for Hand Tracking": https://docs.ultraleap.com/ultralab/hand-tracking-vr-design.html
- Ultraleap, "Interactions overview": https://docs.ultraleap.com/xr-guidelines/Interactions/interactions_overview.html
- Emporio et al., "Continuous hand gesture recognition: Benchmarks and methods", CVIU 2025: https://www.sciencedirect.com/science/article/pii/S1077314225001584
- Benitez-Garcia et al., "IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition": https://arxiv.org/abs/2005.02134
- IPN Hand dataset page: https://gibranbenitez.github.io/IPN_Hand/
- "Continuous Hand Gesture Spotting Through Deep Sequential Encoding and Probabilistic Time-Series Modeling": https://openreview.net/pdf?id=JfhxzTASJr
- Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection": https://arxiv.org/abs/1611.05267
- Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling": https://arxiv.org/abs/1803.01271
- Li et al., "Spatial temporal graph convolutional networks for skeleton-based dynamic hand gesture recognition": https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-019-0476-x
- Han et al., "Spatio-Temporal Dynamic Attention Graph Convolutional Network Based on Skeleton Gesture Recognition": https://www.mdpi.com/2079-9292/13/18/3733
- Wang et al., "Interacting with Soli: Exploring Fine-Grained Dynamic Gesture Recognition in the Radio-Frequency Spectrum": https://cse.buffalo.edu/faculty/dimitrio/courses/cse709_s17/material/papers/deep_soli.pdf
- Casiez et al., "1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems": https://citeseerx.ist.psu.edu/document?doi=8802d2f30dc47e219d4e8da212a7a32e6fce428d&repid=rep1&type=pdf
