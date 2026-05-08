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

Do not make Sprint 3's answer "train an LSTM" or "compare every model family."

Original Sprint 3 bet:

> **Intent-gated gesture phrases plus a small causal TCN trained on phase-labeled continuous landmark features.**

May 2026 update after Sprint 4 evidence and a deeper continuous-gesture research pass:

> AirDesk should now treat the causal TCN window classifier as a scaffold, not the destination. The next recognizer direction is **continuous gesture spotting**: position-invariant skeleton features, stream/phase labels, event decoding, and a small hybrid model that can later grow from TCN to graph/transformer memory if the data supports it.

Recommended architecture:

```text
tracking
  -> per-hand wrist/palm-centered, hand-scale-normalized landmark + motion features
  -> smoothing / quality gates
  -> gesture energy / candidate proposal
  -> causal stream model or template ranker
  -> event decoder with hysteresis, peak detection, recovery, cooldown
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

May 2026 two-hand update: combo and alternating-hand gestures should not require one hand to leave frame before the other hand becomes active. AirDesk's next data/model pass needs two-hand tracking and per-hand stream processing. `--max-num-hands 2` alone is not enough: feature export, DTW/TCN scoring, event decoding, and labels must treat each visible hand as its own temporal stream, then merge decoded events across hands. The one-hand structured combo takes from `sprint4-gpu-swipes-002-structured` were deleted and should not be used for combo training.

May 2026 shared-model update: Caden's intuition was right that the model should watch each hand separately, but the current best bet is a shared per-hand TCN checkpoint, not separate tracker-slot models. Run the same TCN over `hand-0`, `hand-1`, etc. independently, decode events per hand, then merge/cool down globally. This preserves data efficiency and avoids treating MediaPipe tracker slots as stable physical left/right hands. The first 003-to-004 replay check matched 27/48 decoded events with 11 false activations and 4 repeated fires, so it is useful evidence for the architecture but not live-control readiness.

Weak label note: chart labels are prompt-timing labels, not hand-specific truth. For two-hand TCN manifests, use motion-gated target assignment so a resting visible hand is not labeled as the active stroke. The gate should use recent motion energy rather than raw left/right dx sign because mirrored preview and raw camera coordinate conventions can flip or blur the sign evidence across sessions.

May 2026 Recognition V2 update: after Caden's deep research report and live TCN tests, stop treating the current TCN scaffold as the recognizer architecture. It is still useful evidence, but AirDesk now needs a cleaner continuous-spotting architecture: per-hand normalized streams, motion activity proposal, scorer/model adapter, event decoder, command queue, and mode/profile safety policy. Caden saw live `dx > 0.50` while TCN stroke probabilities stayed flat, so the failure is not only a motion threshold. The plan review is complete, and the first deterministic per-hand motion-event baseline now exists through `airdesk gesture spot-motion` and `airdesk gesture evaluate-motion`. Initial replay evidence says the baseline is a diagnostic, not a live recognizer: motion-only spotting exposes raw camera direction issues, false activations on negatives, and weak-left misses.

The model plan is:

1. Use **rule + feature gates** for immediate safety and debugging.
2. Use **DTW/template matching** as a low-data fallback and calibration tool.
3. Train one primary learned model as a **small causal stream model**, starting with TCN because it is easier to debug on small data.
4. Train toward **frame/phase/event spotting**, not only whole-window `background/swipe_left/swipe_right` classification.
5. Defer **LSTM/GRU** unless the TCN/spotting path fails or a later comparison is explicitly worth the time.
6. Defer **ST-GCN / graph transformer** until AirDesk has enough labeled skeleton streams, but keep it as the likely next family if TCN capacity becomes the blocker.

## May 2026 Research Update: Continuous Spotting

Caden's live TCN preview exposed the same failure mode seen in prior LSTM work: fixed rolling windows can miss the beginning or end of a gesture, struggle with fast consecutive gestures, and confuse the recovery/reset motion with the next command. The literature frames this as the core continuous-recognition problem, not a bug in one model family.

Useful terms for future search and paper wording:

- continuous hand gesture recognition
- online gesture recognition
- gesture spotting
- temporal action segmentation
- online action detection
- early gesture detection
- non-gesture / garbage / threshold model
- CTC for unsegmented gesture streams
- skeleton-based online gesture recognition

Key research takeaways:

- **Continuous recognition is not isolated clip classification.** Benchmarks for continuous gestures evaluate segmentation quality, latency, and false detections, not just trimmed-clip accuracy.
- **Gesture phases matter.** Dynamic gestures often include preparation, nucleus/stroke, and retraction/recovery. The nucleus is the most discriminative part; reset/recovery can be misleading.
- **Event decoding is a separate layer.** Even strong models usually need thresholds, finite-state machines, non-maximum suppression, confidence peaks, or candidate filtering to become reliable interaction events.
- **Background is not just another class.** Older HMM work used non-gesture, garbage, or threshold models; newer systems still need explicit false-positive control.
- **Sliding windows remain common but are fragile.** Newer work tries to compensate with memory across windows, CTC-style alignment, or joint detection/classification rather than assuming each fixed window contains one complete gesture.
- **Skeleton/landmark input is the right AirDesk level.** SHREC-style work and newer online hand-gesture papers treat hand skeletons as the main representation for real-time interaction. AirDesk should keep MediaPipe as backend zero, not the project identity.

Relevant source anchors:

- Continuous hand gesture recognition survey: https://www.sciencedirect.com/science/article/pii/S1077314225001584
- NVIDIA online dynamic gesture detection with CTC and gesture phases: https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S18-21.pdf
- SHREC 2021 skeleton-based online hand gesture benchmark: https://www.maghoumi.com/wp-content/uploads/2021/12/SHREC2021-Skeleton-based-hand-gesture-recognition-in-the-wild.pdf
- Controlled CTC for early gesture detection in untrimmed streams: https://www.sciencedirect.com/science/article/pii/S0031320324004849
- HMM-DNN forward spotting and non-gesture thresholding: https://www.mdpi.com/2227-9709/10/1/1
- Continual Graph Transformer for online hand gestures: https://arxiv.org/abs/2502.14939
- HMATr / OMG-Bench for rapid continuous micro-gesture recognition: https://arxiv.org/abs/2512.16727

### What A Transformer Would Mean For AirDesk

A transformer should not be a raw-webcam model for AirDesk. If tried, it should consume normalized hand skeleton streams:

```text
MediaPipe / tracker landmarks
  -> wrist/palm-centered joints
  -> hand-scale normalization
  -> velocities, acceleration, displacement, direction consistency
  -> causal transformer or graph-transformer temporal encoder
  -> per-frame phase probabilities and/or event queries
  -> event decoder
```

A plain transformer over fixed windows would still have the same boundary problem as the current TCN. The interesting variants are:

- **causal transformer** over recent normalized feature tokens,
- **graph + transformer** where graph layers model the hand skeleton and attention models time,
- **memory-augmented transformer** where previous-window features are cached so a gesture is not lost at the window boundary,
- **query/event transformer** where learnable queries propose gesture instances and classes.

This is promising later, but it is not the clean next step. AirDesk's current bottleneck is representation, labels, and event decoding, not model capacity.

### Hybrid Recognizer Direction

The near-term recognizer should combine the useful parts of DTW, TCN, and gesture-spotting literature:

```text
streamed normalized features
  -> motion-energy / quality gate proposes candidate activity
  -> small causal TCN labels phase probabilities
  -> optional DTW/template ranker scores candidate trajectories
  -> event decoder fires one command at the confidence peak
  -> cooldown/recovery prevents repeated fires
```

This gives AirDesk a path to better consecutive swipes without pretending there is a magic windowless recognizer:

- activity gates reduce background false positives,
- phase labels distinguish stroke from reset,
- normalized features reduce camera-position/distance dependence,
- event decoding handles repeated fires,
- DTW remains useful for low-data personalization,
- TCN remains useful for learned stream state,
- graph/transformer memory remains a later upgrade if data grows.

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

- low-data fallback and calibration tool.
- useful for validating feature design before training.
- not the main long-term model bet unless it clearly outperforms the TCN in practice.

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

- deferred learned baseline.
- do not spend Sprint 4 time here unless the causal TCN path is blocked or performs poorly.

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

- primary learned model bet once logs exist.
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
- cross-window memory,
- multimodal context,
- attention over many joints/features.

Weaknesses:

- data hungry,
- easier to overbuild,
- may add latency/complexity,
- not automatically better for tiny personal datasets,
- still needs spotting, background rejection, and event decoding.

AirDesk role:

- later research comparison or upgrade path, not the next implementation by itself.
- The interesting version is a causal or memory-augmented skeleton transformer, not a fixed-window classifier.

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
- likely the right spatial front-end if AirDesk later builds a transformer-style recognizer.

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
- Use the idea even if the implementation is not HMM: non-gesture competition, adaptive thresholds, and forward spotting map directly onto AirDesk's event decoder.

## Feature Strategy

Avoid recognizing dynamic gestures from raw wrist translation alone. That encourages big arm motion.

For conductor-like gestures, AirDesk should compute features in a hand-centered coordinate system and avoid making absolute frame position the gesture identity. Camera placement, distance, and user posture should change as little as possible about the recognized command.

- palm center from wrist/index MCP/pinky MCP,
- wrist/palm-centered landmark coordinates,
- hand-scale-normalized landmark coordinates and displacements,
- palm velocity and acceleration,
- index tip relative to palm,
- thumb-index pinch distance and derivative,
- palm normal / roll estimate,
- hand bounding-box size as distance proxy,
- finger curl/open features,
- signed horizontal/vertical displacement over short causal windows,
- peak velocity,
- direction consistency,
- short trajectory shape,
- confidence and tracking continuity,
- tracking loss flags.

Absolute `palm_x`, `palm_y`, and `palm_z` can remain useful diagnostics, but learned models should have a preset that excludes them or downweights them. The model should learn "the hand moved left relative to itself and its recent path," not "Caden's hand started on the right side of this webcam frame."

## Updated Next Implementation Direction

Do this before wiring any learned swipe recognizer to desktop actions:

1. Add a position-invariant TCN feature preset that excludes absolute palm position and emphasizes normalized motion.
2. Add stream/phase labeling support for `background`, `stroke_left`, `stroke_right`, and `recovery/reset`.
3. Add an event decoder over model probabilities: hysteresis, peak confidence, minimum separation, recovery, and repeated-fire suppression.
4. Add a collection/labeling helper for ordered continuous streams where Caden can provide a sequence like `R L R R L L` without exact timestamps.
5. Compare DTW, current TCN, and the hybrid event decoder on isolated holdout plus chained sessions.
6. Only then decide whether a small graph/transformer model is justified.

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

If a later comparison becomes necessary, run it on the same continuous sessions:

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
- Implement rule scaffolding and DTW fallback for pinch/flick and palm impulse gestures.
- Keep execution dry-run while evaluating false activations.

### Sprint 4

- Add labeling tooling.
- Export training datasets from JSONL.
- Train the first causal TCN recognizer.
- Use rule/DTW as sanity checks and fallbacks, not as a large model bake-off.
- Defer LSTM/GRU unless the TCN path fails.

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

> **Intent-gated gesture phrases with rule/DTW scaffolding now, then a causal TCN trained on phase-labeled continuous data.**

LSTM/GRU should be deferred. They are available as later baselines if the TCN disappoints, but they are not the current bet.

The likely strongest mature path is not a single classifier. It is a hybrid:

```text
state machine / clutch
  + quality gates
  + phrase segmentation
  + causal TCN recognizer
  + rule/DTW fallback
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
