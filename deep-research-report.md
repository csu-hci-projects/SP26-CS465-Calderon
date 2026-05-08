# Real-Time Continuous Hand Gesture Recognition from an RGB Webcam

## Executive Summary

The best practical architecture for your project is **not** a single monolithic classifier. It is a **multi-stage streaming system** built around entity["software","MediaPipe Hands","real-time hand landmark tracking"] or a close alternative for hand tracking, a **causal landmark-sequence model** for framewise gesture spotting, and a **decoder/state machine** that turns noisy frame predictions into clean gesture events. The strongest first build is:

**webcam → hand tracker → normalized landmark features → causal TCN sequence labeler with background + boundary heads → event decoder → command queue**.  

That recommendation follows from the core failure mode you already saw: fixed-length clip classification breaks because the system is solving the wrong problem. Continuous gesture control requires spotting boundaries, rejecting background motion, recognizing incomplete gestures early enough to feel responsive, and separating adjacent gestures in a stream. Recent benchmark reviews and continuous-gesture papers consistently frame this as a **detection/segmentation/decoding** problem, not just an isolated clip classification problem. citeturn13view0turn13view1turn32view0turn10search3

What you should build first is a **landmark-first, causal, always-on architecture** with explicit background modeling and an event decoder. For your first robust version, keep the front-end simple, use a **causal TCN** rather than a plain LSTM, and train on **continuous streams** that include random movement, preparation, gesture execution, and chained gestures. Add a lightweight command grammar after recognition, not inside the core recognizer. citeturn4search0turn22view0turn13view2turn33search5turn25view0

What you should avoid is making a fixed sliding window the semantic unit of recognition, relying on a plain isolated-gesture classifier with post-hoc smoothing, or using a bidirectional/offline model as the main online recognizer. Those designs either cut gestures in half, merge repeated gestures, or force you to trade away responsiveness for stability. End-to-end RGB-only video models can be very strong, but they are a worse first system because they need more data, more compute, and more engineering to get continuous online behavior right. citeturn22view0turn21search1turn13view6turn26view1

## Problem Framing

Your task is not “gesture classification.” It is a pipeline made of four linked problems:

- **Classification**: what gesture class is present, if any.
- **Spotting**: whether a gesture is happening at all in an untrimmed stream.
- **Segmentation**: where it starts and ends.
- **Command decoding**: how recognized gestures become actions, including repeated commands and chains. citeturn13view0turn7view3turn10search3turn22view0

That distinction matters because public continuous-gesture benchmarks explicitly evaluate overlap, false positives, latency, or sequence accuracy rather than only top-1 clip classification. The Montalbano benchmark uses framewise Jaccard overlap for spotting; ChaLearn ConGD is explicitly continuous and evaluated with Jaccard; SHREC 2022 adds latency and false positives to understand practical interaction feasibility; Köpüklü et al. proposed Levenshtein distance because online systems can fail by missing, duplicating, or misordering events even when individual gesture clips are classified well. citeturn7view3turn7view5turn10search3turn10search6turn22view0

This is also why your fixed-window LSTM failed. If the model sees “half a gesture + reset motion” or “end of previous gesture + start of next gesture,” its label target is ill-posed. Continuous finger-gesture work has described gestures in phases such as **preparation / nucleus / retraction**, and HMM-based continuous systems have similarly modeled **pre-stroke / nucleus / post-stroke** states. Those phase transitions are exactly what a real-time desktop control system needs to reason about. citeturn37view2turn29search17

The practical conclusion is that the recognizer should produce **streaming evidence** per frame or per short hop, and a downstream decoder should decide when to emit an event. That is the common shape behind detector–classifier hierarchies, online action detection models, CTC-style systems, and state-machine hybrids. citeturn22view0turn21search1turn13view2turn32view0turn33search5

## Model and Front-End Comparison

The table below is a **system-design synthesis** for your use case: always-on RGB webcam control, dynamic gestures, repeated gestures, and low false positives. It combines findings from continuous gesture benchmark surveys, online action detection work, continuous SHREC benchmarks, detector–classifier systems, and sequence-model papers. citeturn13view0turn13view1turn22view0turn26view1turn32view0turn10search3turn21search1

| Approach | Accuracy ceiling | Online latency | Data need | Fast dynamic gestures | Chained / repeated gestures | Always-on suitability | Practical take |
|---|---|---:|---:|---|---|---|---|
| Landmark + plain LSTM/GRU classifier | Good | Low | Moderate | Fair to good | Fair with decoder | Fair | Strong baseline, but easy to break with bad window alignment |
| Landmark + causal entity["scientific_concept","Temporal Convolutional Networks","causal dilated sequence modeling"] | Very good | Very low | Moderate | Good to very good | Good | Very good | Best practical core model |
| Landmark + tiny causal Transformer / Conformer | Very high | Low to moderate | Moderate to high | Good | Very good | Good | Better long-context modeling, more engineering |
| entity["scientific_concept","Dynamic Time Warping","elastic sequence alignment for time series"] / template matching | Low to moderate overall, high for few-shot personalization | Low | Very low | Good if templates are representative | Poor as sole spotter, good inside segmented gestures | Poor alone | Keep as personalization add-on, not main recognizer |
| HMM / CRF | Moderate | Low | Low to moderate | Fair | Good | Good | Best as decoder or thresholding layer, not best as primary representation learner |
| entity["scientific_concept","Connectionist Temporal Classification","sequence loss for unsegmented labels"] systems | High for unsegmented labeling | Low | Moderate to high | Good | Good, but repeated identical labels need careful blank/separator handling | Good | Useful when boundary labels are costly |
| Online action detection / temporal localization models | High | Low to moderate | High | Good | Very good | Very good | Excellent ideas, often heavier than needed unless adapted to landmarks |
| End-to-end RGB video models | Highest in principle | Moderate to high | High | Very good | Good | Fair to good | Great second-generation system, weak first practical build |
| Hybrid landmark + image / flow branch | Very high | Moderate | Moderate to high | Very good | Very good | Good | Best “accuracy-first” architecture |
| State machine + learned classifier hybrid | Depends on classifier | Very low | Low to moderate | Good | Very good | Excellent | Best system-level control layer |

A few comparisons are especially decisive.

**TCNs vs LSTMs.** The generic TCN paper found convolutional sequence models competitive with and often better than canonical recurrent models on sequence tasks, while avoiding recurrent training bottlenecks; that pattern lines up with practical online gesture systems, where local motion structure and fixed causal receptive fields matter more than very long unconstrained memory. In your setting, a causal TCN also makes it much easier to reason about latency because the receptive field is explicit. citeturn4search0turn32view0turn13view3

**Transformers vs TCNs.** Transformer-based online action detection models such as LSTR are attractive because they separate long-term memory from short-term detail, and recent hand-gesture work also explores graph-transformer hybrids for continuous recognition. But they generally make the most sense once you already have a solid streaming formulation and enough continuous data. For 10–30 desktop gestures on a laptop webcam, they are usually **version two**, not version one. citeturn26view1turn14view0turn25view0

**DTW still matters.** DTW remains valuable because it explicitly aligns variable-length sequences and is robust to time dilation; soft-DTW makes that alignment differentiable for learning. In practice, though, DTW is far better as a **few-shot personalized classifier on already-spotted segments** than as the main always-on recognizer, because it does not itself solve background rejection and repeated-event decoding cleanly. citeturn27search0turn28search0turn27search12turn25view0

**CTC is useful, but not enough by itself.** CTC removes the need for frame-aligned labels and is widely used for continuous visual sequence recognition, including sign-language transformers. But CTC collapses repeated labels and removes blanks during decoding; to represent repeated identical adjacent gestures, the alignment must include blanks or explicit boundary/separator structure. That makes CTC attractive for training or coarse decoding, but risky as the only mechanism for splitting `swipe-left, swipe-left`. citeturn34search1turn36search0turn36search7turn13view2

On the front-end side, the most practical choice is still usually **landmarks first**.

| Front-end | Strengths | Weaknesses | Recommendation |
|---|---|---|---|
| urlMediaPipe Hand Landmarker docsturn7view0 | Real-time, 2-hand support, handedness, image + world landmarks, live-stream tracking | Motion blur, occlusion, orientation edge cases, some jitter | Best default |
| urlMMPoseturn16view2 with RTMPose / RTMW | Stronger benchmark accuracy, flexible training/export, whole-body options | More engineering, less turnkey hand tracking | Best open alternative if GPU and customization matter |
| entity["software","MediaPipe Holistic","whole-body landmarking from RGB"] | Adds body/head context | More compute; hand ROI prediction can fail at non-ideal orientations | Use only if arm/torso context is essential |
| urlHaMeR project pageturn18view0 | Strong monocular 3D hand robustness under occlusion/truncation | Heavier; needs crops and hand-side input; aimed at mesh recovery | Research/high-accuracy branch, not first-line desktop control |
| End-to-end RGB recognizer | Uses all appearance and motion cues | Much harder to personalize and train online | Add later as fused branch if needed |

The evidence for staying with MediaPipe is still strong. The current Hand Landmarker supports live streams, up to multiple hands, handedness, image coordinates, and world coordinates, and uses tracking to avoid rerunning palm detection every frame; the official benchmark shows about 17.1 ms CPU and 12.3 ms GPU latency on a Pixel 6 for the full pipeline. The original paper likewise describes a two-stage palm-detector plus landmark model for real-time on-device tracking. citeturn7view0turn15view0turn15view1

The case **against** blindly using it is also real. A 2023 robustness study found motion blur, occlusion, and illumination shifts remain major obstacles for existing hand-pose estimators, and reported that diagonal motion blur caused substantial failures for MediaPipe Hands. Separately, a 2024 paper showed MediaPipe Holistic’s hand ROI prediction can struggle with non-ideal hand orientations. For your use case, that means fast wrist flicks and off-axis views are still the main places where front-end failures will dominate downstream recognition. citeturn20view0turn20view1

If you replace MediaPipe, the most practical alternative is not a research-only mesh model. It is likely an RTMPose/RTMW-style stack from MMPose: RTMPose reports real-time throughput on CPU/GPU, and RTMW improves whole-body accuracy while remaining deployment-oriented. But there is no clean apples-to-apples public benchmark proving that these models, as packaged, are better than MediaPipe for **two-hand webcam tracking during fast desktop flick gestures**. So the highest-confidence recommendation is: **stay with MediaPipe first, benchmark your own failure cases, then swap only if the front-end is the bottleneck.** citeturn17view0turn17view2turn16view0turn16view1turn13view0

## Best Architecture Recommendation

The architecture I would actually build first is:

**Front-end**
Use entity["software","MediaPipe Hands","real-time hand landmark tracking"] in live-stream mode, tracking up to two hands. Keep image landmarks, world landmarks, handedness, and tracker confidence. Persist track IDs across short dropouts. citeturn7view0turn15view0

**Features**
For each hand, create a canonical representation: wrist- or palm-centered coordinates, scale normalization, optional rotation normalization, finger curl/spread angles, fingertip pair distances, palm normal / orientation, fingertip and wrist velocities, accelerations, and confidence masks. Mirror left hands into a canonical frame for “shape-equivalent” gestures, but retain handedness if left-vs-right matters. For two-hand gestures, add relative translation, relative scale, relative orientation, inter-hand fingertip distances, and a “both hands present” mask. Recent customization and continuous-recognition work on hand keypoints, graph transformers, and MediaPipe-based pipelines supports landmark-centric representations as a data-efficient starting point. citeturn25view0turn13view3turn30view0turn7view0

**Temporal core**
Use a **causal multi-head TCN** over the last roughly 0.5–1.5 seconds of features. The model should output:
- `p(background)` and `p(intentional-motion)`
- per-frame gesture class logits
- `p(start)` and `p(end)` boundary heads
- optional hand-role head: left-only / right-only / both-hands.  

This solves the fixed-window problem because the model is not asked to classify a pre-cut clip. It continually labels the stream and predicts boundaries. The TCN keeps latency explicit and low; the background and boundary heads let the system separate “random movement,” “preparation,” “gesture,” and “return to neutral.” citeturn4search0turn13view2turn22view0turn32view0turn37view2

**Decoder**
Run a lightweight event decoder on top of the framewise outputs. The decoder should:
- wait for either a start peak or sustained intentional-motion evidence,
- accumulate class posteriors while active,
- emit the event when an end peak or a background valley appears,
- split repeated gestures only when there is a **new** start or a clear blank/background separation,
- suppress double-firing from the same posterior peak with a tiny refractory window, not a long cooldown.  

This is the part that lets `swipe left → swipe left → swipe up` work without merging into one long motion. Detector–classifier systems, SHREC-style online benchmarks, Gesture Spotter, and online action detection work all support explicit event logic as a practical necessity. citeturn22view0turn10search3turn33search5turn26view1

**Command layer**
Maintain an event queue separate from the recognizer. The queue stores `(gesture, hand-role, start, end, confidence)` and optionally decodes short histories. For most desktop control, a tiny finite-state grammar is enough. Example rules:
- repeated swipes are allowed if separated by a start/end cycle,
- one-hand gestures can immediately follow a two-hand gesture,
- destructive commands require either higher confidence, a specific context, or a confirm gesture.  

This layer should be simple and hand-written at first. Grammar is useful, but you do not need a learned gesture language model on day one. citeturn33search5turn29search17turn10search3

**Personalization**
Add a second-stage personalized recognizer later. When the spotter segments a candidate gesture, feed the segment to either:
- a prototypical / contrastive embedding head, or
- a DTW/soft-DTW nearest-prototype matcher over normalized trajectories.  

This keeps the always-on detector stable while letting users add personal gestures with only a few examples. Recent on-device customization and one-demonstration gesture work strongly supports this split between **generic spotter** and **custom recognizer**. citeturn24view0turn25view0turn27search0turn28search0

Here is the live pipeline in pseudocode:

```python
state = IDLE
event_queue = []
history = RingBuffer(seconds=2.0)

for frame in webcam_stream():
    hand_obs = hand_tracker(frame)          # up to 2 hands
    feats = preprocess(hand_obs)           # normalization, masks, velocities, two-hand features
    history.push(feats)

    y = tcn(history.recent())              # causal outputs every frame or every 2-3 frames
    p_bg = y.background
    p_cls = y.class_probs
    p_start = y.start_prob
    p_end = y.end_prob

    if state == IDLE:
        if p_start.max() > T_start or intentional_motion(y):
            state = ACTIVE
            segment.reset()
            segment.start_time = now()

    if state == ACTIVE:
        segment.add_frame(p_cls, feats)

        if p_end.max() > T_end or sustained_background(p_bg):
            gesture = decode_segment(segment)   # posterior integral + boundary checks
            if gesture.confidence > T_emit:
                if not duplicate_of_recent(gesture, event_queue):
                    event_queue.append(gesture)
                    execute_or_buffer(gesture, event_queue)
            state = IDLE
```

The diagram version is simpler to reason about:

```text
RGB webcam
  ↓
Hand tracking
  ↓
Landmark normalization + masking + velocity features
  ↓
Causal sequence model
  ├─ background / no-gesture
  ├─ gesture class evidence
  └─ start/end boundaries
  ↓
Streaming decoder / hysteresis / split-merge logic
  ↓
Event queue
  ↓
Command grammar / safety gating / execution
  ↓
Logging for retraining
```

This architecture solves your sliding-window issue because **windows become compute context, not gesture containers**. The model sees rolling context, but the emitted gesture is defined by learned boundaries and decoder logic, not by the arbitrary ends of a 32-frame buffer. That is exactly the conceptual move behind online detection systems, spotting-recognition frameworks, and controlled CTC formulations. citeturn22view0turn13view2turn26view1turn32view0

## Gesture Chaining and Command Decoding

For chains like `swipe left, swipe left, swipe up` or `pinch, swipe right, open palm`, the core rule is simple: **the recognizer should output gesture events, not only frame labels**. Once you have explicit events, chaining becomes a stream-decoding problem instead of a clip-classification problem. citeturn22view0turn33search5turn10search3

The most reliable practical strategy is:

- **independent spotting** with explicit start/end logic,
- **short event queue** storing the last 1–5 emitted gestures,
- **tiny grammatical layer** for special command sequences,
- **minimal refractory period** of only a few frames after emission,
- **duplicate suppression by peak identity**, not by class cooldown.  

That last point matters. A class-level cooldown is exactly how you accidentally kill rapid repeated swipes. Instead, suppress only the *same posterior hill* from firing twice. If a new start peak and a new evidence build-up occur, allow a second identical gesture immediately. citeturn33search5turn22view0turn13view2

A useful decoding rule set for repeated gestures is:

- emit a gesture only once its accumulated posterior passes threshold,
- do not merge adjacent same-class gestures if there is either
  - a boundary-head reactivation,
  - a background valley,
  - a trajectory reset such as velocity sign reversal or hand-shape reset,
  - or a minimum nucleus-duration completion followed by re-acceleration.  

The rationale comes from continuous finger-gesture spotting work that explicitly detects boundaries around the “nucleus” and from classical phase-based continuous recognition. citeturn37view2turn29search17

CTC-style decoding can help if you do not want frame-level boundaries during training, but it needs care for repeated identical gestures. CTC’s collapse operator removes repeats and blanks after alignment, so back-to-back identical tokens need blanks or separators to remain distinct. For your command layer, that means either:
- train with an explicit separator / blank state and decode with beam search, or
- keep CTC only as an auxiliary loss while the final event splitting is done by start/end heads and a decoder.  

For desktop control, I recommend the second option. It is much easier to debug. citeturn36search0turn36search7turn34search1turn13view2

Two-hand to one-hand transitions are easiest if your recognizer has an explicit **hand-role head** and your feature representation always includes masks for missing hands. Then the decoder sees `both-hands active` ending before `right-hand active` begins, instead of interpreting the disappearance of one hand as tracker noise. One-demonstration customization work explicitly supports one-handed and two-handed gestures, which is a strong signal that the representation should be unified rather than split into separate projects. citeturn25view0turn7view0

## Training and Data Collection Plan

If you collect your own data, do **not** build the dataset entirely out of isolated clips. The public continuous-gesture benchmark literature is very clear that real-world streams contain natural motion, clutter, no-gesture segments, and variable transitions, and that these factors are central to the task rather than noise to be removed. IPN Hand is especially relevant here because it was designed with continuous gestures, random breaks, clutter, illumination changes, and natural non-gesture segments, and it explicitly emphasizes large speed variation within classes. citeturn7view4turn13view0

A practical collection plan is:

**Core dataset**
Record three kinds of data from the start:
- **isolated gestures** for fast labeling and bootstrapping class separability,
- **continuous scripted streams** containing chains and repeated gestures,
- **background / random movement streams** with no intended gesture at all.  

For 10–30 gestures, a reasonable starting point is on the order of **100–300 isolated examples per gesture**, plus **many continuous streams** where each stream contains 5–20 gestures mixed with natural movement. The number matters less than the diversity: different speeds, distances, camera positions, seated posture, left/right hand choice, and lighting. That aligns with the way IPN Hand, EgoGesture, and recent customization work stress variation, background classes, and cross-view behavior. citeturn7view4turn8search1turn25view0

**Fast gestures**
For wrist flicks and short swipes, ask performers to do:
- normal speed,
- exaggeratedly fast,
- intentionally tiny amplitude,
- and “messy” versions embedded in surrounding movement.  

Motion blur is a real failure mode for hand tracking, so you want those cases in the training and validation data instead of discovering them only after deployment. citeturn20view0

**Chains**
Script sequences such as:
- `L, L, U`
- `pinch, right, open`
- `zoom, confirm`
- random mixtures with variable pauses
- same gesture repeated 2–4 times.  

Label them as continuous streams. This matters because a model trained only on isolated clips tends to learn gesture interiors, not the transition logic required for chains. Continuous-gesture datasets and online benchmarks exist for exactly this reason. citeturn7view3turn7view5turn7view4turn10search3

**How to label**
If you can afford it, label **start and end frames** for continuous streams. That gives you the cleanest path to boundary heads and event-level evaluation. If full framewise labeling is too expensive, label only gesture intervals and optionally derive frame labels automatically. If even that is costly, use sequence labels plus CTC-style training as a bootstrap, but I would still create at least a smaller boundary-labeled set for validation and decoder tuning. Continuous benchmark work repeatedly notes that evaluation and spotting quality depend heavily on interval-aware labels. citeturn13view0turn7view3turn10search3turn13view2

**Augmentation**
For landmark data, the highest-value augmentations are:
- temporal resampling for speed variation,
- small spatial jitter,
- random frame drops,
- synthetic missing-hand intervals,
- small rotations/scales/translations,
- handedness mirroring where allowed,
- and background-class augmentation from daily non-gesture movement.  

For personalized gestures, keep a few clean templates and add augmented variants rather than forcing the base model to memorize every user. That matches both the few-shot/customization literature and DTW’s strengths on variable-rate sequences. citeturn25view0turn24view0turn27search0turn28search0

**Validation that reflects reality**
Keep one validation split made only of **messy continuous sessions**:
- user talks, scratches face, adjusts posture,
- gestures happen while moving between tasks,
- repeated gestures are common,
- two-hand gestures transition into one-hand gestures,
- and many sessions contain zero commands.  

Measure:
- event precision/recall,
- false positives per minute during idle use,
- detection latency,
- Jaccard/IoU on intervals,
- and sequence Levenshtein / WER on command chains.  

That evaluation setup is much closer to desktop control than clip-level accuracy. citeturn22view0turn7view3turn10search3turn13view0

## Datasets, Papers, Tools, and Final Ranked Answer

The most relevant datasets are the ones that actually include **continuous streams, non-gesture motion, or explicit online evaluation**.

| Dataset | Why it matters for you | Main limits |
|---|---|---|
| **Montalbano / ChaLearn 2014** | Continuous spotting, 20 gestures, interval labels, Jaccard evaluation | RGB-D era; body-skeleton centric, not webcam-only |
| **ChaLearn ConGD** | Large-scale continuous dataset with 249 labels and Jaccard evaluation | RGB-D, large-vocabulary challenge setup, less desktop-like |
| **IPN Hand** | Very relevant: continuous RGB videos, clutter, illumination variation, non-gesture segments, speed variation, touchless control vocabulary | Single-camera RGB, but still curated and domain-specific |
| **SHREC 2021 / 2022** | Online heterogeneous gestures, non-gesture padding, latency and false-positive evaluation | 3D tracked-hand streams, not raw RGB webcam |
| **EgoGesture** | Large dynamic RGB-D benchmark with many classes and subjects | Mostly segmented clips, egocentric/wearable setting |
| **NVGesture** | Classic dynamic benchmark for online detection/classification history | RGB-D/IR, not ordinary laptop webcam |
| **LD-ConGR** | Continuous long-distance gestures with segmentation labels | RGB-D and long-distance meeting/smart-home focus |
| **Briareo** | Automotive dynamic gestures and segmentation/classification context | In-car viewpoint and multi-sensor setup |
| **HaGRIDv2** | Huge modern gesture pretraining resource with dynamic/two-hand classes and richer no-gesture set | Primarily image-centric, not a continuous benchmark |

Evidence for those summaries comes directly from the dataset descriptions and recent benchmark surveys. Montalbano contains more than 14,000 gestures from 20 categories for continuous spotting. ConGD contains 47,933 gestures in 22,535 continuous RGB-D videos across 249 labels. IPN Hand has 50 subjects, more than 800,000 frames, 13 static/dynamic classes, many continuous gestures per video, clutter, illumination variation, and natural non-gesture segments. SHREC 2021 and 2022 explicitly interleave gestures with non-gesture motion and evaluate false positives and latency. LD-ConGR adds fine-grained temporal segmentation and long-distance capture. HaGRIDv2 enlarges the no-gesture class with natural hand movement and adds dynamic plus two-hand gestures. citeturn7view3turn7view5turn7view4turn10search2turn10search3turn35search0turn8search3turn9search0turn13view0

The papers and tools I would prioritize are:
- the 2025 continuous-gesture benchmark review for task framing and evaluation, citeturn13view0
- the 2025 VHGR review for broader model/dataset coverage, citeturn13view1
- MediaPipe Hands and current Hand Landmarker docs for the practical front-end, citeturn7view0turn15view0
- RTMPose / RTMW and MMPose if you want the main practical alternative, citeturn17view0turn17view2turn16view2
- detector–classifier online gesture work for event-emission logic, citeturn22view0
- LSTR and controlled CTC work for causal sequence modeling and earliness trade-offs, citeturn26view1turn13view2
- OO-dMVMT and recent graph-transformer hand papers for state-of-the-art continuous landmark recognition ideas, citeturn32view0turn14view0turn13view3
- recent customization papers if you want user-trainable gestures early. citeturn24view0turn25view0

Useful tool starting points are:
- urlMediaPipe Hand Landmarker docsturn7view0
- urlMediaPipe gesture customizer guideturn23search9
- urlMMPoseturn16view2
- urlHaMeR project pageturn18view0

My final ranked recommendations are:

**Best overall setup**  
Use **MediaPipe Hands + causal TCN sequence labeler + background/start/end heads + event decoder + command queue**.  
Why it wins: best balance of latency, robustness, implementation effort, and extensibility; it directly solves the fixed-window alignment problem; and it supports one-hand, two-hand, repeated, and chained gestures. citeturn7view0turn22view0turn4search0turn32view0

**Best model core**  
A **causal TCN** is the best default temporal model.  
Why it beats alternatives: lower engineering burden than a Transformer, better online structure than a BiLSTM, and more stable/explicit latency than a plain LSTM. citeturn4search0turn26view1

**Best preprocessing**  
Use **canonicalized landmarks + confidence masks + velocities/accelerations + two-hand relational features**. Keep smoothing light and compute motion features carefully so you do not kill flicks. This is the right balance between data efficiency and responsiveness for webcam control. citeturn25view0turn13view3turn30view0

**Best decoding method**  
Use a **state-machine/event decoder** over framewise outputs, not raw argmax over windows. Add a small event grammar only where needed. For identical repeated gestures, require a blank/background valley or a fresh start signal rather than a long cooldown. citeturn33search5turn36search7turn22view0turn13view2

**Best simple but surprisingly effective architecture**  
MediaPipe landmarks + handcrafted normalization + small causal 1D CNN/TCN + hysteresis decoder.  
Why: easier than a Transformer, usually better online than a fixed-window LSTM, and often enough for 10–15 command gestures. citeturn7view0turn4search0turn22view0

**Best high-accuracy but more complex architecture**  
Landmark stream + cropped-hand RGB/flow stream + causal Transformer/Conformer-style fusion + boundary-aware decoder.  
Why: appearance cues help with fast flicks, ambiguous poses, and front-end errors that landmarks alone cannot recover. citeturn13view6turn21search1turn13view3

**Best architecture for user-trainable gestures**  
Generic always-on spotter + personalized embedding or DTW/soft-DTW recognizer on the segmented gesture.  
Why: it separates the hard universal problem of spotting from the inherently user-specific problem of custom-label recognition. citeturn24view0turn25view0turn28search0turn27search0

**Best architecture if staying with MediaPipe**  
Exactly the overall recommendation above. MediaPipe is still the strongest practical choice unless your own test logs prove front-end failure is dominate. citeturn7view0turn20view0

**Best architecture if replacing MediaPipe**  
Use MMPose RTMPose/RTMW-style tracking, but keep the rest of the stack the same: normalized landmarks → causal TCN/Transformer spotter → decoder.  
Why: if you swap the tracker, you still do not want to throw away the event-decoding architecture that actually solves continuous recognition. citeturn17view0turn17view2turn16view2

**Best way to evaluate success**  
Do not optimize for clip accuracy first. Optimize for:
- false positives per idle minute,
- event F1 / Jaccard on continuous streams,
- detection latency,
- chain Levenshtein / WER,
- confusion on repeated same-class gestures,
- and personalization accuracy after 1–5 demos. citeturn13view0turn22view0turn10search3turn7view3

**Open questions / limitations**  
There is still no perfect public benchmark that matches your exact target of always-on **RGB laptop webcam** control with fast flicks, messy continuous motion, and desktop-command chains. Many continuous datasets are RGB-D or tracked-skeleton based, and even the strongest practical front-ends still show weakness under motion blur, occlusion, and off-axis hand orientation. That means your final architecture decision should be guided by a small, realistic in-house benchmark as early as possible. citeturn13view0turn20view0turn20view1turn10search3