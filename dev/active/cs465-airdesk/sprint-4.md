# AirDesk Sprint 4 Plan

## Purpose

Sprint 4 should turn Sprint 3's runtime logs and dynamic gesture recordings into the first real dynamic-gesture training and evaluation pipeline.

Sprint 3 is expected to produce:

- runtime JSONL event logs,
- continuous positive and negative gesture recordings,
- an intent-gated phrase recognizer foundation,
- rule/DTW scaffolding for small dynamic gestures,
- live command feedback,
- pause/kill-switch behavior,
- and optionally guarded Hyprland execution.

Sprint 4 should answer:

> Can AirDesk's chosen causal TCN path recognize intentional dynamic gestures in continuous desktop-control sessions with low false activation and acceptable latency?

The evaluation should be evidence-based and use continuous sessions with background motion, aborted gestures, and chained gestures. Do not optimize isolated clip accuracy alone. AirDesk needs low false activations, acceptable latency, and a feeling of intentional control.

## Sprint Theme

> Build the dataset, labeling, feature pipeline, and first continuous gesture-spotting recognizer.

## Product / Research Stance

- Keep AirDesk's research claims narrow and evidence-based.
- Preserve dry-run as the default for model experiments.
- Treat user-specific calibration as acceptable for the personal prototype.
- Do not pretend a model is better because it has high isolated accuracy.
- Evaluate the causal TCN on continuous streams with negative/background motion.
- Do not spend this sprint comparing every model family.
- Keep model training optional at runtime; AirDesk should still work without ML dependencies.

## Non-Goals

Do not attempt these in Sprint 4:

- polished UI/control center,
- Kinect integration,
- global cursor takeover,
- virtual keyboard,
- multi-user formal study,
- large ST-GCN/Transformer production deployment,
- automatic cloud training,
- real desktop actions driven by experimental ML by default.

## Key Decisions

### Model Strategy

Sprint 4 should build one primary learned recognizer behind the AirDesk interface:

1. **Rule phrase recognizer**
   - safety/debug scaffold
   - interpretable and fast
   - useful for intent gates, cooldowns, and failure analysis

2. **DTW/template recognizer**
   - low-data fallback and calibration tool
   - good for "conducting" gestures with variable speed
   - supports per-user calibration

3. **Causal TCN / stream model**
   - first learned-model scaffold
   - good for framewise phase/state prediction
   - likely easier to train/evaluate than Transformer for small data
   - should evolve away from whole-window classification toward stream/event labels

LSTM/GRU should be deferred unless the causal TCN/spotting path disappoints or a later comparison becomes worth the time. ST-GCN and graph transformers should remain research notes unless the dataset grows enough to justify them. A transformer is only interesting after AirDesk has position-invariant features, phase/event labels, and event decoding; a fixed-window transformer would inherit the same boundary failures as the current TCN.

### Labeling Strategy

Label continuous streams, not just clipped gestures.

Every labeled session should support:

- event-level labels:
  - gesture name,
  - intended command,
  - start timestamp,
  - end timestamp,
  - commit timestamp,
  - success/failure note;
- phase-level labels:
  - background,
  - preparation,
  - armed,
  - stroke,
  - hold,
  - release,
  - cooldown,
  - aborted.

The labeling format should live beside JSONL recordings and never require raw video.

### Feature Strategy

Use normalized landmark-derived features so models stay backend-independent:

- palm center,
- palm velocity and acceleration,
- index-tip relative trajectory,
- thumb-index pinch distance and derivative,
- palm width/hand scale,
- approximate palm roll/yaw if stable,
- finger extension/fold features,
- confidence and tracking continuity,
- tracking loss flags.

The next feature preset should reduce dependence on absolute `palm_x`, `palm_y`, and `palm_z`. Keep them for diagnostics, but train a position-invariant recognizer variant from wrist/palm-centered landmarks, hand-scale-normalized motion, velocity, acceleration, displacement, and direction consistency.

Keep feature extraction deterministic and tested.

### Evaluation Strategy

Primary metrics:

- false activations per minute,
- missed intended gestures,
- repeated-fire count,
- command latency,
- segmental F1,
- gesture stream edit distance / word error rate,
- cancellation success,
- average confidence on background,
- user-rated control/confidence,
- fatigue/discomfort notes.

Model selection should prioritize low false activation and low latency over raw classification accuracy.

## Target CLI Shape

By the end of Sprint 4, these commands should exist or be expanded:

```text
airdesk label init data/recordings/live-window-manager-dry-run.jsonl --out data/labels/live-window-manager-dry-run.labels.json
airdesk label validate data/labels/live-window-manager-dry-run.labels.json
airdesk features export data/recordings/live-window-manager-dry-run.jsonl --labels data/labels/live-window-manager-dry-run.labels.json --out data/features/live-window-manager-dry-run.parquet
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer rule
airdesk gesture train --model-type tcn --features data/features/*.parquet --out data/models/gestures/tcn.pt
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer tcn --model data/models/gestures/tcn.pt
airdesk gesture calibrate --kind dtw --recordings data/recordings/*.jsonl --labels data/labels/*.json --out data/models/gestures/caden-dtw.json
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer dtw --model data/models/gestures/caden-dtw.json
```

Exact file formats can change if a simpler JSONL/CSV path is better, but the command responsibilities should stay.

## Deliverables

### 1. Label Schema

Acceptance criteria:

- Define a typed label schema for continuous gesture sessions.
- Support event labels and phase labels.
- Include session metadata:
  - recording path,
  - profile,
  - participant/user ID,
  - camera settings,
  - tracker settings,
  - notes.
- Validate labels without requiring live dependencies.
- Tests cover valid and invalid labels.

### 2. Labeling CLI

Acceptance criteria:

- Add `airdesk label init` to create a starter label file from a recording.
- Add `airdesk label validate`.
- Add a minimal non-GUI workflow for manually editing timestamps.
- Document the labeling workflow.
- Do not build a polished label editor yet.

### 3. Feature Extraction

Acceptance criteria:

- Add deterministic feature extraction from `TrackingFrame`.
- Features include palm center, velocities, scale-normalized motion, pinch metrics, finger pose metrics, confidence, and continuity.
- Export features to a simple checked format.
- Tests cover feature extraction on synthetic hands and replay fixtures.

### 4. Rule / DTW Fallbacks

Acceptance criteria:

- Add a template representation for dynamic gesture phrases.
- Add DTW or similar trajectory matching over normalized features.
- Support per-user calibration from labeled recordings.
- Evaluate DTW on continuous replay streams.
- Tests cover matching variable-speed synthetic gestures.
- Document that DTW is fallback/calibration, not the primary Sprint 4 model bet.

### 5. Causal TCN Recognizer

Acceptance criteria:

- Add optional ML dependencies only if needed.
- Train or prototype a small causal TCN.
- Models consume exported AirDesk features, not raw video.
- Training is reproducible from ignored local data.
- Model inference can run over replay streams.
- A follow-up stream-label target is documented or implemented so the model can distinguish stroke from reset/recovery.
- LSTM/GRU is explicitly deferred unless the TCN path fails.
- If ML implementation is too large for Sprint 4, document the exact blocker and still complete labels/features/evaluation.

### 5.5. Event Decoder / Hybrid Spotter

Acceptance criteria:

- Convert recognizer probability streams or candidate scores into gesture events.
- Include hysteresis, confidence peak/commit logic, minimum event separation, recovery/cooldown, and repeated-fire suppression.
- Support dry-run/replay evaluation before any live desktop action wiring.
- Compare event-decoded TCN/hybrid output against DTW candidates on isolated holdout and chained sessions.
- Document whether a motion-energy/DTW candidate gate improves false activations or only hides weak model behavior.

### 6. Evaluation Harness

Acceptance criteria:

- Add `airdesk gesture evaluate`.
- Evaluate rule, DTW fallback, and causal TCN recognizers on the same sessions.
- Report false activations, missed gestures, latency, repeated fires, segmental F1, and stream edit distance where practical.
- Export results to JSON or CSV for study/paper use.
- Tests cover metric calculations.

### 7. Dynamic Gesture Protocol

Acceptance criteria:

- Add or update a protocol for collecting:
  - intended flicks,
  - intended slow swipes,
  - palm tilts,
  - aborted gestures,
  - chained gestures,
  - normal desk motion,
  - reaching for keyboard/mouse,
  - hand entering/leaving frame.
- Include guidance for lighting, distance, camera settings, repetitions, and rest breaks.

### 8. Model Selection Decision

Acceptance criteria:

- Document a Sprint 4 decision:
  - whether causal TCN should drive Sprint 5 pilot tasks,
  - which gestures are stable enough,
  - which gestures are deferred,
  - whether rule/DTW fallback remains safer than TCN for any command.
- Update `research-notes.md`, `tasks.md`, and handoff docs.

## Recommended Implementation Order

1. Define label schema and tests.
2. Add label init/validate CLI.
3. Add feature extraction and export.
4. Add evaluation metric utilities.
5. Evaluate the current rule phrase recognizer over synthetic and recorded sessions.
6. Implement the causal TCN training/inference path.
7. Evaluate the TCN on continuous logs.
8. Add DTW/template fallback only where it improves calibration or safety.
9. Document the Sprint 5 recognizer decision.
10. Run `ruff`, `pytest`, and replay evaluation smoke commands.

## Risks and Mitigations

### Too Little Labeled Data

Risk:

- The causal TCN may look promising but overfit.

Mitigation:

- Treat rule/DTW as the Sprint 4 fallback.
- Collect background/negative data first.
- Use the TCN only if it beats simpler fallbacks on held-out continuous sessions.

### Labeling Becomes a Time Sink

Risk:

- Manual phase labeling can take longer than implementation.

Mitigation:

- Start with event labels.
- Add phase labels only for selected sessions.
- Use rough timestamp labels before building a GUI.

### Metrics Are Misleading

Risk:

- A model may score well on gesture clips but fail in live streams.

Mitigation:

- Use continuous sessions with background segments.
- Report false activations per minute.
- Include latency and repeated-fire metrics.

### ML Dependencies Pollute Runtime

Risk:

- Installing PyTorch or similar makes the base AirDesk workflow heavier.

Mitigation:

- Keep ML dependencies optional.
- Keep runtime and tests working without ML extras.

## Definition of Done

Sprint 4 is done when:

- continuous gesture labels have a schema and CLI validation,
- features can be exported from recordings,
- rule and DTW fallback recognizers can be evaluated on continuous sessions,
- a causal TCN prototype is implemented or explicitly blocked with rationale,
- LSTM/GRU remains explicitly deferred unless the TCN path fails,
- evaluation metrics report false activations and latency,
- a Sprint 5 recognizer decision is documented,
- README/tasks/handoff are updated,
- `uv run ruff check .` passes,
- `uv run pytest` passes.

## Sprint 5 Gate

At the end of Sprint 4, choose:

### Path A: One Recognizer Is Pilot-Ready

Proceed to a study/pilot sprint focused on task logging, trial protocols, and evidence for the paper.

### Path B: No Recognizer Beats Safety Thresholds

Stay in recognition robustness:

- collect more data,
- narrow gesture vocabulary,
- tune phrase gates,
- improve DTW features,
- postpone real execution.

### Path C: Rule/DTW Works But TCN Does Not

Use rule/DTW for the study prototype and frame learned dynamic recognition as future work.
