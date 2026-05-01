# AirDesk Sprint 4 Plan

## Purpose

Sprint 4 should turn Sprint 3's runtime logs and dynamic gesture recordings into a real recognition evaluation pipeline.

Sprint 3 is expected to produce:

- runtime JSONL event logs,
- continuous positive and negative gesture recordings,
- an intent-gated phrase recognizer foundation,
- rule/DTW baselines for small dynamic gestures,
- live command feedback,
- pause/kill-switch behavior,
- and optionally guarded Hyprland execution.

Sprint 4 should answer:

> Which recognition approach actually works best for AirDesk's continuous desktop-control setting?

The comparison should be evidence-based and use the same recorded sessions across recognizers. Do not optimize isolated clip accuracy alone. AirDesk needs low false activations, acceptable latency, and a feeling of intentional control.

## Sprint Theme

> Build the dataset, labeling, and model-evaluation loop.

## Product / Research Stance

- Keep AirDesk's research claims narrow and evidence-based.
- Preserve dry-run as the default for model experiments.
- Treat user-specific calibration as acceptable for the personal prototype.
- Do not pretend a model is better because it has high isolated accuracy.
- Compare models on continuous streams with negative/background motion.
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

### Model Comparison Strategy

Sprint 4 should compare recognizers behind the same AirDesk interface:

1. **Rule phrase recognizer**
   - current baseline
   - interpretable and fast
   - useful for safety gates and debugging

2. **DTW/template recognizer**
   - likely best low-data personal recognizer
   - good for "conducting" gestures with variable speed
   - supports per-user calibration

3. **LSTM/GRU baseline**
   - familiar and lightweight
   - included because Caden has prior experience
   - not assumed to solve spotting alone

4. **Causal TCN baseline**
   - preferred first learned model candidate
   - good for framewise phase/state prediction
   - likely easier to train/evaluate than Transformer for small data

ST-GCN and graph transformers should remain research notes unless the dataset grows enough to justify them.

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
airdesk gesture calibrate --kind dtw --recordings data/recordings/*.jsonl --labels data/labels/*.json --out data/models/gestures/caden-dtw.json
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer rule
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer dtw --model data/models/gestures/caden-dtw.json
airdesk gesture train --model-type tcn --features data/features/*.parquet --out data/models/gestures/tcn.pt
airdesk gesture evaluate --recordings data/recordings/*.jsonl --labels data/labels/*.json --recognizer tcn --model data/models/gestures/tcn.pt
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

### 4. DTW / Template Recognizer

Acceptance criteria:

- Add a template representation for dynamic gesture phrases.
- Add DTW or similar trajectory matching over normalized features.
- Support per-user calibration from labeled recordings.
- Evaluate DTW on continuous replay streams.
- Tests cover matching variable-speed synthetic gestures.

### 5. Learned Model Baselines

Acceptance criteria:

- Add optional ML dependencies only if needed.
- Train or prototype an LSTM/GRU baseline.
- Train or prototype a causal TCN baseline.
- Models consume exported AirDesk features, not raw video.
- Training is reproducible from ignored local data.
- Model inference can run over replay streams.
- If ML implementation is too large for Sprint 4, document the exact deferred scope and still complete DTW/evaluation.

### 6. Evaluation Harness

Acceptance criteria:

- Add `airdesk gesture evaluate`.
- Evaluate rule, DTW, and learned recognizers on the same sessions.
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
  - which recognizer should drive Sprint 5 pilot tasks,
  - which gestures are stable enough,
  - which gestures are deferred,
  - whether learned models beat rule/DTW baselines.
- Update `research-notes.md`, `tasks.md`, and handoff docs.

## Recommended Implementation Order

1. Define label schema and tests.
2. Add label init/validate CLI.
3. Add feature extraction and export.
4. Add evaluation metric utilities.
5. Implement DTW/template recognizer.
6. Evaluate rule/DTW over synthetic and recorded sessions.
7. Add LSTM/GRU baseline if data volume is enough.
8. Add TCN baseline if data volume is enough.
9. Compare recognizers on continuous logs.
10. Document model selection decision.
11. Run `ruff`, `pytest`, and replay evaluation smoke commands.

## Risks and Mitigations

### Too Little Labeled Data

Risk:

- Learned models may look promising but overfit.

Mitigation:

- Treat DTW/template as the likely Sprint 4 fallback.
- Collect background/negative data first.
- Use ML only if it beats simpler baselines on held-out continuous sessions.

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
- rule and DTW recognizers can be evaluated on continuous sessions,
- LSTM/TCN baselines are implemented or explicitly deferred with rationale,
- evaluation metrics report false activations and latency,
- a model-selection decision is documented,
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

### Path C: Rule/DTW Works But ML Does Not

Use rule/DTW for the study prototype and frame ML as future work.
