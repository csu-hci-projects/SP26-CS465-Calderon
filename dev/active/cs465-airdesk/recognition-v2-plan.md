# AirDesk Recognition V2 Plan

## Purpose

This is the planning entrypoint for the next recognizer architecture pass. It is based on AirDesk's live/replay evidence and Caden's `deep-research-report.md`.

The important shift is not "TCN is bad." The shift is:

> The current TCN is a sliding-window phase classifier. AirDesk needs a continuous gesture spotting system.

The report's recommended shape is consistent with the evidence:

```text
webcam
  -> hand tracker
  -> normalized per-hand landmark/motion features
  -> continuous spotter
  -> event decoder
  -> command queue / safety policy
```

The temporal core can still be a small causal TCN later, but it should not be trained as one argmax target over an arbitrary rolling window. The model should produce streaming evidence and boundaries; the decoder should turn that evidence into one-shot command events.

## Evidence For The Pivot

Keep the claims narrow:

- rule recognizers failed natural swipes;
- same-batch DTW was optimistic;
- holdout DTW and early TCN exposed left-swipe/generalization weakness;
- gated/window-feature DTW improved isolated holdout but was tuned on existing evidence;
- two-hand feature export and hand-scoped DTW/TCN paths are now implemented;
- shared per-hand TCN is the right deployment shape: one checkpoint applied independently to each visible `hand_id`, then decoded/merged;
- separate `hand-0` / `hand-1` tracker-slot models remain deferred because MediaPipe tracker ids are not stable physical hand identities;
- chart labels are prompt-timing weak labels, not exact active-hand truth;
- recovery-inclusive TCN collapsed into `recovery` during live preview;
- `phase-stroke` removed the recovery class and cleaned the live surface, but did not solve recognition;
- Caden saw live `dx > 0.50` while `L=` / `R=` stayed near zero, so the failure is not just a motion-gate threshold;
- the more sensitive 0.20 motion-gate model improved 003-to-004 replay to 37/48 but raised false activations, so it is diagnostic only.

Conclusion:

> Stop trying to rescue the current TCN with threshold sweeps. Use it as evidence. Build a cleaner continuous-spotting architecture.

## Target Architecture

Recognition V2 should split responsibilities explicitly.

```text
TrackingFrame
  -> FeatureRowStream / normalized per-hand streams
  -> Motion activity proposal
  -> Optional segment/candidate builder
  -> Recognizer scorer
  -> Event decoder
  -> Command queue
  -> Mode/profile/safety policy
```

## Review Conclusion

After reviewing the current code boundaries, this plan survives, but the first
implementation slice should be narrower than a package-wide refactor.

Current useful boundaries already exist:

- `features.landmarks.FeatureRowStream` emits one feature row per visible hand and
  keeps independent per-`hand_id` motion history;
- `gestures.decoder.EventDecoder` already converts score/candidate streams into
  one-shot events and merges hand streams;
- DTW and TCN evaluation paths already consume hand-scoped feature streams;
- command-mode policy and action dispatch already live outside recognizer code.

So the next pass should not start by moving everything into a new
`airdesk/recognition/` package. Start with a small deterministic motion-evidence
module at the existing gesture boundary, prove the replay/evaluation surface, and
only split out a new recognition package when the baseline makes the ownership
clear.

Practical near-term shape:

```text
TrackingFrame / recording JSONL
  -> FeatureRowStream / exported feature CSV rows
  -> deterministic per-hand motion scorer
  -> GestureCandidate-compatible events with motion evidence metadata
  -> existing evaluation / decoder utilities
  -> command queue later, still dry-run only
```

The scorer should be deliberately simple and inspectable. It is not a learned
recognizer and should not become another threshold-sweep project. Its job is to
answer one question:

> Can the current tracker and feature stream produce replay-stable per-hand swipe
> events before a learned scorer is involved?

If the answer is no, TCN v2 is premature.

### Per-Hand Streams

Each visible hand is its own temporal stream. Histories must never mix between hands.

Use a shared model/scorer across streams unless a future labeling system provides stable physical left/right identity.

Needed stream metadata:

- `hand_id`
- handedness from tracker, treated as advisory
- tracking confidence
- `hand_count`
- missing/visible masks
- timestamp/frame index
- feature quality flags

### Features

The current CSV feature rows are useful but should be organized around reusable feature builders rather than one growing export path.

Near-term features:

- palm/wrist-centered landmark coordinates;
- hand-scale normalization;
- optional mirror/canonicalization experiments, with raw handedness retained;
- velocity, acceleration, signed displacement;
- peak horizontal velocity;
- direction consistency;
- no-hand / tracking-drop masks;
- two-hand relational features only after one-hand streams are stable.

Important: do not use raw dx sign as ground truth until mirrored-preview and camera-coordinate conventions are verified.

### Motion Activity Proposal

Before TCN v2, build a deterministic motion-event baseline from the features AirDesk already exports.

It should propose candidate stroke intervals using:

- hand-normalized displacement;
- peak velocity;
- direction consistency;
- background/low-motion valley;
- minimum and maximum duration;
- per-hand stream separation;
- duplicate suppression by peak identity, not only class cooldown.

This baseline is not the final model. It is a necessary diagnostic:

- if it works live, tracking/features are usable and the learned model is the weak link;
- if it fails live, the problem is lower in the stack: tracking, feature normalization, camera posture, or gesture definition.

Implementation notes after code review:

- prefer a small new module such as `airdesk/gestures/motion.py` over a broad
  package migration;
- consume `FrameFeatureRow` values from exported CSVs and from `FeatureRowStream`
  so replay and live paths share the same evidence;
- group rows by `hand_id`; no candidate window may cross from one hand stream to
  another;
- emit existing `GestureCandidate` objects first, with metadata for
  `window_start`, `window_end`, `peak_time`, normalized displacement, peak
  velocity, direction consistency, and a stable peak/evidence id;
- keep user-facing direction semantics explicit in docs and JSON. Raw camera
  `dx` sign remains diagnostic until preview/camera convention is verified.

### Event Decoder

The decoder should become a first-class package boundary.

It should consume either deterministic candidate evidence or model probabilities and output command-like events:

```text
GestureEvent(
  name,
  hand_id,
  start_time,
  peak_time,
  end_time,
  confidence,
  evidence,
)
```

Decoder rules:

- start from a motion/start peak or sustained intentional-motion evidence;
- integrate class evidence while active;
- emit on an end peak or background valley;
- split repeated same-class gestures only after a new peak/valley/start cycle;
- suppress duplicate emission from the same posterior/motion hill;
- keep cooldown short and evidence-based, not a long class-level lockout.

The existing decoder can be reused for the first slice, but review it while
adding the motion baseline. Two known risks should stay visible:

- global same-gesture suppression can hide distinct near-simultaneous events
  from different hands if the separation window is too broad;
- repeated same-direction swipes must be split by a new peak/valley/start cycle,
  not by waiting for a long class-level cooldown.

### Command Queue

Recognition should not directly execute actions. It should emit events into a queue. Mode/profile policy decides what actions are allowed.

Queue responsibilities:

- preserve event order across hands;
- allow repeated gestures when distinct peaks exist;
- support short combo/sequence decoding;
- expose dry-run logs for study and replay;
- enforce safety policy before execution.

Live desktop actions stay disabled for learned/DTW/dynamic swipes until replay evidence supports guarded execution.

## TCN V2 Shape

Do not build this first. Build the deterministic motion-event baseline first.

When the plan is reviewed and the baseline gives evidence, TCN v2 should be:

- shared across per-hand streams;
- causal;
- trained on continuous streams, not isolated windows as semantic truth;
- boundary-aware.

Preferred heads:

- `background` / `intentional_motion`
- `stroke_left` / `stroke_right` class evidence
- `start` boundary
- `end` boundary
- optional `hand_role` later: one-hand / both-hands / left physical / right physical, only if labels support it

Training targets should be derived from reviewed intervals, not raw prompt timing. Recovery/reset can be useful as decoder context but should not be a user-facing command target.

## Refactor Plan

This is a real architecture shift, so the next session should start with a review/refinement pass before implementation.

### Phase A: Planning And Boundaries

- Review `deep-research-report.md`, this plan, and current code.
- Confirm package/module boundaries.
- Decide what can be done without breaking existing commands.
- Update this plan if research or code inspection changes the direction.

Expected outcome: a final implementation checklist before code edits.

### Phase B: Recognition Boundary Cleanup

Likely package shape:

```text
airdesk/features/
  stream.py             per-hand rolling stream helpers
  normalized.py         canonicalized/masked feature builders

airdesk/recognition/
  evidence.py           common evidence/event dataclasses
  motion.py             deterministic motion-event proposal
  decoder.py            event decoder and duplicate suppression
  queue.py              command event queue / ordering
  tcn_v2.py             later model adapter, not first slice

airdesk/gestures/
  rules/templates/etc   keep current recognizers, adapt over time
```

This can be adjusted after code review. Avoid a giant rename if a smaller module boundary gets the job done.

Code-review adjustment: defer this package shape until after Phase C unless the
baseline needs it immediately. The first safe cleanup is a thin boundary around
motion evidence, not a wholesale migration of existing DTW/TCN/decoder modules.

### Phase C: Deterministic Motion-Event Baseline

Implement a per-hand swipe spotter that emits replayable events from current feature streams.

Minimum CLI/evaluation surfaces:

- replay candidate export on existing recordings;
- replay evaluation against existing labels;
- JSON output for candidates/events with evidence fields;
- live diagnostic preview only after replay output exists;
- tests for per-hand separation, repeated same-direction swipes, background rejection, and merged ordering.

Acceptance:

- gives a clearer replay/live diagnostic than current live TCN probabilities;
- preserves hand-scoped event ordering;
- allows repeated same-direction events when distinct peaks exist;
- rejects idle/background streams without leaning on desktop action policy;
- does not trigger desktop actions;
- reports false activations and repeated fires on replay.

Suggested first commands:

```bash
uv run airdesk gesture spot-motion --recording data/recordings/... --out data/evaluations/.../motion-candidates.json
uv run airdesk gesture evaluate-motion --recording data/recordings/... --labels data/labels/... --out data/evaluations/.../motion-summary.json
```

Names can change during implementation, but the surfaces should remain
replay-first and JSON-backed.

### Phase D: Targeted Live Calibration Slice

Only after the baseline exists, collect a tiny targeted slice if needed.

Suggested slice:

- 1-2 minutes real posture;
- single-hand left/right swipes;
- repeated swipes;
- idle face/desk movement;
- both hands visible sometimes;
- `--max-num-hands 2`;
- immediate feature export and replay scoring.

Do not collect broad combo data until this targeted failure mode is understood.

### Phase E: TCN V2

Only after Phase C/D evidence:

- add multi-head training targets;
- build start/end/intent labels from reviewed intervals;
- train/evaluate on continuous splits;
- compare against the deterministic motion baseline and DTW, not just window accuracy;
- keep it preview/replay only until event-level evidence is strong.

## Evaluation Metrics

Use interaction-style metrics, not clip accuracy alone:

- event precision/recall;
- false activations per idle minute;
- repeated-fire count;
- mean/median latency;
- sequence order score / Levenshtein for chained commands;
- per-hand stream correctness;
- background rejection under natural movement;
- replay/live agreement.

## Next-Session Stop Rule

The next session should not immediately dive into code unless the plan survives review.

Expected next-session flow:

1. Read docs and report.
2. Review current code boundaries.
3. Refine this plan.
4. Only then start Phase C or the smallest prerequisite refactor.
5. Keep live desktop actions disabled.

If the next agent finds the plan too broad, it should narrow Phase C rather than skipping directly to TCN v2.
