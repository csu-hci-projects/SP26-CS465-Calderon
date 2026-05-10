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

Implementation update: the first narrow slice now exists. `airdesk/gestures/motion.py`
adds a deterministic per-hand motion-event baseline, and the CLI exposes
`gesture spot-motion` plus `gesture evaluate-motion` for replay-first JSON
candidate export and label evaluation. This does not add live desktop actions.

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

### V2 Classifier Feature Contract

The current collection-ready preset is `stream-invariant-v2`. It was added after
auditing the older `stream-invariant` preset, which excluded absolute
`palm_x/y/z` but still fed raw image-space motion, `hand_scale`, and unscaled
finger/pinch geometry into the model.

`stream-invariant-v2` excludes:

- absolute palm position: `palm_x`, `palm_y`, `palm_z`;
- raw projected palm motion: `palm_vx`, `palm_vy`, `palm_speed`, accelerations,
  raw `palm_window_dx`, and raw `palm_window_peak_abs_vx`;
- setup/scale leakage: `hand_scale` and `hand_count`;
- unscaled finger/pinch distances and velocities.

`stream-invariant-v2` includes:

- `dt`, `tracking_present`, and tracker `confidence`;
- hand-scale-normalized palm velocity, speed, acceleration, trailing
  displacement, and trailing peak x velocity;
- palm-window direction consistency;
- palm-centered hand-scale-normalized index-tip and pinch geometry;
- simple finger-count shape features.

Keep absolute position, raw scale, raw motion, and hand count in exported rows,
dashboard diagnostics, and JSONL logs. They are essential for debugging tracker
artifacts and false activations, but they are not gesture identity inputs for
the classifier unless a later evidence review explicitly reintroduces one.

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

The deterministic motion-event baseline now exists and has done its job: it made
the lower-level failure modes visible without hiding them behind a learned model.
Do not keep polishing that baseline indefinitely. The next implementation slice
should start the TCN v2 data/model surface while preserving the motion baseline
as a replay regression and proposal diagnostic.

TCN v2 should be:

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

Training targets should be derived from reviewed intervals, not raw prompt timing.
Recovery/reset can be useful as decoder context but should not be a user-facing
command target.

Important nuance: "continuous" does not mean no rolling context. The model can
still consume causal context windows/receptive fields. The difference from the
old scaffold is that a window is compute context, not the semantic gesture unit.
The output should be frame/event evidence for a decoder, not one argmax label for
an arbitrary clip.

### Atomic Gestures And Combo Grammar

The TCN should not learn every command sequence as a separate class. AirDesk
should train the model to spot atomic gesture evidence quickly:

```text
intentional_motion
stroke_left
stroke_right
start
end
```

The decoder should emit a stream of atomic events such as:

```text
R, R, L
```

Then a second command-grammar layer can interpret short histories:

```text
R R L within 1.5s -> optional combo command
```

This keeps repeated same-direction swipes possible, avoids class explosion, and
lets the system feel like it supports combos without forcing the TCN to relearn
`swipe_right` inside every possible sequence. `chart-record` already writes
atomic `swipe_left` / `swipe_right` event labels for blocks such as `RR` or
`R L`; combo labels should remain a command-layer concept unless a future
experiment proves an end-to-end combo class is necessary.

### Public Dataset Branch

Public hand-gesture datasets may help AirDesk learn broader atomic motion
priors, but they should not replace AirDesk recordings as the pass/fail gate.
The next session should survey candidates before committing to one. Initial
notes:

- IPN Hand is the best first candidate because it is a continuous RGB hand
  gesture benchmark with natural non-gesture hand movements, real background /
  illumination variation, 50 subjects, 4k+ gesture samples, and 800k RGB frames.
  Its "throw left" / "throw right" classes are plausible stand-ins for AirDesk
  swipes, and other classes may map to future click/select/control ideas.
- Jester is useful for large-scale webcam-style motion pretraining: 148k short
  clips, 27 classes, and 1,376 actors. It is less directly aligned because it is
  mostly short clip classification rather than continuous desktop-control
  spotting.
- Other datasets may be better for specific goals. Do a small survey before
  downloading anything large.

Recommended experiment order:

1. Build an adapter that runs public videos through the same MediaPipe /
   `FrameFeatureRow` / `stream-invariant-v2` feature path.
2. Train a TCN v2 model on IPN-only atomic targets and inspect replay/live feel.
3. Compare against AirDesk-only on the new held-out V2 slice.
4. Try hybrid training only after the IPN-only importer is trustworthy:
   pretrain on public data, fine-tune on AirDesk, then evaluate on AirDesk
   source-held-out sessions.

Do not train an RGB-video model first. The useful question is whether public
datasets improve AirDesk's existing landmark-feature recognizer and event
decoder, not whether a separate video classifier can score a benchmark.

## Refactor Plan

This is a real architecture shift. The initial review/refinement pass is now
complete, and Phase C's first deterministic motion baseline exists. The next
session should move from Phase C into a scoped Phase E start: TCN v2
manifest/target/model plumbing plus a targeted continuous-data plan. Old replay
data stays useful as a regression suite, not as final proof of V2 quality.

### Phase A: Planning And Boundaries

- Review `deep-research-report.md`, this plan, and current code.
- Confirm package/module boundaries.
- Decide what can be done without breaking existing commands.
- Update this plan if research or code inspection changes the direction.

Expected outcome: a final implementation checklist before code edits.

Status: complete for the first slice. The current checklist is to inspect
motion-baseline false activations, repeated fires, weak-left misses, hand ids,
and direction metadata before any live preview.

Status update: the first inspection pass is complete. `spot-motion` now writes
label-aware `motion_diagnostics` rows that keep rejected near-misses visible.
The focused failures split into two main buckets: natural background motion can
look like clean lateral motion to the baseline, and weak-left examples can fall
below the displacement gate when tracking drops/reset the rolling motion window.
Direction convention remains explicit, but it is not the only blocker.

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

These surfaces are now implemented. Use their JSON output as regression evidence
while building TCN v2; do not require the motion baseline to become a live
recognizer before moving on.

### Phase D: Targeted Continuous Data Slice

Collect a targeted new continuous slice after the initial TCN v2 data/model
surface exists, so the recording protocol matches the model targets.

Suggested slice:

- repeated same-direction swipes such as `R R` and `L L`;
- alternating swipes such as `R L R L`;
- weak/tiny left swipes;
- natural desk motion negatives;
- hand enters/leaves frame and tracking-drop cases;
- near/far and left/center/right starts;
- both hands visible sometimes, with one hand resting;
- `--max-num-hands 2`;
- exact-ish labels or an explicit label-review workflow;
- immediate feature export and replay scoring.

Do not collect broad combo data before this targeted slice.

### Phase E: TCN V2

Next implementation slice:

- add a TCN v2 manifest/target shape with per-frame/event evidence targets;
- preserve one shared model applied independently to each `hand_id` stream;
- add boundary/intent targets without making recovery a user-facing command;
- evaluate on old replay data as a regression check against known failures;
- collect the targeted continuous slice above for the real V2 train/test split;
- compare against the deterministic motion baseline and DTW, not just window accuracy;
- keep it preview/replay only until event-level evidence is strong.

Status update: the first TCN v2 surface now exists. `build-tcn-dataset
--target-mode v2-evidence` writes hand-scoped causal context windows with
per-frame evidence heads: `intentional_motion`, `stroke_left`, `stroke_right`,
`start`, and `end`. `train-tcn-v2` trains the optional sequence-evidence model,
and `evaluate-tcn-v2` routes stroke evidence through the existing replay event
decoder while keeping intent/boundary evidence in metadata. This is still a
surface and regression harness, not proof of V2 quality.

Status update after the pre-training architecture review: `train-tcn-v2` is no
longer the early plain-conv scaffold. New v2 checkpoints use a schema-2 residual
dilated causal TCN with per-frame layer normalization, dropout, a default 29-frame
receptive field at `levels=3` / `kernel_size=3`, weighted/focal BCE for sparse
evidence heads, calibrated per-head threshold metadata, per-head metrics, and
batched manifest prediction. Schema-1 v2 checkpoints still load for old replay
compatibility. `evaluate-tcn-v2` now converts `start` evidence into a
boundary-backed activation boost and `end` evidence into release/background
pressure instead of using only stroke-derived scores. The next proof point is a
replay check on old regression data, not live action wiring.

Status update after the schema-2 replay check: the stronger TCN v2 checkpoint
is a real improvement over the first underconfident smoke. On
`sprint4-swipes-001`, the 25-epoch schema-2 model matched `9/16` at the old
`0.35/0.2/0.35` decoder settings; `diagnose-tcn-v2-events` showed the 7
"misses" were strong same-gesture peaks just before hand-labeled event starts.
The v2 evaluation path now supports `--early-match-tolerance-seconds`; with
`0.25`, isolated swipes score `16/16` with `5` false activations, all from
normal-desk-motion negatives. Applying the same model to
`sprint4-chained-003` scored `8/10` with `0` false activations but `3` repeated
fires and high mean latency. Interpretation: do not rewrite the TCN architecture
again right now, and do not collect broad combo data. The next gate is
negative-motion intent rejection plus repeated-fire/boundary timing, then a
targeted V2 continuous slice.

Status update after the live-safety review: schema-2 v2 now has a separate
no-action preview command, `airdesk gesture watch-tcn-v2`. It loads
`causal_tcn_v2_evidence` checkpoints, runs the shared model per visible
`hand_id` stream, defaults to a resizable dashboard with webcam landmarks,
per-hand intent/stroke/start/end evidence bars, decoded-gesture history,
emit-vs-peak delay, prediction/candidate counts, and tracker timing, decodes
candidate swipes with the same start/end-aware event decoder, and can write
prediction/candidate JSONL through `--events-out`. The old camera-only compact
overlay remains available with `--preview-layout camera`. The live decoder does
not flush open events before release evidence, so the preview is closer to
continuous runtime behavior than repeatedly decoding a truncated replay prefix.
This does not change the live-action stance: learned swipes still do not
dispatch desktop actions.

Status update after the first live feel-test and source-holdout check: same-source
old replay is not enough evidence. The new `airdesk gesture holdout-tcn-v2`
command trains/evaluates v2 on the same filename-ordered split shape as DTW and
legacy TCN holdout. On `sprint4-swipes-001`, the schema-2 source holdout trained
on takes 001-006 and tested on takes 007-008 scored `2/4` held-out swipes with
`5` false activations, despite train frame accuracy around `0.986` and validation
frame accuracy around `0.976`. Live wrist-twist false positives are plausible
from the older feature geometry: `stream-invariant` did not use absolute
`palm_x`, `palm_y`, or `palm_z`, but projected wrist rotation could still
perturb raw image-space motion, hand scale, and unscaled finger-relative
features. `stream-invariant-v2` removes those fields from classifier input while
keeping them visible in the dashboard and JSONL logs. Next data should be
targeted and held out, not broad combo collection.

### Public Dataset Training Aid

The public dataset survey is in `public-dataset-survey.md`. Use IPN Hand as the
first public-data experiment because it is continuous and contains natural
non-gesture motion. The importer exposed as `airdesk public-data ipn-convert`
turns downloaded IPN videos into normal AirDesk replay recordings, labels, CSV
features, and optional `stream-invariant-v2` / `v2-evidence` manifests.

For the first IPN model, keep the output heads unchanged:
`intentional_motion`, `stroke_left`, `stroke_right`, `start`, and `end`.
Map only IPN `G05` / `G06` to AirDesk `swipe_left` / `swipe_right`; leave other
IPN gesture classes as background/negative until AirDesk adds explicit heads for
click/select or push. Do not train combo labels from public data.

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
4. Use the `watch-tcn-v2` dashboard for no-action live feel-tests and log predictions/candidates plus motion features.
5. Use `holdout-tcn-v2` and TCN v2 diagnostics to keep train/test claims honest before changing thresholds.
6. Plan or collect the targeted continuous V2 slice above only as a held-out train/test slice with explicit wrist-twist and desk-motion negatives.
7. Keep live desktop actions disabled.

If the next agent finds the plan too broad, it should narrow TCN v2 to the
manifest/target/evaluation surface first rather than jumping straight to live
preview or broad collection.
