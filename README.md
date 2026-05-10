# AirDesk

AirDesk is a CS465 HCI / 3DUI research project and personal computing prototype exploring webcam-based mid-air hand gestures as an OS-level spatial input layer for a Hyprland Linux desktop.

The project is motivated by **situationally impaired interaction**: moments when keyboard and mouse are inconvenient, unavailable, dirty, or physically costly, such as cooking, painting, repairing hardware, presenting away from a desk, wearing gloves, or managing wrist strain.

The long-term vision is broader than a small gesture demo: AirDesk should become a pluggable, profile-driven desktop control system where webcam, depth sensors, hand gestures, keyboard, mouse, and desktop context can blend into practical command, cursor, media, presentation, accessibility, and hybrid interaction modes.

Start here:

- `dev/active/cs465-airdesk/context.md` - current state and project framing
- `dev/active/cs465-airdesk/plan.md` - research plan, prototype scope, study design
- `dev/active/cs465-airdesk/architecture.md` - proposed system architecture and package boundaries
- `dev/active/cs465-airdesk/recognition-v2-plan.md` - current recognizer architecture pivot and next-session plan
- `dev/active/cs465-airdesk/research-notes.md` - technical research notes and current working positions
- `dev/active/cs465-airdesk/dynamic-gesture-research.md` - dynamic gesture recognition research and model strategy
- `dev/active/cs465-airdesk/sprint-0.md` - first implementation sprint plan and acceptance criteria
- `dev/active/cs465-airdesk/sprint-1.md` - live camera/tracking/recording sprint plan
- `dev/active/cs465-airdesk/sprint-2.md` - tracking-quality and dry-run command-mode plan
- `dev/active/cs465-airdesk/sprint-3.md` - pilot-safe live command-mode plan
- `dev/active/cs465-airdesk/sprint-4.md` - gesture dataset, labeling, and model-evaluation plan
- `dev/active/cs465-airdesk/sprint-5.md` - study tooling, pilot, and paper-evidence plan
- `dev/active/cs465-airdesk/tracking-samples.md` - local tracking sample protocol
- `dev/active/cs465-airdesk/tasks.md` - implementation and paper checklist
- `dev/active/cs465-airdesk/handoff-prompt.md` - prompt for a fresh agent
- `dev/active/cs465-airdesk/context-reset-prompt.md` - concise prompt for clearing context and restarting
- `dev/active/cs465-airdesk/next-session-prompt.md` - current copy-paste prompt for the next implementation session

## Development

AirDesk currently uses Python, `uv`, `ruff`, and `pytest`.

```bash
uv sync --dev
uv run airdesk --help
uv run pytest
uv run ruff check .
```

The public CLI entrypoint remains `airdesk.cli:app`; command ownership is split
across focused `src/airdesk/cli_*.py` modules so recording/collection,
label/features, system checks, TCN tooling, replay/offline gesture diagnostics,
runtime/live-action wiring, live tracking/watch diagnostics, and live preview
helpers can evolve without turning the entrypoint back into one large command
file. Dry-run action routing and guarded Hyprland execution are isolated in
`src/airdesk/cli_runtime.py` so the safety boundary is easy to audit.

Live camera/tracking support is optional:

```bash
uv sync --dev --extra live
uv run airdesk camera probe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG
uv run airdesk view --device /dev/video0
uv run airdesk tune --device /dev/video0 --max-frames 300 --show
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120
uv run airdesk benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --max-frames 120
scripts/airdesk-nvidia-mediapipe-wayland benchmark --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --max-frames 120
uv run airdesk track --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --no-show
uv run airdesk record --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-frames 120 --out data/recordings/sample.jsonl
uv run airdesk replay data/recordings/sample.jsonl
uv run airdesk analyze data/recordings/sample.jsonl
```

The MediaPipe backend uses the Tasks Hand Landmarker API and downloads the model bundle into ignored `data/models/` on first use.
MediaPipe tuning flags include `--model-path`, `--max-num-hands`, `--min-detection-confidence`, `--min-presence-confidence`, and `--min-tracking-confidence`.
Live TCN/DTW watch commands also expose `--hand-delegate cpu|gpu`; CPU remains the default until the local GPU path is benchmarked on the T550.
On Caden's Hyprland/Arch/T550 setup, plain `--hand-delegate gpu` can still initialize MediaPipe on the Intel/Mesa EGL renderer. Use `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu` to launch AirDesk with the NVIDIA GLVND EGL vendor and Wayland EGL platform selected before Python starts. A successful T550 path prints a MediaPipe log line containing `OpenGL ES 3.2 NVIDIA` and `NVIDIA T550 Laptop GPU`.
Most live CLI paths still default to one hand for lower latency, but this is no longer acceptable for swipe-combo data. May 2026 finding: one-hand tracking contaminated structured combo collection because the active tracked hand could block the other visible hand from becoming active. The chart recorder now defaults to `--max-num-hands 2`; use that two-hand path for future combo collection after running replay checks on the per-hand feature/export and recognizer path.

Offline ML training is also optional:

```bash
uv sync --dev --extra ml
```

If you need live camera work and offline ML in the same environment, sync both extras:

```bash
uv sync --dev --extra live --extra ml
```

Useful safe commands:

```bash
uv run airdesk doctor
uv run airdesk camera list
uv run airdesk camera probe --device /dev/video0
uv run airdesk profile validate configs/profiles/study-safe.toml
uv run airdesk replay tests/fixtures/replay-one-frame.jsonl
uv run airdesk run --backend replay --recording tests/fixtures/replay-one-frame.jsonl --profile configs/profiles/study-safe.toml --dry-run --events-out data/logs/replay-dry-run.jsonl
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --dry-run --show --events-out data/logs/live-window-manager-dry-run.jsonl
uv run airdesk collect --out-dir data/recordings/sprint4-smoke --label swipe-left-positive --label swipe-right-positive --reps 5 --duration 6 --countdown 3 --show
scripts/airdesk-nvidia-mediapipe-wayland collect --out-dir data/recordings/sprint4-gpu-swipes-001 --label swipe-left-positive --label swipe-right-positive --label normal-desk-motion-negative --reps 5 --duration 6 --countdown 3 --hand-delegate gpu --show
scripts/airdesk-nvidia-mediapipe-wayland record --out data/recordings/sprint4-gpu-swipes-002-structured/structured-a-right-heavy.jsonl --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --duration 82 --countdown 3 --wait-for-space --label structured-a-right-heavy --hand-delegate gpu --show --segment "0:10:R R" --segment "10:20:rest" --segment "20:30:R L" --segment "30:40:rest" --segment "40:50:R R R" --segment "50:60:rest" --segment "60:70:R L R" --segment "70:80:rest"
# Hold off on new chart-combo collection until the two-hand feature/evaluation path is tested on replay.
# Future combo collection should use --max-num-hands 2, which chart-record now defaults to.
uv run airdesk collection-summary data/recordings/sprint4-smoke
uv run airdesk label init data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --out data/labels/swipe-left-positive-001.labels.json
uv run airdesk label suggest data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --gesture swipe_left --out data/labels/swipe-left-positive-001.labels.json --apply
uv run airdesk label add-phase data/labels/swipe-left-positive-001.labels.json --phase stroke_left --start 2.4 --end 3.1 --gesture swipe_left
uv run airdesk label add-event data/labels/swipe-left-positive-001.labels.json --gesture swipe_left --start 2.4 --end 3.1
uv run airdesk label validate data/labels/swipe-left-positive-001.labels.json
uv run airdesk features export data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/features/swipe-left-positive-001.csv
uv run airdesk gesture evaluate --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-rule.json
uv run airdesk gesture calibrate --kind dtw --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/models/gestures/caden-dtw.json
uv run airdesk gesture evaluate --recognizer dtw --model data/models/gestures/caden-dtw.json --recording data/recordings/sprint4-smoke/swipe-left-positive-001.jsonl --labels data/labels/swipe-left-positive-001.labels.json --out data/evaluations/swipe-left-positive-001-dtw.json
uv run airdesk gesture build-tcn-dataset --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/models/gestures/tcn-sprint4-swipes-001-manifest.json
uv run airdesk gesture train-tcn --manifest data/models/gestures/tcn-sprint4-swipes-001-manifest.json --out data/models/gestures/tcn-sprint4-swipes-001.pt --epochs 25
uv run airdesk gesture build-tcn-dataset --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/models/gestures/tcn-v2-sprint4-swipes-001-manifest.json --feature-preset stream-invariant --target-mode v2-evidence
uv run airdesk gesture train-tcn-v2 --manifest data/models/gestures/tcn-v2-sprint4-swipes-001-manifest.json --out data/models/gestures/tcn-v2-sprint4-swipes-001.pt --epochs 25
uv run airdesk gesture evaluate-tcn-v2 --manifest data/models/gestures/tcn-v2-sprint4-swipes-001-manifest.json --model data/models/gestures/tcn-v2-sprint4-swipes-001.pt --out data/evaluations/sprint4-swipes-001-tcn-v2/summary.json
uv run airdesk gesture watch-tcn-v2 --model data/models/gestures/tcn-v2-sprint4-swipes-001-schema2-regression.pt --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-num-hands 2 --show --events-out data/logs/live-tcn-v2-preview.jsonl
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-tcn --model data/models/gestures/tcn-sprint4-003-004-two-hand-motion-gated020-phase-stroke.pt --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-num-hands 2 --hand-delegate gpu --show --profile-timing --confidence-threshold 0.35
uv run airdesk gesture watch-dtw --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-window-features-gated.json --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --show
uv run airdesk gesture watch-dtw --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-window-features-gated.json --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
scripts/airdesk-nvidia-mediapipe-wayland gesture watch-dtw --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-window-features-gated.json --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --hand-delegate gpu --show
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout.json
uv run airdesk gesture holdout-dtw --recordings-dir data/recordings/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-dtw-holdout/summary-gated.json --model-out data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --negative-distance-margin 1.3 --min-palm-dx-fraction 0.65
uv run airdesk gesture holdout-tcn --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-tcn-holdout/summary.json --model-out data/models/gestures/tcn-sprint4-swipes-001-holdout.pt
uv run airdesk gesture diagnose-features --features-dir data/features/sprint4-swipes-001 --labels-dir data/labels/sprint4-swipes-001 --out data/evaluations/sprint4-swipes-001-feature-diagnostics/summary.json
uv run airdesk gesture spot-dtw --recording data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl --model data/models/gestures/caden-dtw-sprint4-swipes-001-holdout-gated.json --out data/evaluations/sprint4-chained-001/gated-dtw-candidates.json
uv run airdesk gesture spot-motion --recording data/recordings/sprint4-chained-003/chained-structured-swipes-001.jsonl --out data/evaluations/sprint4-chained-003/motion-candidates.json
uv run airdesk gesture spot-motion --recording data/recordings/sprint4-swipes-001/swipe-left-positive-007.jsonl --labels data/labels/sprint4-swipes-001/swipe-left-positive-007.labels.json --out data/evaluations/sprint4-swipes-001/motion-left-007-candidates.json
uv run airdesk gesture evaluate-motion --recording data/recordings/sprint4-chained-003/chained-structured-swipes-001.jsonl --labels data/labels/sprint4-chained-003/chained-structured-swipes-001.labels.json --out data/evaluations/sprint4-chained-003/motion-summary.json
uv run airdesk gesture score-sequence --candidates data/evaluations/sprint4-chained-002/gated-dtw-candidates.json --expected-sequence "R L R R L L R R L L" --out data/evaluations/sprint4-chained-002/gated-dtw-sequence-score.json
uv run airdesk hyprland dry-run workspace r+1
uv run airdesk cursor run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --events-out data/logs/cursor-dry-run.jsonl
```

In `airdesk collect --show`, use the webcam preview itself: `space` starts the countdown, then `k` keeps, `r` redoes, `s` skips, and `q` quits.

Sprint 3 guarded real execution is opt-in and allowlisted:

```bash
uv run airdesk run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --profile configs/profiles/window-manager.toml --execute --allow-profile-execute --show --events-out data/logs/live-window-manager-execute.jsonl
```

Dry-run remains the default. Use `--pause-on-start` or press `p` in the live preview to suppress actions while tracking continues.

Cursor control is also dry-run by default. Real cursor movement is opt-in and uses Hyprland's `movecursor` dispatcher:

```bash
uv run airdesk cursor run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --execute --events-out data/logs/cursor-execute.jsonl
```

In cursor mode, pinch-hold activates relative cursor movement, releasing the pinch exits cursor movement, `p` pauses/resumes, and `q`/`esc` exits. Mouse click/drag injection is intentionally not enabled yet because this machine does not currently have a pointer-button injector installed.

`airdesk label suggest` is a bootstrap helper for dynamic gestures. It finds the strongest palm-motion window in a recording, applies a phase/event label, and should still be reviewed before training or evaluation.
`airdesk gesture chart-record` is the structured "Guitar Hero for swipes" collection path. It takes compact charts such as `RR | rest | RL | rest | RRR`, shows an on-screen colored chart HUD with a default 3-second lead-in, a smooth progress bar for the current cue, and fixed upcoming cards for get-ready/stroke/reset/rest prompts, records the replayable landmark stream, and writes coarse chart-derived stroke/recovery/event labels by default. Combo blocks such as `RRR` are shown as one active prompt so the swipes can happen at a natural pace inside that block. Chart recording now defaults to two tracked hands and should remain the recommended combo path; treat the generated labels as prompt-timing labels that may still need manual refinement before final training.
`airdesk gesture build-tcn-dataset` builds a dependency-free JSON manifest of sliding windows over exported CSV features. The first target is intentionally narrow: `background`, `swipe_left`, and `swipe_right`. The manifest stores feature-file paths, row ranges, target labels, and frame-count summaries; it does not train a model and does not add PyTorch to the base runtime. Feature export includes causal trailing-window motion summaries for swipe analysis, including signed palm displacement, displacement normalized by hand scale, peak horizontal velocity, and direction consistency.
Use `--target-mode v2-evidence` for the new TCN v2 surface. It keeps windows as causal compute context, but stores per-frame decoder-facing evidence heads: `intentional_motion`, `stroke_left`, `stroke_right`, `start`, and `end`. Manifest summaries include source-frame `evidence_frame_counts` so these heads remain visible even when the collapsed window target is `background`. No-hand/tracking-drop rows stay background-only for these evidence heads, and boundary targets are scoped to tracked intentional evidence inside the labeled event interval. This is the target shape for the next continuous data slice; old replay data should be used as regression coverage, not final V2 proof.
The frame-evidence target logic lives in `src/airdesk/ml/tcn_v2_evidence.py`; `src/airdesk/ml/tcn_v2_train.py` owns the optional sequence-evidence training/prediction path; `src/airdesk/analysis/tcn_v2.py` owns the replay decoder/evaluation glue. `src/airdesk/ml/dataset.py` is responsible for generic CSV loading, stream grouping, and manifest/window construction.
Use `--feature-preset stream-invariant --target-mode phase-stroke` to build the current live-preview stream-model manifest that excludes absolute `palm_x`, `palm_y`, and `palm_z` and targets only `background`, `stroke_left`, and `stroke_right`. The older `--target-mode phase` also trains a `recovery` class, but live testing showed that recovery can dominate the output without helping user-facing gesture recognition.
For live diagnosis, the current more sensitive preview checkpoint uses `--motion-gate-min-dx-per-hand-scale 0.20`; it improved 003-to-004 held-out matching from 26/48 to 37/48, but also raised false activations to 18. Treat it as a diagnostic model, not a live action model.
For two-hand chart data, use `--target-assignment motion-gated` so weak prompt-time stroke labels are assigned only to hand streams with enough recent hand-normalized motion. This keeps a stationary visible second hand from being trained as the prompted gesture while avoiding brittle raw-camera left/right sign assumptions from mirrored preview.
`airdesk gesture train-tcn` is an optional offline PyTorch training scaffold for that manifest. It saves a checkpoint with model weights, target mapping, feature columns, normalization stats, window settings, and training metrics. `airdesk gesture watch-tcn` is a live/replay classifier preview for inspecting TCN probabilities without triggering desktop actions; it now defaults to two tracked hands, applies one shared checkpoint independently to each visible `hand_id` stream, disables static fist/pinch overlay labels, shows both hand streams in one stable HUD line, and only prints stroke/gesture targets by default. Add `--include-recovery` or `--include-background` only when debugging raw phase probabilities. `airdesk gesture watch-dtw` is the equivalent live/replay diagnostic preview for calibrated DTW candidates, also without desktop actions. `airdesk gesture evaluate-tcn` evaluates checkpoint predictions with the same intended/matched/missed/false-activation summary used by rule and DTW evaluation. `airdesk gesture holdout-tcn` trains and evaluates on the same filename-ordered split shape as DTW holdout. Keep TCN in preview/replay/evaluation workflows until it beats gated DTW on held-out continuous sessions.
`airdesk gesture train-tcn-v2` trains the optional sequence-evidence TCN over the v2 manifest. The v2 trainer now defaults to a residual, dilated causal TCN with per-frame layer normalization, dropout, weighted/focal BCE for sparse evidence heads, calibrated per-head threshold metadata, receptive-field metadata, and schema-versioned checkpoints; older schema-1 v2 checkpoints still load for replay compatibility. `airdesk gesture evaluate-tcn-v2` converts evidence heads into the existing replay event decoder, using `start` evidence to help boundary-backed stroke activation and `end` evidence to force release/background instead of preserving boundary heads only as passive metadata. Offline v2 evaluation decodes a deduplicated all-row evidence stream so the window remains compute context rather than the semantic gesture unit. It also supports `--early-match-tolerance-seconds` so causal peaks that fire slightly before a hand-labeled event start can be counted intentionally instead of as both a miss and a false activation. `airdesk gesture watch-tcn-v2` is the no-action live/replay preview for schema-2 evidence checkpoints: it runs the shared v2 checkpoint per visible hand stream, shows compact intent/stroke/start/end evidence in the HUD, flashes a large preview banner for decoded swipes, decodes candidates with the same start/end-aware event decoder, avoids flushing half-finished live events before release evidence, and can write predictions/candidates to JSONL with `--events-out`. Terminal candidate lines include both the current emit time and the earlier peak time because live decoding intentionally waits for release/recovery evidence. This is still replay/evaluation/live-preview tooling only; it does not route learned swipes into desktop actions.
`airdesk gesture diagnose-tcn-events` and `airdesk gesture diagnose-tcn-v2-events` write per-event decoded TCN diagnostics for missed labels, false activations, repeated fires, nearest same-gesture candidates, hand ids, window timing, and score peaks. Use them before collecting more data when a summary count does not explain whether the problem is label timing, active-hand assignment, mirrored direction, or decoder thresholds.
`airdesk gesture refine-chart-labels` writes non-destructive experimental label copies aligned to nearby per-hand motion peaks, plus an optional JSON report. Current 003-train / 004-test evidence says not to adopt those refined labels as training truth yet: prompt labels scored 27/48 decoded matches, while refined labels with 0.75s padding scored 22/48, and a stricter 0.75 motion-score pass scored 16/48. Use the command for review candidates and label-timing diagnostics, not as the default TCN target source.
`airdesk gesture evaluate-tcn --event-decoder` and `airdesk gesture decode-candidates` apply the first replayable hysteresis/peak/cooldown event decoder. This is evaluation tooling only; it still does not wire learned or DTW swipes into desktop actions.
`airdesk gesture spot-motion` and `airdesk gesture evaluate-motion` provide the first Recognition V2 deterministic per-hand motion-event baseline. The baseline consumes the same exported/live `FrameFeatureRow` evidence as DTW/TCN, groups rows by `hand_id`, emits `GestureCandidate`-compatible `swipe_left` / `swipe_right` events with motion evidence metadata, and writes replay JSON before any live preview or desktop action path. Use `--positive-dx-gesture swipe_left` only when verifying a mirrored-preview/raw-camera direction flip.
`airdesk gesture spot-motion` also writes `motion_diagnostics`, a compact list of the strongest per-hand motion rows and rejection reasons. Add `--labels ...` when inspecting misses so diagnostic rows include `phase` / `event` label context.
`airdesk gesture diagnose-features` compares feature, timing, and tracking-quality summaries across the same filename-ordered holdout split. Use it before changing the feature export or recognizer thresholds; on `sprint4-swipes-001`, it confirms that held-out left swipes are weaker/slower than train-left examples while label alignment is roughly consistent.
The current learned-recognizer direction is continuous gesture spotting, not fixed-window classification. Use `dynamic-gesture-research.md` as the source of truth: the next model work should prioritize position-invariant features, phase/event labels, event decoding, and explicit non-gesture handling before any live desktop action wiring.
`airdesk gesture calibrate --kind dtw` builds a dependency-free personalized template model for replay evaluation; keep it in dry-run/evaluation workflows until false activations are low on negative recordings.
`airdesk gesture holdout-dtw` runs a deterministic train/test replay evaluation for a collection batch and writes closest-window diagnostics for rejected DTW matches. The first `sprint4-swipes-001` holdout matched 2/4 held-out swipes, missed both held-out left swipes, and produced 0 false activations on two held-out negative recordings, so the same-batch DTW result should still be treated as optimistic. An optional calibrated horizontal-displacement gate is available through `--min-palm-dx-fraction`; the first gated variant matched 4/4 held-out swipes with 0 held-out false activations, but it still needs a fresh chained recording before live-control use.
`airdesk gesture spot-dtw` runs a DTW model over an unlabeled continuous recording and exports candidate timestamps for review.
`airdesk gesture score-sequence` compares spotted candidates with a remembered R/L order when exact timestamps are not available.
`airdesk label add-sequence` can turn a coarse ordered sequence such as `R L R R L L` into evenly spaced stroke/recovery/event labels for chained sessions. Treat those as weak labels for replay scoring and later CTC-style work, not final hand-tuned timestamps.

Tests and replay do not require webcam, Hyprland, or MediaPipe access.
