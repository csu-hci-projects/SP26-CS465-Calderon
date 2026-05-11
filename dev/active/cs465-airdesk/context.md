# AirDesk Context

## Date

2026-05-06

## Project Summary

AirDesk is a proposed CS465 final research project for an HCI / 3D user interfaces course.

The system uses a webcam and hand tracking to recognize mid-air gestures, then maps those gestures to desktop actions in Hyprland. The goal is not to replace keyboard and mouse for all work. The goal is to evaluate whether gestures are useful as a **secondary interaction channel** when traditional input is inconvenient, unavailable, undesirable, or physically costly.

The prototype is now being framed as a broader desktop-control system with command gestures, modeful cursor control, virtual keyboard/text entry, media controls, presentation controls, accessibility profiles, hybrid keyboard-plus-hand interaction, and optional depth/body sensing. The research paper should stay more focused than the product: evaluate a clean slice of the interaction rather than claiming every possible feature is solved.

## Updated Product Vision

AirDesk should become an OS-level spatial input layer:

> A configurable desktop control system where hands, keyboard, mouse, webcam, depth sensors, and desktop context blend into one profile-driven interaction layer.

This is closer to a browser extension for the desktop than a standalone demo:

- a background daemon owns tracking, recognition, modes, bindings, and logs
- a small overlay shows state and quick controls
- a larger control center configures profiles, thresholds, bindings, calibration, replay, and training

AirDesk should be pluggable from the beginning. MediaPipe may be the first hand-tracking backend, but the architecture should support alternatives such as OpenVINO, Kinect/depth input, recorded replay streams, and future hardware like Ultraleap.

## Current Hardware Context

Caden's current target setup:

- generic ThinkPad webcam, likely around 720p60
- Intel i7 CPU
- NVIDIA T550 laptop GPU
- Hyprland Linux desktop
- Kinect v2 available for experimentation

Implications:

- webcam remains the universal baseline
- NVIDIA/i7 makes local inference and model experiments realistic
- Kinect v2 is promising for depth, body presence, distance from desk, and coarse arm context, but should not be assumed to solve precise finger tracking
- recorded replay should be a first-class backend so tracking and recognition can be tuned without repeating gestures manually

## Course Fit

CS465 focuses on input devices, interaction techniques, multimodal interaction, and 3D user interfaces. AirDesk fits because:

- the input modality is spatial, mid-air, hand-based interaction
- the system communicates with a computer through embodied gestures rather than mouse/keyboard only
- it can be evaluated using HCI methods: task time, error rate, false activations, workload, fatigue, usability, and preference
- it touches accessibility and situationally induced impairments without requiring the project to make unsupported claims about permanent disability populations

## Motivation

Keyboard and mouse are excellent for precision tasks, but many desktop actions are coarse and command-like:

- switch workspace
- pause or rewind media
- move focus between windows
- toggle fullscreen
- dismiss a notification
- move a window to another workspace
- launch an application

These tasks may be useful to trigger without touching the computer when the user is:

- cooking with dirty hands
- painting or working with materials
- repairing hardware or a car
- wearing gloves
- standing away from the desk
- presenting
- dealing with temporary wrist strain
- dealing with chronic or recurring hand pain, arthritis, RSI, or limited reach
- otherwise situationally limited

This is the HCI framing: **situationally induced impairments and disabilities** (SIIDs), also called situational impairments. The project can also cite **ability-based design**, which argues that systems should adapt to users' available abilities instead of assuming one ideal interaction channel.

AirDesk is also accessibility-motivated. Users with arthritis, repetitive strain, hand pain, or limited reach should be considered during design. However, unless the study actually recruits those users, the paper should not claim validated accessibility benefits for those populations. Safer phrasing:

> AirDesk explores an alternate input channel that may reduce reliance on repetitive keyboard/mouse actions and may be relevant to users with temporary, situational, or chronic limitations.

## Working Title Options

- AirDesk: Mid-Air Gestures for Situationally Impaired Desktop Interaction
- AirDesk: Webcam-Based Gesture Control for a Tiling Linux Desktop
- Evaluating Mid-Air Gestures as a Secondary Control Layer for Desktop Window Management
- Hands Off the Keyboard: Gesture-Based Desktop Control Under Situational Impairment

Current preferred title:

**AirDesk: Mid-Air Gestures for Situationally Impaired Desktop Interaction**

## Important Design Stance

Do not claim that hand gestures are universally better than keyboard/mouse.

The more defensible claim is:

> Mid-air gestures may be useful for command-like, spatial, low-text desktop actions when traditional input is inconvenient, unavailable, painful, or undesirable.

This avoids the trap of trying to make hand tracking beat a keyboard shortcut at keyboard-shortcut things.

## Product Direction

Think of AirDesk as a modeful input system, not one giant always-on gesture recognizer.

Potential modes:

- **Command Mode**: open palm clutch, then one discrete command gesture. Best for workspace, focus, media, fullscreen, and cancel actions.
- **Cursor Mode**: pinch to take control of the cursor, move the hand to move the pointer, release to stop. Best for coarse pointing and occasional click/drag.
- **Mouse + Keyboard Mode**: cursor mode plus an on-screen virtual keyboard for limited text entry.
- **Media Mode**: a small safe vocabulary for pause/play, rewind, volume, and next/previous.
- **Window Manager Mode**: workspace switching, focus movement, fullscreen/floating, send-to-workspace, and later resize/move.
- **Accessibility Profile**: lower movement gain, larger dwell targets, longer confirmation windows, reduced pinch force, fewer destructive mappings.
- **Presentation Mode**: next/previous slide, pointer/highlight, zoom, fullscreen.
- **Hybrid Keyboard + Hands Mode**: keyboard/mouse handle precision and text while hand gestures handle spatial commands, window movement, mode switches, and large desktop actions.

The early prototype should prioritize Command Mode because it is safest and easiest to evaluate. Cursor Mode should be implemented as modeful pinch takeover, not as an always-on replacement for the mouse.

The broader product should not artificially narrow scope. The class study can evaluate a focused slice while the implementation is designed to grow into full hands-first or hybrid desktop control.

## Key Research Terms

- HCI
- 3D user interfaces
- mid-air interaction
- gesture interaction
- situational impairments / SIIDs
- ability-based design
- desktop window management
- tiling window manager
- multimodal input
- clutching / mode activation
- modeful interaction
- cursor takeover
- virtual keyboard
- Midas Touch problem
- gesture vocabulary
- false activation
- fatigue / "gorilla arm"

## Related Work Seeds

Use these as starting points, not as a final bibliography:

- MediaPipe Hands: On-device Real-time Hand Tracking  
  https://arxiv.org/abs/2006.10214

- Understanding Mid-Air Hand Gestures: A Study of Human Preferences in Usage of Gesture Types for HCI  
  https://www.microsoft.com/en-us/research/?p=163789

- Gesture Elicitation Studies for Mid-Air Interaction: A Review  
  https://www.mdpi.com/2414-4088/2/4/65

- Proceedings of the CHI 2019 Workshop: Addressing the Challenges of Situationally-Induced Impairments and Disabilities in Mobile Interaction  
  https://arxiv.org/abs/1904.05382

- Situationally-Induced Impairments and Disabilities Research  
  https://arxiv.org/abs/1904.06128

- Ability-Based Design  
  https://cacm.acm.org/research/ability-based-design/

- Hyprland IPC  
  https://wiki.hypr.land/IPC/

- Hyprland Dispatchers  
  https://wiki.hypr.land/Configuring/Dispatchers/

## Current Decision

Proceed with the Hyprland hand-control project rather than forcing the existing portfolio / point-engine work into the class assignment.

Reason:

- the point engine ideas were mostly output/art-style driven
- AirDesk is directly about input, interaction technique, user performance, and situational usability
- the prototype may remain useful after the class

## Current Implementation State

Sprint 2 established a working live and replay foundation:

- OpenCV camera probing can request width, height, FPS, and FOURCC.
- `/dev/video0` works at `640x480 @ 30 FPS` with MJPG when normalized to OpenCV camera index `0`.
- MediaPipe Tasks Hand Landmarker is backend zero and remains replaceable behind the tracking interface.
- Live preview is mirrored by default and shows landmarks, skeleton, bounding boxes, handedness/confidence, hand count, and gesture indicators.
- MediaPipe model path, hand count, and confidence thresholds are CLI-tunable.
- JSONL recording/replay and replay analysis are available.
- Command-mode policy, profile binding resolution, and dry-run runtime routing are implemented.
- Cursor mode now has an explicit `airdesk cursor run` command. Dry-run is default; `--execute` uses Hyprland `movecursor` for real cursor movement while pinch is held. Release exits cursor movement, `p` pauses/resumes, and `q`/`esc` exits. This older cursor model uses pinch as the cursor-move clutch; the crunch-time logic-control pivot should revise that so open/relaxed hand movement controls the pointer and pinch becomes click/scroll/drag. Real pointer button/scroll injection is still pending. The 2026-05-11 planning check found `hyprctl` installed, `/dev/uinput` writable by `caden`, no `ydotool`/`dotool`/`wtype`, and no Python `evdev` package installed.
- `airdesk label suggest` can bootstrap swipe labels by finding the strongest palm-motion window and applying phase/event labels for review. This is a labeling accelerator, not a final recognizer.
- `airdesk gesture calibrate --kind dtw` and `airdesk gesture evaluate --recognizer dtw --model ...` now provide a dependency-free personalized DTW/template baseline for replay evaluation.
- Sprint 4 batch `data/recordings/sprint4-swipes-001` has 24 local takes: 8 left swipes, 8 right swipes, and 8 normal desk-motion negatives. Generated labels/features/evaluations live under ignored `data/labels`, `data/features`, and `data/evaluations`.
- Rule recognizer result on that batch was poor: 0/16 positive swipe matches and high static-pose false activations.
- DTW calibrated on the same batch matched all 16 intended swipes, missed 0, produced 18 candidates total, 2 false activations, 0 repeated fires, and about 0.44 s mean latency. Negative recordings produced 0 DTW candidates.
- DTW holdout evaluation now exists via `airdesk gesture holdout-dtw`. On a deterministic `sprint4-swipes-001` split using takes 001-006 for train and 007-008 for test, DTW matched 2/4 held-out swipes, missed both held-out left swipes, produced 0 false activations on two held-out negatives, and had about 0.40 s mean latency on matched events.
- Holdout diagnostics show that loosening the left threshold enough to catch both held-out left swipes also introduces false activations. Treat this as a left-swipe/negative feature-separation problem, not a threshold-tuning win.
- An optional calibrated horizontal-displacement gate now exists for DTW. With `--negative-distance-margin 1.3 --min-palm-dx-fraction 0.65`, the same deterministic holdout matched 4/4 held-out swipes with 0 false activations and about 0.36 s mean latency. Because this was tuned after viewing the holdout, it needs fresh chained-session validation before live control.
- `airdesk gesture spot-dtw` now spots DTW candidates in unlabeled continuous recordings. On fresh chained recording `data/recordings/sprint4-chained-001/chained-left-right-swipes-001.jsonl`, gated DTW found 16 candidates, roughly matching Caden's "15-ish" swipe count and including back-to-back swipe clusters.
- `airdesk gesture score-sequence` now compares spotted DTW candidates with a remembered R/L order. On structured chained recording `data/recordings/sprint4-chained-002/chained-structured-swipes-001.jsonl`, expected sequence `R L R R L L R R L L` scored against detected sequence `R L R R L R R L` as 8/10 matched in order, 2 missed-or-wrong-order, and 0 extra-or-wrong-order.
- The causal TCN scaffold now has deterministic manifest/window building, optional PyTorch training, same-batch evaluation, and holdout evaluation. Same-batch TCN was optimistic: 16/16 matched with 1 false activation. Holdout TCN reproduced the plain-DTW weakness: 2/4 held-out swipes matched, both held-out left swipes missed, and 0 false activations.
- `airdesk gesture diagnose-features` now compares feature, timing, and tracking-quality summaries across that same split. On `sprint4-swipes-001`, held-out left swipes are weaker/slower than train-left examples (`palm_dx` about `0.181` vs `0.235`, normalized `palm_dx_per_hand_scale` about `1.387` vs `1.857`, max palm speed about `3.230` vs `5.163`), while label/frame alignment looks consistent at about one frame inside the event interval.
- Feature export now includes causal trailing-window swipe features: signed horizontal palm displacement, hand-scale-normalized displacement, peak absolute horizontal velocity, and direction consistency. DTW models remain backward-compatible with older saved feature vectors.
- With the new features, DTW on the same deterministic holdout can match 4/4 held-out swipes with 0 false activations using `--negative-distance-margin 0.75`, but this was tuned on existing evidence. TCN holdout still misses both held-out left swipes and adds 1 false activation. On the structured chained session, the looser `1.3` gated DTW variant with new features scored 9/10 in order with 1 extra-or-wrong-order detection, while the conservative `0.75` variant under-detected badly.
- Caden recorded `data/recordings/sprint4-chained-003/chained-structured-swipes-001.jsonl` with a 10-seconds-active / 10-seconds-rest protocol and coarse half-window event labels for sequence `R L R R L L R R L L`. Old gated DTW matched 7/10 event windows with 0 false activations and scored 8/10 in order. The looser window-feature gated DTW matched 8/10 event windows, scored 10/10 in order, but produced 3 extra-or-wrong-order sequence detections and 2 repeated fires. Conservative `0.75` DTW matched only 1/10 in the continuous stream. TCN matched 3/10.
- `airdesk gesture watch-tcn` now provides a live/replay TCN classifier preview. It displays both hand streams in one stable HUD line with per-hand left/right stroke probabilities, disables the unrelated static fist/pinch overlay, and prints stroke/gesture predictions by default while suppressing `background` and `recovery` terminal spam because recovery is an internal phase rather than a user-facing gesture. For two-hand testing it defaults to `--max-num-hands 2` and applies one shared checkpoint independently to each visible `hand_id` stream. This is an observation/debug tool only and does not trigger desktop actions.
- `airdesk gesture chart-record` now provides a structured "Guitar Hero for swipes" collection path. A compact chart such as `RR | rest | RL | rest | RRR` expands into an on-screen colored chart HUD with a default 3-second lead-in, a smooth current-cue progress bar, and fixed upcoming cards for get-ready, stroke, reset, and rest prompt windows in the live preview, waits for space before countdown by default, records the replayable landmark stream, and writes coarse chart-derived stroke/recovery/event labels beside the recording. Combo blocks stay grouped as one active prompt so Caden can perform the swipes naturally inside the block. These labels capture prompt timing and should still be treated as weak labels until reviewed.
- May 2026 two-hand pivot: chart combo collection exposed a design flaw. Live collection defaulted to `max_num_hands=1`, and feature export used only `frame.hands[0]`, so alternating-hand or both-hands-visible combo recordings can be contaminated. If one hand remains tracked in frame, the other hand's gesture may not become active. The feature exporter now emits one row per visible hand with independent per-`hand_id` motion history and one background row when no hands are visible. DTW/live-DTW and TCN dataset windows now score hand-scoped streams, and event decoding decodes hand streams before merging events with cooldown suppression. The chart recorder now defaults to `--max-num-hands 2`. The `sprint4-gpu-swipes-002-structured` combo recordings were deleted and should be recollected only after this path passes full replay checks. The `sprint4-gpu-swipes-002-singles` batch remains local but should be treated as legacy/single-hand-only until reviewed or recollected with two-hand background/rest conditions.
- May 2026 shared per-hand TCN check: Caden collected two new two-hand chart batches, `data/recordings/sprint4-gpu-swipes-003-two-hand` and `data/recordings/sprint4-gpu-swipes-004-two-hand-extra`, producing 25,795 local feature rows across 10 charts. The recommended architecture remains one shared TCN checkpoint applied independently to each `hand_id` stream, followed by event decoding/merge/cooldown. Training separate tracker-slot models is deferred because `hand-0`/`hand-1` are tracker streams, not stable physical left/right identities.
- The first two-hand TCN replay check exposed a labeling issue rather than a model-family issue. Plain prompt-time labels would assign 856 stroke windows across 003+004, including stationary visible hands. A new TCN manifest option, `--target-assignment motion-gated`, assigns prompted stroke/recovery labels only when the per-hand stream has enough recent hand-normalized motion; it deliberately gates on motion energy, not raw dx sign, because mirrored preview/raw camera conventions made direction sign brittle across the new batches. With `--feature-preset stream-invariant --target-mode phase --target-assignment motion-gated`, 003+004 builds 4,081 windows: 3,543 background, 139 stroke_left, 170 stroke_right, and 229 recovery.
- Held-out shared per-hand TCN evidence is improved but not live-ready. Training on 003 and evaluating decoded events on 004 matched 27/48 intended events, missed 21, produced 40 candidates, 11 false activations, 4 repeated fires, and about 0.85 s mean latency. Per gesture: swipe_left 12/24 matched with 3 false activations; swipe_right 15/24 matched with 8 false activations. Raw non-decoded phase windows still score 0/48 because `stroke_left`/`stroke_right` phase predictions must be decoded/mapped into `swipe_left`/`swipe_right` events. The all-data preview checkpoint `data/models/gestures/tcn-sprint4-003-004-two-hand-motion-gated.pt` is a diagnostic model only, not a live action model.
- Caden's first live test of the recovery-inclusive phase checkpoint produced mostly high-confidence `recovery`, which is an internal phase rather than a gesture. `--target-mode phase-stroke` now trains only `background`, `stroke_left`, and `stroke_right`; recovery labels become background. On the same 003-to-004 held-out split, this cleaner target mode scored 26/48 matched, 22 missed, 47 candidates, 17 false activations, 4 repeated fires, and about 0.93 s mean latency. The all-data phase-stroke preview checkpoint is `data/models/gestures/tcn-sprint4-003-004-two-hand-motion-gated-phase-stroke.pt`.
- Caden then saw live `dx` exceed 0.50 while stroke probabilities stayed flat, so the live miss is not only a too-high motion gate. A more sensitive `phase-stroke` model trained with `--motion-gate-min-dx-per-hand-scale 0.20` improved the 003-to-004 held-out split to 37/48 matched, 11 missed, 59 candidates, 18 false activations, 4 repeated fires, and about 0.98 s mean latency. The all-data preview checkpoint is `data/models/gestures/tcn-sprint4-003-004-two-hand-motion-gated020-phase-stroke.pt`; it is more sensitive but likely noisier.
- `airdesk gesture diagnose-tcn-events` now writes detailed decoded TCN failure reports. On the 003-trained / 004-tested split, diagnostics showed most misses had nearest same-gesture candidates too early or too late rather than absent, and increasing match tolerance from 0.5 s to 3.0 s raised matches from 27/48 to 36/48 while false activations only fell from 11 to 9. Interpretation: many chart labels remain prompt-timing labels, not exact gesture timing; improving active-hand/timestamp labels is more useful than only sweeping decoder thresholds.
- `airdesk gesture refine-chart-labels` now writes non-destructive experimental label copies aligned to nearby per-hand motion peaks and a JSON report for review. Do not use these refined labels as training truth yet. On the same 003-train / 004-test split, 0.75s padding changed 92/100 events and scored 22/48 with 31 false activations; a stricter 0.75 motion-score pass changed 64/100 events and scored 16/48 with 28 false activations. The result says naive nearest-motion-peak relabeling is too noisy in combo/repeated sections, even though the report is useful for finding labels that need human review.
- Separate physical-hand TCNs remain deferred. The current live/dataset path already gives each visible hand one responsibility by splitting rows by `hand_id` and running the shared checkpoint per stream. Training separate `hand-0`/`hand-1` checkpoints would only be justified after adding stable physical-hand identity labels, because MediaPipe tracker ids can swap and are not guaranteed to mean left hand vs right hand.
- Caden added `deep-research-report.md`, which supports a larger recognizer architecture pivot: keep a causal temporal model as a possible core, but move from sliding-window classification to continuous gesture spotting with per-hand normalized streams, motion/activity proposals, boundary-aware event decoding, and a command queue. The planning entrypoint is now `dev/active/cs465-airdesk/recognition-v2-plan.md`; the plan review is complete, and the first deterministic motion baseline is implemented.
- Recognition V2 first implementation slice is now in place. `airdesk/gestures/motion.py` adds a deterministic per-hand motion-event baseline over existing `FrameFeatureRow` streams. `airdesk gesture spot-motion` writes replay JSON candidates with hand id, peak timing, normalized displacement, peak velocity, direction consistency, and a stable evidence id. `airdesk gesture evaluate-motion` evaluates those events with the same intended/matched/missed/false-activation summary used by rule, DTW, and TCN evaluation. This is replay/diagnostic tooling only and still does not trigger desktop actions.
- First bounded replay check: on all 24 `sprint4-swipes-001` recordings, the default raw-positive-dx direction mapping matched 0/16 intended events and produced 12 false activations, confirming the mirrored-preview/raw-camera direction concern. Flipping with `--positive-dx-gesture swipe_left` improved only to 5/16 matched with the same 12 false activations. Raising `--min-dx-per-hand-scale` to 1.0 reduced false activations to 5 but dropped matches to 3/16. On `sprint4-chained-003`, the default mapping matched 4/10 with 0 false activations but 3 repeated fires; flipped mapping matched 0/10. Interpretation: the baseline is a useful diagnostic, not a control recognizer. Motion energy alone is too loose for negatives and still misses weak left swipes.
- Motion-baseline diagnostics now exist in `spot-motion` JSON. Add `--labels` to annotate the strongest per-hand motion rows with label phase/event context and rejection reasons. Focused replay inspection showed `normal-desk-motion-negative-007` has background lateral motion strong enough to pass the baseline (`dx_per_hand_scale` about `1.5-1.7`, direction consistency `1.0`), while `swipe-left-positive-007` has label-time motion far below the default displacement gate (`dx_per_hand_scale` about `0.28`) after a tracking dropout resets the rolling motion window. `swipe-right-positive-007` confirms the raw-camera sign convention: with flipped mapping its negative raw dx maps to the user-facing `swipe_right` label. Interpretation: the current blocker is not one thing. Direction convention must stay explicit, but the larger reliability problems are negative-motion rejection/intent evidence and weak-left/tracking-continuity behavior.
- `airdesk benchmark` now reports timing slices for live MediaPipe runs: camera read, color conversion, MediaPipe inference, normalization, preview draw, and total loop time.
- The T550 GPU path is now opt-in through `scripts/airdesk-nvidia-mediapipe-wayland ... --hand-delegate gpu`. Plain `--hand-delegate gpu` can still land on Intel/Mesa EGL under Hyprland; the launcher selects the NVIDIA GLVND EGL vendor and Wayland EGL platform before Python starts. The success signal is a MediaPipe log line containing `OpenGL ES 3.2 NVIDIA` and `NVIDIA T550 Laptop GPU`.
- Short bounded smoke evidence on 2026-05-06: CPU delegate inference averaged about `16.84 ms`; plain GPU-on-Intel averaged about `13.60 ms`; T550 GPU via the launcher averaged about `4.17 ms`. Capture is still camera-paced around 30 FPS, so the practical benefit should be judged with hand-in-frame fast-swipe tracking continuity, not FPS alone.

Current next step:

> Crunch-time pivot: stop trying to turn the all-IPN/TCN gesture recognizer into
> the live command layer. Learned/DTW/motion recognizers should stay in
> preview/replay/evaluation. The next implementation should build a deterministic
> landmark-logic control path: stable open palm, fist, sideways open palm, pinch
> taps/holds, palm zones, hold timing, cooldowns, and a rolling combo buffer over
> stable pose-transition events. Use that grammar to recreate the practical
> mouse/window-manager loop: move pointer, left/right click, scroll, open the
> launcher, switch workspaces, move the focused/window-under-cursor to adjacent
> workspaces, and close the active window through a deliberate combo. Keep all
> execution dry-run-first with visible overlay feedback and JSONL logs.

2026-05-11 logic-control architecture decision:

- Do not wire learned all-IPN heads to live Hyprland actions for the class demo.
- Build a "mid-air mouse plus window manager" grammar from direct MediaPipe
  landmark facts.
- Keep this as a separate live-control lane, not as another feature stuffed into
  the old learned/dynamic `gestures` stack.
- Prefer a new `src/airdesk/control/` package and `airdesk control run` command.
  Keep `airdesk gesture ...`, old `airdesk run`, and old `airdesk cursor run`
  stable as diagnostic/legacy surfaces until the new control runtime is proven.
- Emit stable pose events only after debouncing/hold confirmation; do not treat
  every frame as a command event.
- Keep the last about four stable pose events per hand in a two-second combo
  buffer.
- Match combos same-hand by default, consume matched events, and apply cooldowns.
- Avoid grammar overlap: `fist` alone is window-grab/hold state; close window is
  a deliberate combo such as `open_palm -> fist -> open_palm`.
- Show `Seeing`, `Combo`, `Armed`, `Target window`, `Executed`, and `Suppressed`
  states in preview/dashboard rather than relying on terminal text.

MVP grammar candidate:

- Open/relaxed hand in cursor mode: move cursor through Hyprland `movecursor`.
- Index pinch tap: left click through a future input target.
- Thumb/middle pinch tap: right click through a future input target.
- Index pinch hold plus vertical movement: scroll through a future input target.
- Sideways open palm held left/right: `hyprctl dispatch workspace -1` / `+1`.
- Fist held center: arm window move and show active window title.
- Fist moved left/right zone: `hyprctl dispatch movetoworkspace -1` / `+1`.
- Open palm -> sideways open palm: `hyprctl dispatch global caelestia:launcher`.
- Open palm -> fist -> open palm: close active window via
  `hyprctl dispatch killactive`, with visible close-armed feedback.

Cleanup/separation rule for the next implementation:

- The new control runtime must not import `gestures.dtw`, `gestures.motion`,
  `gestures.learned_filter`, TCN modules, or IPN helpers.
- Shared landmark math may be extracted into `src/airdesk/poses/` or kept first
  in `src/airdesk/control/poses.py`.
- Leave `StaticHandPoseRecognizer` compatible for older tests and runtime paths;
  do not do a broad module move before the MVP works.
- Treat large gesture/model files as parked future work, not dead code to delete
  during the crunch.

Current learned-recognition filter update:

- `watch-tcn-v2` now has mode-aware custom-head filtering for all-IPN checkpoints.
- Modes are diagnostic only: `command` enables the lateral throw proxy heads,
  `cursor` enables click/double-click plus zoom heads, `zoom-media` isolates
  zoom heads, and `all-ipn` / `--debug-all-heads` keeps the filtered debug view.
- IPN point heads (`ipn_b0a` / `ipn_b0b`) are suppressed from learned-head
  preview/replay because they are noisy and redundant with direct MediaPipe
  finger/pose logic if pointing is needed later.
- The plain-language "Recognized" callout now requires an enabled head,
  per-head/default threshold, top-vs-runner-up margin, short persistence, and
  per-hand cooldown. Suppressed top heads remain visible in dashboard/log
  diagnostics.
- Replay the same policy with `airdesk gesture replay-tcn-v2-log` before live
  tuning. On
  `data/logs/live-ipn-all-tcn-v2-calibration-20260511-122007.jsonl`, default
  command-mode filtering over 328 predictions emitted 2 diagnostic recognitions,
  both `ipn_g05` / Throw left; the dominant false-fire heads (`Throw up`, `Open
  twice`, `Zoom out`) were suppressed by mode.

Current V2 feature contract:

- `stream-invariant-v2` is the default preset for new V2 classifier manifests.
- It excludes absolute `palm_x`, `palm_y`, `palm_z`, raw `palm_vx` /
  `palm_vy` / `palm_speed` / acceleration, raw `palm_window_dx`, raw
  `palm_window_peak_abs_vx`, `hand_scale`, `hand_count`, and unscaled
  finger/pinch distances or velocities.
- It includes timing and quality/mask fields (`dt`, `tracking_present`,
  `confidence`), hand-scale-normalized palm motion, hand-scale-normalized
  trailing displacement/peak velocity, direction consistency, palm-centered
  hand-scale-normalized index/pinch geometry, and simple finger-count shape
  features.
- The older `stream-invariant` preset remains available for compatibility with
  old replay/regression work, but it should not be the default for the clean V2
  collection pass because it still feeds raw image-space motion and scale.

Current recognition strategy update:

- Treat gestures as atomic events first. The TCN should learn fast
  `stroke_left` / `stroke_right` / boundary evidence, not classes such as
  `right_right_left`.
- Combos should be decoded in a second command-grammar layer over emitted
  atomic events, for example `R R L` inside a short time window.
- Public datasets are now a serious branch to explore. IPN Hand is the first
  candidate because it is continuous and includes thousands of hand-gesture
  examples plus natural non-gesture motion. Jester is much larger and webcam-like
  but mostly clip-classification shaped. The first IPN experiment maps only
  IPN `G05 Throw left` / `G06 Throw right` into AirDesk's existing left/right
  atomic evidence labels as a lateral-motion proxy. It is not an AirDesk swipe
  dataset, and AirDesk-only / hybrid comparisons should happen only after the
  IPN-only result is understood.
- AirDesk recordings remain the authority for pass/fail because the final task
  is Hyprland desktop control under Caden's camera/setup, not benchmark accuracy
  on public videos.

Current TCN v2 implementation state:

- `airdesk gesture build-tcn-dataset --target-mode v2-evidence` keeps windows as
  causal per-hand compute context but stores per-frame evidence targets rather
  than one semantic argmax label for the whole window.
- The v2 evidence heads are `intentional_motion`, `stroke_left`, `stroke_right`,
  `start`, and `end`. Recovery/reset is not a user-facing command target.
- `airdesk gesture train-tcn-v2` trains an optional PyTorch sequence-evidence
  model with one shared checkpoint shape over hand-scoped streams. The current
  v2 default is a residual dilated causal TCN: `hidden_channels=32`, `levels=3`,
  `kernel_size=3`, `dropout=0.10`, per-frame layer normalization, and two
  causal convs per residual block. At 30 FPS this gives about a 29-frame / roughly 0.9-second
  receptive field, which is much closer to the target swipe duration than the
  earlier shallow scaffold.
- V2 training now uses weighted/focal BCE instead of plain unweighted BCE.
  Positive weights are computed per evidence head, capped, and multiplied for
  sparse `start` / `end` heads. Checkpoints store the positive weights,
  per-head calibration thresholds, per-head precision/recall/F1 metrics,
  receptive-field metadata, and schema version `2`. Schema-1 v2 checkpoints
  still load for replay compatibility.
- `airdesk gesture evaluate-tcn-v2` maps evidence through the existing replay
  event decoder, but `start` and `end` are no longer passive metadata: `start`
  can boost a boundary-backed stroke into activation, and `end` suppresses
  stroke scores / raises background for release. It is still replay/evaluation
  tooling only. It supports `--early-match-tolerance-seconds` for causal peaks
  that fire slightly before hand-labeled event starts.
- `airdesk gesture evaluate-tcn-v2-heads` is the held-out metric for custom
  evidence-head manifests such as all-IPN. It scores each evidence head on the
  final frame of each causal window and writes per-head precision/recall/F1,
  macro/micro summaries, and a gesture-head confusion table. Use this for the
  all-IPN model; the older `evaluate-tcn-v2` command is still the AirDesk
  left/right swipe event-decoder view.
- `airdesk gesture diagnose-tcn-v2-events` now writes the detailed v2 replay
  diagnostics that the summary lacks: matches, misses, false activations,
  repeated fires, nearest candidate/event timing, decoder scores, and raw
  evidence heads.
- `airdesk gesture watch-tcn-v2` is now the safe live/replay preview for schema-2
  evidence checkpoints. It loads `causal_tcn_v2_evidence` models, applies one
  shared checkpoint independently to each visible hand stream, and now defaults
  to `--preview-layout dashboard`: a resizable OpenCV dashboard with the webcam
  view, landmark overlay, per-hand evidence bars, decoded-gesture history,
  emit-vs-peak delay, prediction/candidate counts, tracker timing summaries, and
  the motion features driving each score (`pos`, hand scale, normalized dx, peak
  x velocity, and direction consistency).
  The old compact camera overlay remains available with `--preview-layout camera`.
  `--camera-buffer-size` defaults to `1` on this command to reduce stale-frame
  backlog where OpenCV honors the setting. The command decodes candidates through
  the same start/end-aware event decoder and can write live prediction/candidate
  JSONL via `--events-out`; prediction events include the same motion-feature
  diagnostics and candidate events include top-level peak/emit/delay fields.
  Terminal candidate lines still show both emit time and peak time because live
  decoding waits for release/recovery evidence, but the dashboard is now the
  primary live feedback surface. It does not call runtime policy or action
  targets.
- V2 manifest summaries now include `evidence_frame_counts` so `start`/`end`
  and intent evidence are visible even when the collapsed window display target
  is `background`.
- Offline V2 evaluation decodes a deduplicated all-row evidence stream, keeping
  the prediction with the fullest causal context for each source/hand/timestamp.
- No-hand windows are now represented explicitly as `__no_hand__` so old
  tracking-drop/background rows do not accidentally load interleaved tracked-hand
  rows during training.
- First cleanup pass tightened the V2 evidence contract: no-hand/tracking-drop
  rows now stay background-only for decoder-facing evidence heads, and `start` /
  `end` boundary targets are assigned only to tracked intentional evidence inside
  the corresponding labeled event interval. This prevents weak/missing events
  from moving boundary targets onto unrelated later motion.
- The TCN v2 evidence target boundary now lives in
  `src/airdesk/ml/tcn_v2_evidence.py`. It owns the framewise
  `intentional_motion`, `stroke_left`, `stroke_right`, `start`, and `end`
  target construction, motion-gated weak-label checks, collapsed v2 display
  target selection, and evidence-count summaries. `src/airdesk/ml/dataset.py`
  now focuses on generic feature CSV loading, stream grouping, manifest
  serialization, and sliding-window construction.
- The TCN v2 train/evaluate boundary now lives outside the older window
  classifier surface: `src/airdesk/ml/tcn_v2_train.py` owns
  `prepare_tcn_v2_training_arrays`, `train_causal_tcn_v2`, and
  `predict_causal_tcn_v2_manifest`; `src/airdesk/analysis/tcn_v2.py` owns v2
  prediction dedupe, evidence-to-decoder frame mapping, and
  `evaluate_tcn_v2_manifest`. Public package exports remain stable through
  `airdesk.ml` and `airdesk.analysis`.
- Old `train-tcn` / `evaluate-tcn` / `watch-tcn` remain intact for the previous
  window-classifier scaffold and diagnostic live preview; use `watch-tcn-v2` for
  the schema-2 sequence-evidence model.
- The first pre-training TCN architecture cleanup is complete. It addressed the
  concrete weak spots called out in review: receptive field, residual/dilated
  block design, normalization/dropout, sparse boundary-head imbalance,
  weighted/focal BCE, calibration metadata, batched manifest prediction,
  checkpoint metadata/versioning, schema-1 compatibility, and decoder use of
  explicit `start` / `end` evidence. The next review should validate this
  architecture on old replay data and then decide whether the targeted V2 slice
  is ready to collect.

Current TCN v2 old-data regression:

- Current-code `sprint4-swipes-001` label-assigned v2 manifest:
  24 sources, 760 windows, source-frame evidence counts
  `intentional_motion=187`, `stroke_left=88`, `stroke_right=99`, `start=16`,
  `end=16`.
- `sprint4-swipes-001` motion-gated v2 manifest is much sparser:
  `intentional_motion=50`, `stroke_left=8`, `stroke_right=42`, `start=11`,
  `end=11`; use it as a diagnostic view, not current training truth.
- A 5-epoch quick model on the label-assigned swipes manifest trained cleanly
  (`train_frame_accuracy=0.983`, `validation_frame_accuracy=0.983`), but event
  replay at `activation/min_peak=0.35` decoded 0 candidates and matched `0/16`.
- A permissive `0.30` replay pass on `sprint4-swipes-001` matched `1/16`, missed
  `15`, produced `3` candidates and `2` false activations. Applying the same
  model to `sprint4-chained-003` matched `0/10`.
- The 25-epoch schema-2 replay model on the current manifest reached
  `train_frame_accuracy=0.990` and `validation_frame_accuracy=0.985`. Checkpoint
  metadata reports schema `2`, a 29-frame / about 0.94-second receptive field,
  weighted/focal BCE, validation calibration thresholds
  `intentional_motion=0.85`, `stroke_left=0.90`, `stroke_right=0.90`,
  `start=0.80`, `end=0.75`, and weak validation `start` F1 (`0.316`).
- At `activation=0.35`, `release=0.2`, `min_peak=0.35`, and no early-match
  tolerance, schema-2 replay matched `9/16` on `sprint4-swipes-001`, missed `7`,
  produced `22` candidates, `13` false activations, and `0` repeated fires.
  Diagnostics showed all 7 misses had strong same-gesture candidates just before
  label start (`-0.02` to `-0.22` seconds).
- With `--early-match-tolerance-seconds 0.25`, the same isolated replay scores
  `16/16`, `0` missed, `22` candidates, `5` false activations, `0` repeated
  fires, and about `0.065 s` mean latency. The remaining false activations are
  all in normal-desk-motion negative recordings.
- Source-level `airdesk gesture holdout-tcn-v2` now exists for the same
  filename-ordered split used by DTW/legacy TCN holdout. On `sprint4-swipes-001`,
  the schema-2 holdout trained on 18 files and tested on 6 files:
  `intended=4`, `matched=2`, `missed=2`, `candidates=7`,
  `false_activations=5`, `repeated_fires=0`, `mean_latency=0.214`.
  The held-out test files were takes 007-008 for left, right, and
  normal-desk-motion negatives. This confirms the strong same-source replay
  number was optimistic and not live-quality proof.
- The same swipes-trained model on `sprint4-chained-003` scores `8/10`, `0`
  false activations, `3` repeated fires, and about `1.75 s` mean latency; the
  two misses have nearest same-gesture candidates roughly `0.93 s` and `1.50 s`
  before the coarse label starts, so a small early-match tolerance does not
  explain them away.
- Interpretation: the architecture cleanup fixed underconfidence enough to make
  the evidence useful, but current old-data/live behavior is not good enough for
  action wiring. Targeted V2 collection is now justified only as a held-out
  train/test slice focused on twist/desk-motion negatives, weak/short swipes,
  near/far starts, left/right frame positions, tracking drops, and repeated
  swipes.

Current CLI cleanup state:

- The public entrypoint remains `airdesk.cli:app`.
- Offline TCN/model/evaluation commands now live in `src/airdesk/cli_tcn.py`.
- Replay/offline gesture diagnostic commands now live in
  `src/airdesk/cli_gesture_replay.py`: rule/DTW/motion evaluation, DTW
  calibration/holdout, `spot-dtw`, `spot-motion`, `evaluate-motion`,
  `decode-candidates`, `score-sequence`, and diagnostic chart-label refinement.
- Label and feature-export commands now live in `src/airdesk/cli_labeling.py`.
- Camera, Hyprland dry-run, and profile validation commands now live in
  `src/airdesk/cli_system.py`.
- Small shared CLI helpers live in `src/airdesk/cli_support.py`.
- Live preview/status formatting helpers now live in `src/airdesk/cli_live.py`.
- Live tracking/watch diagnostic command bodies now live in
  `src/airdesk/cli_live_commands.py`: `gesture watch-tcn`, `gesture watch-dtw`,
  `track`, `tune`, `view`, and `benchmark`.
- Runtime `airdesk run`, `airdesk cursor run`, preview pause wiring, cursor
  session event logging, dry-run action routing, and guarded Hyprland execution
  policy now live in `src/airdesk/cli_runtime.py`.
- Recording, replay summaries, prompted collection, chart prompt parsing, chart
  label writing, collection paths, and preview key handling now live in
  `src/airdesk/cli_recording.py`; shared tracker construction lives in
  `src/airdesk/cli_tracking.py`.
- Shared hand/no-hand feature stream helpers now live in
  `src/airdesk/feature_streams.py` and are re-exported through
  `src/airdesk/features/`; DTW, motion, TCN dataset building, and live TCN
  preview use the same grouping contract.
- `src/airdesk/cli.py` now owns app/subcommand wiring plus `doctor` and
  `analyze`. Keep `airdesk.cli:app` as the stable public entrypoint; new CLI
  surfaces should register through focused modules rather than rebuilding a god
  file.

Next review/refactor emphasis:

- The recording extraction chunk is complete: `cli.py` dropped from about 3,371
  LOC to about 1,971 LOC, recording/chart behavior moved behind a dedicated
  module boundary, and CLI help/behavior tests cover the extracted surfaces.
- The runtime/live-action boundary chunk is complete: `cli_runtime.py` owns
  runtime command registration, action-target selection, cursor execution, and
  preview pause controls; CLI tests cover default dry-run, explicit guarded
  execute, unsafe dispatcher refusal, and cursor dry-run event logging.
- The replay/offline gesture diagnostics chunk is complete: `cli_gesture_replay.py`
  owns old rule/DTW/motion evaluation and candidate/sequence utilities, while
  command names/help remain stable under `airdesk gesture ...`.
- The live diagnostic chunk is complete: `cli_live_commands.py` owns the live
  tracking/watch command bodies, while `cli_live.py` remains a smaller helper
  module for preview/status formatting shared by those commands.
- A dead-code scan with `vulture` found no confirmed removable production code
  in this pass; its high-confidence hits were required keyword names in runner
  protocols/test doubles (`check`, `capture_output`) and were left intact.
- Continue prioritizing real bugs, dead code, oversized files/functions,
  duplicated logic, unclear package ownership, missing tests, and anything that
  could make the targeted V2 recording session ambiguous or fragile.
- Caden explicitly wants the next context to push harder on structure and do
  what is right/best rather than stopping after cosmetic extraction. With the
  CLI command surfaces split and TCN v2 target/train/evaluate concerns isolated,
  the next best chunk should be a TCN architecture/training review and
  improvement pass before collection. If the model/training/evaluation contract
  needs a larger rewrite, prefer doing it now over preserving early scaffold
  code. Focused test cleanup can happen in the same lane if it removes friction
  without distracting from the recognizer architecture.
- Likely audit targets: `src/airdesk/cli.py`, extracted `cli_*.py` modules,
  `src/airdesk/ml/dataset.py`, `src/airdesk/ml/train.py`,
  `src/airdesk/analysis/evaluation.py`, `src/airdesk/features/`, and
  `src/airdesk/gestures/`.
- Do not collect the new V2 data until this cleanup pass is complete and tests
  pass, unless Caden explicitly changes the plan.

## Current Research Direction Update

Caden's live `watch-tcn` test showed that `swipe_left` works better than `swipe_right`, fast consecutive swipes are weak, and the model often needs the hand to reset before another gesture. This is consistent with the continuous-gesture literature: the hard problem is spotting gesture events inside an untrimmed stream, not classifying a clean fixed window.

The first continuous-spotting implementation pass is now in place:

- TCN manifests support `--feature-preset stream-invariant-v2`, which excludes absolute palm position, raw image-space motion, raw hand scale/count leakage, and unscaled finger/pinch geometry for new V2 classifier work. The older `stream-invariant` preset remains available for replay compatibility.
- TCN manifests support `--target-mode phase`, with default targets `background`, `stroke_left`, `stroke_right`, and `recovery`.
- Label files accept `recovery` / `reset` phases, and `airdesk label add-sequence` can create coarse ordered L/R stroke+recovery labels for chained sessions when exact timestamps are unavailable.
- `airdesk gesture evaluate-tcn --event-decoder` and `airdesk gesture decode-candidates` add a replayable hysteresis/peak/cooldown decoder over probability or candidate streams.

Initial replay evidence is still mixed:

- Current gated DTW with window features remains the best isolated holdout result: 4/4 matched, 0 false activations, 0 repeated fires.
- Current TCN with window features remains weak on the same holdout: 2/4 matched, 1 false activation, both held-out left swipes missed.
- TCN plus the new event decoder on that holdout matched 3/4 but introduced 2 false activations at permissive thresholds, so the decoder is not a free reliability win.
- A new stream-invariant phase TCN/event-decoder holdout smoke matched 2/4 with 1 false activation. The representation/target plumbing works, but the current local data is not enough to improve the model yet.
- Decoding the looser chained-session DTW candidates reduced repeated fires from 2 to 0 on `sprint4-chained-003`, but also reduced matches from 8/10 to 6/10. This is useful filtering behavior, not enough evidence for live control.

Updated stance:

- treat the current TCN as a scaffold, not the destination;
- make learned features less dependent on absolute frame position and distance from camera;
- train toward phase/event stream labels such as `background`, `stroke_left`, `stroke_right`, and `recovery`;
- add an event decoder with hysteresis, confidence peaks, cooldown, and repeated-fire suppression;
- use DTW/template and motion-energy gates as low-data personalization and candidate-filtering tools;
- consider graph/transformer memory only after labels and event decoding are in place.

See `dynamic-gesture-research.md` for the deeper research notes and source anchors.

## Public Dataset Pivot

The public-dataset survey is now recorded in
`public-dataset-survey.md`. Recommendation: start with IPN Hand, not Jester, for
the first public-data TCN v2 experiment. IPN is continuous, webcam/RGB based,
CC BY 4.0, includes natural non-gesture hand motion, and maps directly to the
current AirDesk atomic left/right swipe heads through `G05` / `G06`. Jester is
larger and useful later for clip-level pretraining, but it is less aligned with
AirDesk's immediate boundary/continuous-spotting problem.

The first local-data-only importer is available:

```bash
uv run airdesk public-data ipn-convert \
  --videos-dir data/public/ipn/videos \
  --annotations-dir data/public/ipn/annotations-download \
  --out-dir data/public/ipn/airdesk \
  --split train \
  --limit 1 \
  --manifest-out data/public/ipn/airdesk/tcn-v2-ipn-smoke-manifest.json \
  --mapping-out data/public/ipn/airdesk/ipn-airdesk-mapping.csv
```

It runs downloaded IPN MP4s through MediaPipe, writes AirDesk replay JSONL,
exports feature CSVs, maps only IPN `G05 Throw left` / `G06 Throw right` into
AirDesk's left/right atomic evidence labels, and can build a
`stream-invariant-v2` / `v2-evidence` manifest. That mapping is a proxy for
lateral throw motion, not a claim that IPN contains AirDesk swipe gestures. Other
IPN classes remain background/negative examples for the first left/right atomic
pass. Keep raw public datasets and generated artifacts in ignored `data/public/`.

2026-05-10 local acquisition update: the official IPN Hand Drive annotations
and all five video archives are now present under ignored `data/public/ipn/`.
Extraction produced 200 `.avi` videos in `data/public/ipn/videos/`, and a
bounded one-video smoke conversion succeeded with 26 segments, 2 mapped atomic
left/right throw-proxy segments, 1 replay recording, 1 feature CSV, and a
`stream-invariant-v2` / `v2-evidence` smoke manifest.

2026-05-10 IPN-only training update: the first IPN-only TCN v2 checkpoint was
trained from `data/public/ipn/airdesk-train/tcn-v2-ipn-train-manifest.json`
only; no AirDesk recordings were mixed in. The checkpoint is
`data/models/gestures/tcn-v2-ipn-train-atomic-10ep.pt`. Training covered 148 IPN
train videos and 296 mapped `G05` / `G06` throw-left/right segments. Held-out
IPN evaluation on 52 test videos / 104 mapped throw-left/right segments shows
the model learned left/right lateral evidence but the current decoder is too
chatty on other IPN gesture motion: `99/104` matched at permissive thresholds
with `2,183` false activations; `91/104` at default thresholds with `915` false
activations; `88/104` at stricter `0.80/0.45/0.80` thresholds with `247` false
activations; and `52/104` at `0.90/0.50/0.90` with `49` false activations.
Interpretation: useful pretraining signal, not a deployable AirDesk swipe model.

All-IPN correction: Caden correctly pointed out that the two-head IPN proxy
model should not be blamed for every false activation when 11 real IPN gestures
were being treated as background. The next public-data model should train all
non-`D0X` IPN gestures as named evidence heads. AirDesk now supports custom v2
evidence targets, and ignored all-IPN manifests have been generated from the
existing tracked IPN recordings without rerunning MediaPipe:
`data/public/ipn/airdesk-train-ipn-all/tcn-v2-ipn-all-train-manifest.json`
has 148 sources, 3,117 labeled gesture events, and 99,510 windows; the held-out
test manifest has 52 sources, 1,101 labeled events, and 35,706 windows. This is
the first all-IPN training target. A pre-launch review on this checkout
confirmed the generated labels match the official non-`D0X` annotation events,
`D0X` remains background, the manifests use `stream-invariant-v2`, and CUDA sees
the NVIDIA T550.

All-IPN training results: the original 0.8s-window `h64/l4` run finished in
about 12 minutes and wrote
`data/models/gestures/tcn-v2-ipn-all-80ep-h64-l4.pt`, with held-out
`gesture_macro_f1=0.505` and `gesture_micro_f1=0.695`. A controlled 1.6s-window
rerun using the same architecture improved held-out ranking and is the current
best checkpoint: `data/models/gestures/tcn-v2-ipn-all-w16-80ep-h64-l4.pt` with
`gesture_macro_f1=0.521`, `gesture_micro_f1=0.742`, top-1 gesture-positive
final-frame accuracy `0.757`, and top-3 `0.934`. A wider `h96` 1.6s run fit the
random validation split better but fell back to `gesture_macro_f1=0.503` on the
official held-out split, so simply widening the TCN is not the next best lever.
`start` / `end` stayed weak across all runs, which points to boundary target
design/tolerance rather than CUDA, caching, or basic architecture failure.
`airdesk gesture evaluate-tcn-v2-boundaries` now evaluates those sparse boundary
heads as temporally matched peak events. On the current best `w16_h64`
checkpoint, held-out boundary scores are about `start_f1=0.455` and
`end_f1=0.468` at ±0.5s, rising to about `0.53` at ±1.0s, so the signal is near
the annotations but not yet clean enough for event decoding.

2026-05-11 live all-IPN preview update: Caden tested the best all-IPN checkpoint
live with the no-action dashboard. The preview path now treats custom all-IPN
heads fairly: it shows named `ipn_*` evidence directly, disables the AirDesk
left/right swipe decoder for custom-head checkpoints, and displays
plain-language recognition callouts when the top custom head crosses
`--evidence-threshold`. The live test exposed the next blocker: the model is a
useful IPN prior but too eager as a global command recognizer. Open-hand and
ordinary hand presence can trigger `Throw up`; `Point one finger` / `Point two
fingers` are easy to do accidentally; `Open twice` and `Zoom out` can spike
without intentional commands. A parsed calibration run
(`data/logs/live-ipn-all-tcn-v2-calibration-20260511-122007.jsonl`) had 328
predictions; top heads above `0.80` were dominated by `Open twice` (28),
`Throw up` (26), `Throw left` (11), `Throw down` (9), and `Point one finger`
(8). This does not invalidate the fair held-out IPN evaluation, but it means
AirDesk needs mode-aware filtering, per-head thresholds/margins, and targeted
AirDesk negatives before any learned command binding.

Mode decision from the live test: do not keep all 13 IPN heads globally enabled.
Point/click/double-click heads belong in cursor mode, where point-like hand
postures are expected. Zoom heads belong in a separate zoom/media mode. The
global command mode should start with only robust command gestures after
AirDesk-specific negative testing; `Throw up`, `Open twice`, and `Zoom out`
should be disabled globally for now. Live desktop actions remain blocked.

Official IPN model note: the public IPN baselines are RGB/video models such as
ResNeXt/ResNet variants, not MediaPipe-landmark TCN checkpoints. They are useful
for comparison or a separate heavier RGB fallback experiment, but they are not a
drop-in replacement for the current landmark stream. The official continuous
recognition numbers are also much weaker than isolated classification, so they
do not remove the need for AirDesk-specific mode/negative handling.

## Current Roadmap

## Logic-Control Implementation Update

The first deterministic control slice is now in place:

- `src/airdesk/control/` owns primitive control pose facts, stable pose
  debouncing, a per-hand combo buffer, the first dry-run grammar, and a runtime
  loop.
- `airdesk control run` is registered as the side-by-side class-demo surface.
  It defaults to dry-run and writes JSONL events for what the system is seeing,
  stable pose events, combo state, requested actions, cursor moves, and action
  results.
- Existing `airdesk run`, `airdesk cursor run`, and `airdesk gesture ...`
  commands remain stable. The new control runtime does not import DTW, motion,
  learned-filter, TCN, or IPN modules.
- The first action boundary includes dry-run pointer button/scroll requests,
  guarded Hyprland dispatches for the demo grammar, and open-hand relative
  cursor movement through the existing cursor target abstraction.
- Pinch behavior is now split so quick index/middle pinch releases become
  left/right clicks, while index-pinch hold plus vertical palm motion emits
  dry-run scroll ticks and suppresses the tap.
- Control pose facts are prioritized to reduce overlap from noisy sideways/fist
  tracking: fist suppresses pinch artifacts, sideways-open-palm suppresses pinch
  artifacts, and clean pinch suppresses plain open-palm.
- Guarded Hyprland move/close actions can query the active window title so the
  status/log surface can show a target window before real testing.
- Live Hyprland testing showed move-window was too touchy when any side-zone
  fist could fire repeatedly. The current control grammar now requires center
  fist to arm move-window for a short window; a side-zone fist fires once and
  consumes the arm, while fist release or expiry returns to neutral.
- Live testing also showed sideways-hand shapes are too unreliable for workspace
  switching. Workspace now uses the same arming style: center open palm arms
  workspace switching briefly, then open palm in a side zone fires once and
  consumes the arm.
- The default side zones are pushed outward (`left <= 0.30`, `right >= 0.70`)
  and cursor gain defaults to `3.0`; both are exposed on `airdesk control run`
  for live tuning.
- Real pointer click/scroll injection is available through explicit
  `--pointer-execute` using `/dev/uinput`. Without that flag, pointer
  click/scroll remains dry-run even when Hyprland execution is enabled.

### Sprint 3: Pilot-Safe Live Command Mode

Build the live command loop:

- runtime event logs,
- continuous positive/negative recordings,
- intent-gated phrase recognizer foundation,
- rule/DTW dynamic gesture scaffolding,
- live command-state feedback,
- pause/kill switch,
- guarded opt-in Hyprland execution only if dry-run behavior supports it.

### Sprint 4: Gesture Dataset, Labeling, and Continuous Gesture Spotting

Turn recordings into evidence:

- label schema and CLI,
- feature extraction,
- DTW/template baseline and holdout evaluation,
- train and evaluate one small causal stream model over AirDesk features,
- rule/DTW fallback for safety, calibration, and debugging,
- LSTM/GRU deferred unless the TCN path fails,
- continuous-stream evaluation metrics,
- Sprint 5 recognizer decision for the pilot,
- position-invariant features, phase labels, and event decoding before live actions.

### Sprint 5: Study Tooling, Pilot, and Paper Evidence

Turn the prototype into class-ready evidence:

- pilot protocol,
- study event logging,
- baseline task workflow,
- CSV/summary export,
- Caden-only pilot,
- paper outline and limitations.
