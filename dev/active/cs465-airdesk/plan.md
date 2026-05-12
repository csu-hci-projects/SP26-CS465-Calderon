# AirDesk Plan

## Research Question

Primary:

> Can a small, clutch-based mid-air gesture vocabulary provide usable secondary control for common desktop tasks under situational impairment, and how does it compare to keyboard/mouse interaction in speed, error rate, fatigue, and user preference?

Alternative tighter version:

> Which desktop window-management tasks are suitable for webcam-based mid-air hand gestures, and where do gestures break down compared to keyboard/mouse input?

Product-oriented question:

> How should a Linux desktop expose modeful hand-gesture controls for command actions, cursor takeover, and lightweight text input without making accidental activation or fatigue worse?

## Hypothesis

Expected outcome:

- keyboard/mouse will be faster for precise and text-heavy tasks
- gestures may be competitive or preferred for coarse commands when the user is away from the keyboard or unable to touch input devices
- a clutch gesture will reduce false positives and improve user confidence
- gesture control will introduce fatigue if overused, so the best design is secondary / opportunistic rather than primary / continuous
- modeful cursor control may be useful for occasional pointing but will likely be slower and less precise than a mouse
- virtual keyboard entry may be useful as a fallback, but should be evaluated separately from window/media commands

## Prototype Scope

Build a local daemon or app that:

1. Reads webcam frames.
2. Tracks hands using a pluggable backend.
3. Normalizes hand/body state into backend-independent data structures.
4. Recognizes gestures using rule-based recognizers first.
5. Applies debounce, confidence thresholds, cooldowns, mode state, and profile safety rules.
6. Sends commands to Hyprland using `hyprctl dispatch` through an action adapter.
7. Shows a minimal status overlay or log of the current gesture / command / confidence.
8. Records enough data for replay, debugging, study instrumentation, and future training.

The first implementation should be rule-based. Do not train a neural network until AirDesk has enough recorded/labeled data and rule/DTW fallback behavior to beat.

## Architecture Strategy

AirDesk should be built as a pluggable, profile-driven spatial input daemon.

Pipeline:

```text
camera / sensor
  -> capture backend
  -> tracking backend
  -> normalized hand/body state
  -> smoothing and calibration
  -> gesture recognizers
  -> mode/profile state machine
  -> action router
  -> Hyprland / media / cursor / overlay / logs
```

Primary runtime surfaces:

- **Daemon**: real-time tracking, recognition, modes, actions, logs.
- **Overlay**: active state, current mode, gesture feedback, fake cursor, quick pause/resume.
- **Control Center**: profiles, bindings, threshold tuning, calibration, replay, training, study sessions.

See `architecture.md` for package boundaries and open architecture questions.

## Tracking Strategy

Do not make AirDesk depend on one tracker.

Initial and future tracking backends:

- **MediaPipe Hand Landmarker**: fastest first implementation path, 21 landmarks, handedness, confidence, live-stream mode.
- **OpenVINO hand tracking**: possible alternative if MediaPipe is unstable on Linux/ThinkPad hardware.
- **Kinect v2 / libfreenect2**: depth/body/presence backend for distance, coarse arm gestures, segmentation, and context.
- **Recorded Replay**: first-class backend for debugging, recognizer iteration, and model training.
- **Ultraleap / Leap Motion**: optional future high-quality hand/finger tracking hardware.

The normalized hand/body state should hide backend differences from gesture recognizers.

## Recognition Strategy

Recognition should evolve in layers:

1. **Rule-based primitives**: open palm, fist, pinch, point, swipe, push, hold, dwell.
2. **Intent-gated gesture phrases**: dynamic commands framed as preparation, stroke, commit, release, cooldown, or abort.
3. **Template / DTW fallback**: personalized motion gestures and custom paths for calibration and safety fallback.
4. **Personal ML models**: a small causal TCN trained from phase-labeled AirDesk logs as the first learned temporal recognizer.
5. **Alternative ML models**: LSTM/GRU, ST-GCN, Transformer, or public-dataset approaches only after the causal TCN path is tested.

The first hard problem is gesture spotting, not just classification:

- when the gesture starts
- when it ends
- whether it was intentional
- whether mode/profile state allows it
- whether confidence is high enough to execute a command

### Recognition V2 Pivot

May 2026 update: AirDesk is due for a recognizer architecture cleanup before more broad data collection. The first pre-training TCN v2 architecture cleanup is now complete: residual dilated causal blocks, normalization/dropout, weighted/focal evidence loss, sparse-boundary weighting, calibration metadata, schema-versioned checkpoints, batched prediction, and start/end-aware decoder scoring are in place. The stronger schema-2 same-source replay check improved isolated old swipes to `16/16` with a small causal early-match tolerance, but still produced `5` negative false activations; chained replay reached `8/10` with `3` repeated fires. A new source-level `holdout-tcn-v2` command shows the same old-data model family is still not live-quality: on `sprint4-swipes-001`, training on takes 001-006 and testing on takes 007-008 scored `2/4` held-out swipes with `5` false activations. The feature-contract audit is now complete enough for targeted collection: use `stream-invariant-v2`, which excludes absolute palm position, raw image-space motion, raw hand scale/count setup leakage, and unscaled finger/pinch distances while retaining those fields in logs/dashboard diagnostics. `watch-tcn-v2` now provides a no-action live/replay dashboard and JSONL logging path for schema-2 checkpoints, with per-hand evidence bars, decoded-gesture history, emit/peak delay, tracker timing, and motion-feature diagnostics visible without relying on terminal output. Keep learned swipes out of desktop actions; the next recognizer gate is targeted held-out V2 data plus explicit wrist-twist/desk-motion negatives, not live action wiring.

The current TCN work proved useful infrastructure, but the live tests showed that the implementation is still too close to sliding-window phase classification. The next architecture should treat TCN as one possible temporal core inside a continuous spotting system, not as the whole recognizer.

The new planning entrypoint is `recognition-v2-plan.md`.

Working stance:

- keep MediaPipe as backend zero, not project identity;
- keep one shared per-hand model/scorer applied independently to each `hand_id` stream;
- do not train separate `hand-0` / `hand-1` models unless stable physical-hand identity labels exist;
- stop threshold-sweeping the current TCN after live `dx` rose above 0.50 while stroke probabilities stayed flat;
- build a deterministic motion-event baseline first to prove the tracking/features path;
- then build a boundary-aware TCN v2 only if the plan and baseline evidence support it;
- keep learned and dynamic swipes in replay/preview only until event-level evidence supports guarded execution.

Target continuous-spotting shape:

```text
per-hand normalized feature streams
  -> motion activity proposal
  -> recognizer/scorer
  -> event decoder
  -> command queue
  -> mode/profile/safety policy
```

### Crunch-Time Logic-Control Pivot

2026-05-11 update: live all-IPN testing showed that the learned model is not
safe enough for a class-demo global command recognizer. The near-term AirDesk
pilot should pivot away from semantic learned gestures and toward deterministic
landmark logic that recreates the core mouse/window-manager affordances:
pointer movement, left/right click, scroll, launcher, workspace switching,
moving windows between workspaces, and closing windows.

This is not a rejection of the learned-recognition work. TCN/IPN remains useful
as a preview/evaluation/pretraining lane, but live desktop actions should now
come from simple observable pose/motion facts:

- thumb/index and thumb/middle pinch distances;
- stable open palm, fist, and sideways-open-palm poses;
- palm position relative to screen/camera zones;
- palm velocity and hold time;
- per-hand stable-pose transitions stored in a short combo buffer.

Target logic-control shape:

```text
MediaPipe landmarks
  -> per-hand primitive pose features
  -> stable pose debouncer
  -> pose transition events
  -> rolling combo buffer, max 4 events / about 2 seconds
  -> mode/action grammar
  -> guarded Hyprland and input adapters
  -> overlay and JSONL action logs
```

The implementation should also cleanly separate this new lane from the old
gesture-recognition system. Treat `gestures/`, TCN, DTW, and IPN as a future
diagnostic/model lane. Build the class-demo control path beside it, preferably
under a new `control/` package and an `airdesk control run` CLI command, so the
demo runtime is not forced through old swipe/model abstractions.

The combo buffer should record stable events, not every frame. Example event
stream:

```text
hand-0 open_palm entered
hand-0 open_palm held 400ms
hand-0 fist entered
hand-0 fist held 350ms
combo matched: open_palm -> fist -> open_palm
```

Combos should be same-hand by default, consume matched events, expire after
about two seconds, and carry cooldowns. High-risk commands need clearer grammar
than ordinary movement. For example, closing a window should be a combo such as
`open_palm -> fist -> open_palm`, not a raw fist pose, so it does not conflict
with fist-held window movement.

Current MVP grammar candidate:

| Input pattern | Intended action | Safety notes |
| --- | --- | --- |
| Open hand / relaxed tracked hand in cursor mode | Move cursor | Cursor movement should be modeful and visible. |
| Index pinch tap | Left click | Requires pointer-button injection through an input adapter. |
| Index pinch hold | Hold left button for select/drag | Press on hold threshold, release when pinch releases. |
| Thumb/middle pinch tap | Right click | Keep separate threshold/hysteresis from index pinch. |
| Thumb/middle pinch hold + vertical movement | Scroll | Use accumulated dy, dead zone, and repeat rate limit. |
| Fist held in center | Arm one fist command | Show target window title before any move-window action. |
| Fist held then moved left/right zone | Move active/window-under-cursor to adjacent workspace | Dispatch `movetoworkspace r-1` / `movetoworkspace r+1` by default; holding repeats after a cooldown, returning near the anchor stops repeats. |
| Fist held then moved up/down zone | Switch workspace without moving a window | Dispatch `workspace r-1` / `workspace r+1` by default; holding repeats after a cooldown, returning near the anchor stops repeats. |
| Open palm -> sideways open palm | Open launcher | Dispatch `global caelestia:launcher` on Caden's setup. |
| Open palm -> fist -> open palm | Close active window | Dispatch `killactive`; show "close armed" and focused window before firing. |

The goal is not a large vocabulary. The goal is a small "mid-air mouse plus
window manager" grammar that strings together without accidental overlap.

Live-control hardening update: the command fist gate should not depend mostly on
image-y fingertip fold. It now combines closed-finger scores, intermediate-joint
evidence, fingertip clustering, thumb support, low open-palm evidence, and the
older fold threshold as one signal. Pinch taps must be canceled when an
ambiguous or forming-fist frame appears, because live logs showed accidental
clicks when a pinch briefly entered while Caden was making a fist and then
released through an ambiguous frame. Hyprland workspace selectors default to
current-monitor relative `r+1` / `r-1`; keep `+1` / `-1` available as a CLI
override for live setup comparison.

Held-repeat update: live testing after the fist primitive fix showed workspace
and move-window dispatches working, but one-shot arm consumption made multi-step
workspace travel clumsy. The fist anchor now remains active while the fist stays
stable, repeated workspace/window steps are rate-limited by
`--fist-repeat-cooldown-seconds`, and moving back near the original anchor or
releasing fist returns to neutral. Middle pinch now defaults to the same strict
`0.06` threshold as index pinch.

Cleanup rule: do not delete or rewrite the old recognizer stack during the first
logic-control implementation. Park it as preview/replay/evaluation future work,
avoid importing it from the new control runtime, and only extract shared
landmark pose math when it directly reduces duplication.

## Interaction Modes

AirDesk should be designed as a set of explicit modes. This is both a product-design decision and a research decision: modes make accidental activation easier to reason about, and they let the study compare specific interaction techniques rather than one vague "gesture control" bucket.

### Command Mode

Purpose:

- fast discrete commands
- workspace navigation
- focus movement
- media control
- fullscreen/floating
- cancel/recover

Activation:

- open palm held for ~300 ms
- after activation, accept one command then exit listening mode

Why first:

- low implementation risk
- maps well to Hyprland
- safest for the first user study

### Cursor Mode

Purpose:

- occasional pointer control
- coarse clicking
- click/drag when the mouse is unavailable
- future on-screen keyboard input

Activation:

- explicit cursor/control mode is visible and pauseable
- open or relaxed hand movement controls the pointer
- pinch becomes a button/scroll primitive rather than the pointer-move clutch
- release/idle returns to neutral without clicking

Important design constraints:

- cursor control must be modeful, never always-on
- show a visible cursor-mode indicator
- use smoothing and configurable gain
- support a dead zone to avoid jitter
- route real clicks/scroll through an isolated input adapter, with dry-run and
  visible event logs before global execution
- route real control cursor movement through a real pointer-motion source when
  hover feedback matters; compositor cursor warps are not enough for every app

Possible implementation path:

1. Keep Hyprland `movecursor` as the fallback real pointer movement path.
2. Add a dry-run input target for click/scroll/drag events and tests.
3. Use the Linux `uinput` target for `airdesk control run --execute
   --pointer-execute` cursor movement and pointer clicks/scroll, so apps receive
   normal relative pointer motion and hover states update.
4. Add index-pinch click, middle-pinch right click, and pinch-drag scroll.
5. Keep a fake/diagnostic overlay view for tuning thresholds safely.

### Text Mode / Virtual Keyboard

Purpose:

- limited text entry when a physical keyboard is unreachable or undesirable
- possible accessibility fallback

Interaction options:

- large on-screen keyboard with pinch-to-press
- dwell select after cursor hover
- row/column scanning as an accessibility profile
- predictive words as a stretch goal

Study note:

Text entry is a separate research problem. Do not mix it into the first evaluation unless required. It can be a future-work section or a small demo feature.

### Profiles

Profiles should change gesture mappings and thresholds by context:

- **Kitchen/Media**: play/pause, rewind, volume, next/previous.
- **Window Manager**: workspace, focus, fullscreen, move-to-workspace.
- **Presentation**: next/previous, pointer/highlight, zoom.
- **Accessibility**: reduced movement gain, larger dwell targets, longer confirmations, safer non-destructive defaults.
- **Cursor**: pointer takeover, click, drag, large target selection.
- **Virtual Keyboard**: pinch/dwell text entry and command-palette style input.
- **Hybrid Keyboard + Hands**: gestures augment keyboard/mouse instead of replacing them.
- **Study Safe**: restricted commands, high logging, dry-run options, no destructive actions.
- **Experimental**: new gestures, models, sensors, and bindings.

## Why Hyprland

Hyprland is a strong research platform because it exposes window-management actions through `hyprctl dispatch`, and tiling desktop operations already map naturally onto spatial gestures:

- workspace left/right
- focus up/down/left/right
- move active window
- toggle fullscreen/floating
- move active window to another workspace
- media control
- volume control

## Gesture Vocabulary

Keep the first vocabulary small. Better to evaluate a reliable small set than a flashy unreliable one.

The current class-demo vocabulary is the deterministic logic-control grammar in
the crunch-time pivot section above. The older dynamic command vocabulary below
is historical planning context and should not drive the next implementation
slice unless Caden explicitly pivots back to learned/dynamic gesture work.

### Core Safety Gesture

**Open palm held for ~300 ms**

- Meaning: enter gesture command mode / listening window
- Purpose: prevents accidental activation
- HCI framing: clutching avoids the Midas Touch problem

### Command Gestures

**Swipe left / right**

- Action: switch workspace left/right
- Hyprland: `hyprctl dispatch workspace r-1` / `r+1`, or `m-1` / `m+1` depending desired behavior

**Point left / right / up / down**

- Action: move focus between windows
- Hyprland: `hyprctl dispatch movefocus l/r/u/d`

**Pinch hold + lateral flick**

- Action: move active window to adjacent workspace
- Hyprland: `hyprctl dispatch movetoworkspace r-1` / `r+1` or equivalent

**Palm push**

- Action: play/pause media
- Implementation: playerctl, MPRIS, or desktop media key simulation

**Fist**

- Action: cancel current gesture mode; possibly close overlay
- Do not map fist to destructive close at first.

### Stretch Gestures

**Two-hand spread / pinch**

- Action: resize floating window or adjust volume

**Pinch drag**

- Action: move floating window

These are more visually impressive but riskier. Treat as stretch goals after the core command gestures work.

## Cursor and Text Feature Plan

Cursor and text features are now central to the crunch-time prototype because
the live demo is a "mid-air mouse plus window manager" loop rather than a
learned semantic gesture recognizer. Text entry remains future work.

### Cursor Mode MVP

- explicit cursor/control mode is visible and pauseable
- open/relaxed hand movement maps to pointer movement
- index pinch tap triggers left click
- thumb/middle pinch tap triggers right click
- index pinch hold plus vertical movement triggers scroll
- visible overlay indicates cursor mode is active
- configurable gain/smoothing/dead-zone

### Cursor Mode Risks

- jitter from hand landmarks
- fatigue from sustained pointing
- poor precision for small UI targets
- accidental drag/click
- inconsistent behavior across multiple monitors
- compositor restrictions around synthetic pointer input

### Virtual Keyboard MVP

- large keys
- pinch-to-press
- no tiny modifier gymnastics
- first target: short strings only
- optional dwell select if pinch is uncomfortable

Virtual keyboard should be described as product ambition or future work unless it becomes reliable enough to evaluate separately.

## Study Design

Use a within-subjects design. Each participant completes the same set of tasks in multiple conditions.

### Conditions

Minimum:

1. Keyboard/mouse baseline
2. AirDesk gesture control

Stronger:

1. Keyboard/mouse at desk
2. AirDesk gesture control at desk
3. AirDesk gesture control under simulated situational impairment

The simulated impairment should be simple and ethical:

- standing 1-2 meters from the keyboard
- wearing gloves
- holding a kitchen utensil/object
- hands lightly covered by disposable gloves to simulate "do not touch keyboard" scenarios

Avoid creating mess or unsafe conditions.

### Tasks

Pick 5-6 tasks. Use tasks that gestures might reasonably support.

Candidate tasks:

1. Pause and resume a video while standing away from the keyboard.
2. Switch to an adjacent workspace.
3. Focus a neighboring window.
4. Toggle fullscreen on the active window.
5. Move the active window to another workspace.
6. Open launcher or a known app.
7. Dismiss/cancel an accidental action.
8. Optional cursor-mode task: click a large target or drag a large item in a controlled test window.
9. Optional text-mode task: enter a short word using a virtual keyboard.

Avoid heavy text entry. That is not what this system is for.

### Dependent Variables

Quantitative:

- task completion time
- task success rate
- number of recognition errors
- false activations
- number of correction actions
- number of times clutch mode was entered
- optional: command latency

Subjective:

- SUS usability score
- NASA-TLX workload
- perceived fatigue rating
- perceived usefulness rating
- preference ranking
- short semi-structured interview

### Independent Variables

Core:

- interaction method: keyboard/mouse vs mid-air gestures

Optional:

- context: normal vs situationally impaired
- task type: navigation, media control, window control

## Expected Paper Structure

Follow the example papers' shape:

1. Introduction and Motivation
2. Related Work
3. System / Prototype Description
4. Gesture Vocabulary and Interaction Design
5. Experiment Design
6. Metrics and Data Collection
7. Expected Results / Limitations
8. Deliverables
9. Conclusion
10. References

## Core Argument

AirDesk is not trying to replace the keyboard and mouse. It explores where a spatial input channel can complement them.

Good phrasing:

> Mid-air gestures are likely unsuitable for continuous precision desktop work, but may be valuable for short, coarse, command-like interactions when conventional input is temporarily unavailable or inconvenient.

Accessibility phrasing:

> AirDesk is accessibility-motivated and designed with users experiencing temporary, situational, or chronic interaction constraints in mind. This project does not claim validated benefit for arthritis, RSI, or motor impairment populations unless those users are included in the study; it instead evaluates whether the interaction technique is promising enough to justify that future work.

## Implementation Architecture

See `architecture.md` for the full architecture. The short version:

```text
src/airdesk/
  capture/          webcam, video file, recorded stream, Kinect stream
  tracking/         MediaPipe, OpenVINO, Kinect/depth, mock/replay, future Ultraleap
  state/            normalized hand/body state, calibration, smoothing
  gestures/         rules, templates, classifiers, temporal recognizers
  modes/            command, cursor, media, window manager, presentation, keyboard
  profiles/         bindings, thresholds, safety rules, profile selection
  actions/          Hyprland, media, input, shell, dry-run
  overlay/          status UI, fake cursor, radial/menu surfaces, virtual keyboard
  logging/          JSONL/CSV logs, recordings, replay
  study/            tasks, trials, participant/session metadata, exports
  config/           schema, defaults, profile loading
```

Possible alternatives:

- Python + OpenCV + MediaPipe: fastest path for webcam/landmarks, but keep it behind a backend interface
- OpenVINO backend: useful if MediaPipe is unstable or too slow on the target Linux setup
- Kinect/libfreenect2 backend: useful for depth, body/presence context, and coarse gestures
- Node/Electron/Tauri overlay: nicer UI, but more overhead
- Qt/GTK/layer-shell overlay: more native desktop feel, likely better long-term OS integration
- Web app + local bridge: useful for a control center, but probably not the first real-time overlay

Default recommendation:

**Python daemon first, debug/overlay second, full control center later.**

## Metrics Logging

The daemon should write JSONL raw logs and optional CSV study exports:

- timestamp
- participant id
- condition
- task id
- tracking backend
- active profile
- active mode
- gesture candidate
- gesture detected
- confidence
- action requested
- action success/failure
- cooldown/cancel events
- false activation marker if manually tagged

The study runner can manually mark task start/end with hotkeys, a small CLI, or the overlay/control center.

Recording/replay should support multiple levels:

- metadata only
- landmarks only
- landmarks plus debug frames
- full video only with explicit consent

## Risks

- false activations if no clutch gesture exists
- poor webcam lighting or background clutter
- MediaPipe hand landmark jitter or backend-specific tracking failure
- gesture fatigue
- gestures being slower than keyboard shortcuts
- user confusion if gesture vocabulary is not memorable
- destructive actions from misrecognition
- cursor jitter and fatigue in cursor mode
- virtual keyboard becoming its own full project
- architecture collapsing around one tracker or one UI surface too early
- model training before enough real AirDesk data exists

Mitigations:

- use clutch mode
- start with non-destructive actions
- provide visual/audio feedback
- keep command set small
- use cooldowns and confidence thresholds
- include practice trials
- record both objective and subjective data
- keep cursor/text modes explicit and visually indicated
- evaluate command mode first, then treat cursor/text as separate slices
- keep tracking, recognition, modes, profiles, and actions as separate modules
- add recording/replay before serious recognizer iteration
- train the causal TCN after rule/DTW fallback behavior and real logs exist
