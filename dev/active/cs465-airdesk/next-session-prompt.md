# AirDesk Next Session Prompt

You are working with Caden on AirDesk at:

`/home/caden/projects/AirDesk`

This is a context-reset continuation after the learned all-IPN/TCN live-preview
lane proved useful but not safe enough for class-demo desktop actions. Start
with a staff-level review stance, preserve user changes, plan before editing,
use `apply_patch` for manual edits, add/update tests with code, run
`uv run ruff check .` and `uv run pytest`, then commit/push meaningful chunks to
`origin/main`.

Before editing:

1. Run `git status`.
2. Read `AGENTS.md` if present.
3. Read:
   - `README.md`
   - `deep-research-report.md` if present
   - `dev/active/cs465-airdesk/context.md`
   - `dev/active/cs465-airdesk/recognition-v2-plan.md`
   - `dev/active/cs465-airdesk/architecture.md`
   - `dev/active/cs465-airdesk/plan.md`
   - `dev/active/cs465-airdesk/tasks.md`
   - `dev/active/cs465-airdesk/public-dataset-survey.md`
   - `dev/active/cs465-airdesk/tracking-samples.md`
4. Do not discard user changes.

Current product decision:

Do **not** try to rescue all-IPN/TCN as the live action recognizer for the class
demo. Keep learned/DTW/motion recognizers in preview/replay/evaluation only.

The next implementation target is a deterministic MediaPipe-landmark
logic-control lane: a small "mid-air mouse plus window manager" grammar that
recreates core keyboard/mouse/window affordances with observable pose facts,
stable timing, visible feedback, and dry-run-first safety.

Separation/cleanup decision:

Build this as a side-by-side live-control path, not as more code inside the old
learned/dynamic gesture stack. The first side-by-side slice now exists in
`src/airdesk/control/`, with `airdesk control run` registered as the new dry-run
surface. Keep `airdesk gesture ...`, old `airdesk run`, and old
`airdesk cursor run` stable as diagnostic/legacy surfaces until the new control
runtime is proven live.

Why the pivot happened:

- Best all-IPN checkpoint:
  `data/models/gestures/tcn-v2-ipn-all-w16-80ep-h64-l4.pt`.
- Held-out all-IPN scoring was useful but not live-safe:
  `gesture_macro_f1=0.521`, `gesture_micro_f1=0.742`, top-1 `0.757`,
  top-3 `0.934`.
- Live preview produced high-confidence false activations from ordinary hand
  presence/motion: `Throw up`, `Open twice`, `Zoom out`, point-like heads, and
  unstable lateral throw direction.
- Mode-aware learned filtering is implemented for diagnostics, but live desktop
  actions should now come from deterministic logic.

Architecture to implement:

```text
MediaPipe landmarks
  -> per-hand primitive pose features
  -> stable pose debouncer
  -> pose transition events
  -> rolling combo buffer, max 4 events / about 2 seconds
  -> mode/action grammar
  -> guarded Hyprland and input adapters
  -> overlay/status and JSONL logs
```

Important design rules:

- Emit stable pose events, not per-frame command spam.
- Combos should be same-hand by default, consume matched events, expire after
  about 2 seconds, and have cooldowns.
- Avoid gesture overlap:
  - `fist` alone is a window-grab/hold state.
  - close window should be deliberate, for example
    `open_palm -> fist -> open_palm`.
- Keep all execution dry-run-first.
- Show what AirDesk is seeing and what it is about to do: `Seeing`, `Combo`,
  `Armed`, `Target window`, `Executed`, and `Suppressed`.

MVP grammar:

| Input pattern | Action |
| --- | --- |
| Open/relaxed hand in cursor mode | Move cursor |
| Index pinch tap | Left click |
| Index pinch hold | Hold left button for select/drag |
| Thumb/middle pinch tap | Right click |
| Thumb/middle pinch hold + vertical movement | Scroll |
| Fist held center | Arm one fist command and show target window |
| Fist moved left/right zone | Move active/window-under-cursor to workspace left/right |
| Fist moved up/down from fist start | Switch workspace up/down without moving a window |
| Open palm -> sideways open palm | Open launcher |
| Open palm -> fist -> open palm | Close active window |

Caden's Hyprland setup:

- Launcher: `hyprctl dispatch global caelestia:launcher`
- Workspace: `hyprctl dispatch workspace -1` / `+1`
- Move window to workspace: `hyprctl dispatch movetoworkspace -1` / `+1`
- Close active window: `hyprctl dispatch killactive`
- Move cursor: `hyprctl dispatch movecursor <x> <y>`
- `hyprctl` is installed.
- `/dev/uinput` is writable by `caden`.
- No external pointer helper (`ydotool`, `dotool`, `wtype`) is installed.
- Python `evdev` is not installed as of the planning pass.

Current implementation status:

- `src/airdesk/control/` contains primitive control pose facts, stable pose
  debouncing, a per-hand combo buffer, the first dry-run grammar, and a runtime
  loop.
- `airdesk control run` exists and defaults to dry-run. It logs what the system
  is seeing, stable pose events, combo state, cursor moves, requested actions,
  action results, per-pose evidence/confidence, ambiguity reasons, and
  grammar diagnostics.
- The current grammar covers open-hand relative cursor movement, index
  pinch-tap left click, index-pinch-hold left-button drag/select,
  middle-pinch-tap right click, middle-pinch-hold vertical scroll, center-fist
  armed move-window/workspace switching, launcher combo, and deliberate
  close-window combo.
- Control pose facts are resolved through a suppression policy rather than a
  blind priority list. Fist must clearly dominate pinch evidence, sideways palm
  must clearly dominate pinch artifacts, clean pinch can suppress plain
  open-palm, and close calls emit no command pose with an ambiguity reason.
- Guarded move/close actions can query the active window title for target-window
  feedback.
- Fist is now the command clutch: a stable fist creates a short-lived anchor,
  left/right motion or side-zone crossing from that anchor fires one
  `movetoworkspace`, and up/down motion from that anchor fires one `workspace`
  switch. Firing, release, or expiry consumes the arm. Ambiguous diagonal motion
  logs a suppression reason and emits no command. Side zones default to
  `left <= 0.30` and `right >= 0.70`; workspace motion defaults to `0.10`;
  move-window motion defaults to `0.12`; cursor gain defaults to `12.0` with
  smoother `0.25` alpha and a `1px` dead zone. The control runtime filters to
  one active hand, and the recommended live command now uses `--max-num-hands 1`.
- Fist detection now requires multi-finger, multi-landmark evidence: all four
  fingertips must sit at least `0.09` normalized y-units below their MCP joints,
  the curled fingertips must remain clustered/low-spread, and the thumb must
  support the fist shape. Relaxed curl, partial curl, pinch-like fist artifacts,
  and noisy sideways hands are covered by tests.
- Pinch taps are more forgiving: tap max is now `0.45s`, and a short pinch
  release can still click if tracking briefly drops on release. Holding an index
  pinch past the tap window presses and holds the left button until release;
  middle-pinch hold is reserved for scrolling.
- Pointer button/scroll real execution is available with explicit
  `--pointer-execute` through `/dev/uinput`.
- Focused tests cover primitive control poses, debouncing, combo
  matching/consumption, grammar routing/cooldown, dry-run pointer routing,
  guarded Hyprland command allowlisting, and `airdesk control run`.

Latest live-test findings and next priority:

- Stop treating workspace switching and move-window as done until a fresh
  post-hardening live dry-run proves them.
- Latest pre-hardening log read:
  - `control-live-dry-run.jsonl` saw mostly `index_pinch`,
    `middle_pinch`, and relaxed states. It had only two single-frame fist
    sightings, no `fist:entered` or `fist:held`, no armed state, and no
    `workspace_*` or `move_window_*` intents.
  - `control-live-execute.jsonl` saw many fists, usually with
    `suppressed=index_pinch,middle_pinch`. It fired eight `workspace_*`
    intents and Hyprland returned `ok` for each, but it fired zero
    `movetoworkspace` intents. The move-window failure in that log was grammar
    level; the newest dry-run failure was primitive/conflict level.
- The next live pass should inspect `features[].pose_evidence`,
  `features[].ambiguity`, and `grammar_diagnostics` before changing thresholds.
  If fist never stabilizes, tune primitive thresholds. If fist stabilizes but
  `workspace_*` / `move_window_*` do not fire, tune anchor-motion thresholds or
  axis ambiguity. If Hyprland returns `ok` but the desktop does not change,
  verify Caden's desired workspace token (`+1` / `-1` versus `r+1` / `r-1`) in
  the live setup before changing the gesture grammar.

Recommended next implementation slice:

1. Review the latest live logs first:
   `data/logs/control-live-dry-run.jsonl` and any newer `control-live-*.jsonl`.
2. The primitive hardening slice is implemented. Start the next pass with a
   fresh live dry-run and inspect the new evidence fields rather than reverting
   to blind threshold sweeps.
3. Retest command grammar with:
   - workspace switch: fist held, then vertical motion from fist anchor;
   - move-window: fist held, then left/right motion/zone from fist anchor;
   - both should consume the arm, expire quickly, and log why they did or did
     not fire.
4. Then run live dry-run testing with:
   `uv run airdesk control run --backend mediapipe --device /dev/video0 --width 640 --height 480 --fps 30 --fourcc MJPG --max-num-hands 1 --cursor-gain 12.0 --cursor-smoothing-alpha 0.25 --cursor-dead-zone-px 1 --left-zone-max 0.30 --right-zone-min 0.70 --top-zone-max 0.30 --bottom-zone-min 0.70 --fist-fold-threshold 0.09 --workspace-motion-threshold 0.10 --move-window-motion-threshold 0.12 --scroll-motion-threshold 0.045 --events-out data/logs/control-live-dry-run.jsonl --show`
5. Improve live status/dashboard rendering so it clearly shows `Seeing`,
   `Combo`, `Armed`, `Target window`, `Executed`, and `Suppressed`.
6. Only after dry-run feels stable, consider guarded real Hyprland movement
   with pointer button/scroll still dry-run.
7. Keep running `uv run ruff check .` and `uv run pytest` after changes.

Safety stance:

- Learned/DTW/motion gestures stay out of live Hyprland actions.
- The new control runtime should not import `gestures.dtw`, `gestures.motion`,
  `gestures.learned_filter`, TCN modules, or IPN helpers.
- Leave old recognizer files parked for future work; do not delete or move large
  files before the logic-control MVP works.
- If shared landmark math is useful, extract it deliberately into
  `src/airdesk/poses/` or `src/airdesk/control/poses.py` and keep old imports
  compatible.
- `killactive` is high risk: require the close combo and visible close-armed
  feedback with the active window title.
- Pointer click/scroll injection must be isolated behind an input action target
  with dry-run tests before real execution.
- If scope gets too large, finish the combo buffer and dry-run grammar first.
