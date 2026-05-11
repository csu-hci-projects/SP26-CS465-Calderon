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
| Thumb/middle pinch tap | Right click |
| Index pinch hold + vertical movement | Scroll |
| Sideways open palm held left/right | Switch workspace left/right |
| Fist held center | Arm window move and show target window |
| Fist moved left/right zone | Move active/window-under-cursor to workspace left/right |
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

Recommended first implementation slice:

1. Review existing:
   - `src/airdesk/gestures/primitives.py`
   - `src/airdesk/modes/cursor.py`
   - `src/airdesk/actions/hyprland.py`
   - `src/airdesk/actions/cursor.py`
   - `src/airdesk/cli_runtime.py`
   - existing cursor/action/runtime tests.
2. Add primitive logic features for:
   - stable open palm
   - stable fist
   - sideways open palm
   - index pinch
   - middle pinch
   - palm zone
   - pinch vertical motion for scroll.
3. Add the stable-event debouncer and combo buffer.
4. Implement the grammar in dry-run first.
5. Expand guarded action adapters only as needed.
6. Add dashboard/status and JSONL logging for pose/combo/action state.
7. Add focused tests for primitives, debouncing, combo matching/expiry,
   grammar conflicts, cooldown, dry-run routing, and guarded Hyprland allowlist.

Safety stance:

- Learned/DTW/motion gestures stay out of live Hyprland actions.
- `killactive` is high risk: require the close combo and visible close-armed
  feedback with the active window title.
- Pointer click/scroll injection must be isolated behind an input action target
  with dry-run tests before real execution.
- If scope gets too large, finish the combo buffer and dry-run grammar first.
