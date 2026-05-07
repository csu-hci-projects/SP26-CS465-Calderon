# AirDesk Sprint 5 Plan

## Purpose

Sprint 5 should turn the prototype and recognition evidence into a small, credible HCI pilot package.

By Sprint 5, AirDesk should have:

- live tracking and replay,
- runtime event logs,
- command-mode state feedback,
- a pause/kill switch,
- a chosen dynamic gesture recognizer or a documented fallback,
- and enough replay/evaluation data to decide which gestures are safe for a pilot.

Current caveat after the live TCN preview: AirDesk does not yet have a reliable learned swipe recognizer for fast chained gestures. Sprint 5 should either narrow the pilot to recognizers with replay evidence, or treat recognizer failure as a documented design finding while the implementation continues toward continuous gesture spotting.

Sprint 5 should answer:

> Can AirDesk support a narrow, measurable desktop-control task set well enough to produce useful CS465 evidence?

This sprint is not about broad product polish. It is about study tooling, a Caden-only pilot, and paper-ready artifacts.

## Sprint Theme

> Convert the prototype into study evidence.

## Product / Research Stance

- Keep research claims narrow.
- Evaluate AirDesk as a secondary command layer, not a keyboard/mouse replacement.
- Treat failures as useful findings.
- Use dry-run for study tasks unless real execution is demonstrably safe.
- Keep raw video collection off by default.
- Prefer reliable command gestures over ambitious unstable gestures.

## Non-Goals

Do not attempt these in Sprint 5:

- polished public release,
- full daemon/service lifecycle,
- control center,
- virtual keyboard,
- general cursor takeover,
- Kinect/depth integration,
- multi-participant study unless the course timeline and approval path make it realistic,
- accessibility claims about populations not studied.

## Key Decisions

### Study Slice

The preferred Sprint 5 evaluation slice is:

> clutch-based mid-air command gestures for common Hyprland desktop actions under situationally impaired conditions.

Candidate tasks:

- switch workspace,
- move focus left/right,
- pause/play media or safe dry-run equivalent,
- toggle fullscreen only if verified safe,
- cancel/recover,
- compare with keyboard/mouse baseline.

Keep the task set small enough to run repeatedly without fatigue.

### Execution Mode

Sprint 5 should support three modes:

1. **Replay evaluation**
   - deterministic
   - useful for debugging

2. **Live dry-run pilot**
   - safest study mode
   - logs intended actions without controlling the desktop

3. **Live execute pilot**
   - optional
   - only for a small allowlisted action set
   - only after Sprint 3/4 data supports it

Dry-run remains the default.

Do not promote DTW or TCN swipes to live desktop actions only because they work in preview. Require event-level replay evidence on chained continuous sessions, including false activations and repeated-fire counts.

### Study Metrics

Objective metrics:

- task completion time,
- success/failure,
- false activations,
- missed intended gestures,
- cancellation/recovery count,
- command latency,
- repeated-fire count,
- tracking loss count,
- number of keyboard/mouse fallback actions.

Subjective metrics:

- perceived control,
- fatigue/discomfort,
- frustration,
- confidence,
- preference versus baseline,
- short notes/interview.

### Paper Strategy

The paper should emphasize:

- motivation: situationally impaired desktop interaction,
- system: OS-level spatial input layer prototype,
- method: clutch-based command gestures, logging/replay, safety gates,
- evaluation: pilot evidence and limitations,
- contribution: interaction design and infrastructure, not novel hand tracking.

Do not claim broad accessibility benefit unless studied.

Recognizer wording should be careful:

- The current learned model is a diagnostic preview, not validated live control.
- The evidence suggests the main technical obstacle is continuous gesture spotting and event decoding under chained motion.
- The proposed next step is a hybrid stream recognizer with position-invariant features, phase/event labels, and explicit non-gesture handling.

## Target CLI Shape

By the end of Sprint 5, these commands should exist or be expanded:

```text
airdesk study new --participant caden --condition baseline --out data/studies/pilot-0/session-baseline.jsonl
airdesk study task start workspace-switch
airdesk study task end --success true
airdesk study export data/studies/pilot-0/session-baseline.jsonl --out data/studies/pilot-0/results.csv
airdesk run --backend mediapipe --profile configs/profiles/window-manager.toml --dry-run --events-out data/studies/pilot-0/live-dry-run.jsonl
airdesk study summarize data/studies/pilot-0/*.jsonl
```

Exact command names can change if a simpler workflow fits the codebase better.

## Deliverables

### 1. Pilot Protocol

Acceptance criteria:

- Add `studies/pilot-0.md`.
- Define:
  - research question,
  - hypotheses/expectations,
  - conditions,
  - task list,
  - measures,
  - procedure,
  - rest breaks,
  - failure/abort rules,
  - data files collected,
  - privacy notes.
- Explicitly state this is a Caden-only pilot unless expanded later.

### 2. Study Event Schema

Acceptance criteria:

- Add typed study/trial event structures.
- Include participant/session IDs.
- Include condition labels:
  - baseline,
  - AirDesk dry-run,
  - AirDesk execute if used,
  - simulated impairment if used.
- Include task start/end, success, notes, and timestamps.
- Tests cover serialization.

### 3. Study CLI

Acceptance criteria:

- Add simple commands to create sessions, start/end tasks, and write notes.
- Commands write JSONL.
- Commands do not require live camera dependencies.
- Tests cover CLI behavior with temp files.

### 4. Runtime + Study Log Integration

Acceptance criteria:

- Runtime logs can include a study session/task ID.
- AirDesk action events can be associated with active trial/task.
- Study summaries can combine trial events with runtime gesture/action events.
- Dry-run pilot can be run without real Hyprland dispatch.

### 5. CSV / Summary Export

Acceptance criteria:

- Export study logs to CSV.
- Include:
  - task,
  - condition,
  - duration,
  - success,
  - false activations,
  - missed gestures if annotated,
  - cancellation count,
  - notes.
- Tests cover summary calculations.

### 6. Baseline Task Workflow

Acceptance criteria:

- Document how to run keyboard/mouse baseline tasks.
- Add timing/logging helpers for baseline trials.
- Keep baseline comparable to AirDesk tasks.
- Do not overclaim if baseline timing is manual.

### 7. Pilot Run

Acceptance criteria:

- Run at least one Caden-only pilot session.
- Include baseline and AirDesk dry-run conditions.
- Optionally include execute mode only if safe.
- Summarize:
  - what worked,
  - what failed,
  - false activations,
  - fatigue/discomfort,
  - recognition issues,
  - next design changes.

### 8. Paper Scaffold

Acceptance criteria:

- Add `studies/paper-outline.md` or `paper/outline.md`.
- Include:
  - introduction,
  - related work buckets,
  - system description,
  - gesture vocabulary,
  - study method,
  - results placeholders,
  - limitations,
  - future work.
- Ensure wording does not claim gestures replace keyboard/mouse.
- Include the dynamic recognition strategy as design rationale.

## Recommended Implementation Order

1. Write `studies/pilot-0.md`.
2. Add study event types and serialization tests.
3. Add study CLI commands for session/task logging.
4. Add CSV/summary export.
5. Integrate runtime event logs with study IDs.
6. Add baseline task workflow docs.
7. Run a dry-run pilot and summarize results.
8. Add paper outline with evidence placeholders.
9. Update README/tasks/handoff.
10. Run `ruff`, `pytest`, replay smoke, and study CLI smoke.

## Risks and Mitigations

### Study Scope Is Too Large

Risk:

- Too many tasks or gestures make the pilot noisy and tiring.

Mitigation:

- Use 3-5 tasks.
- Prefer repeated small trials.
- Drop unstable gestures.

### Recognition Is Still Not Pilot-Ready

Risk:

- False activations prevent useful pilot evidence.

Mitigation:

- Use dry-run.
- Reduce gesture vocabulary.
- Evaluate design failure honestly.
- Frame Sprint 5 as pilot findings, not final validation.

### Baseline Comparison Is Unfair

Risk:

- Keyboard shortcuts may be much faster and make AirDesk look worse.

Mitigation:

- Focus on situational impairment conditions.
- Measure preference and perceived convenience.
- Discuss that gestures target different contexts, not raw desktop speed.

### Paper Claims Overreach

Risk:

- It is tempting to describe a broad hands-first OS when the study only validates a slice.

Mitigation:

- Separate product vision from evaluated contribution.
- State limitations clearly.
- Use "accessibility-motivated" rather than validated accessibility benefit.

## Definition of Done

Sprint 5 is done when:

- pilot protocol exists,
- study event logging works,
- study summaries/CSV export work,
- runtime events can be associated with study tasks,
- a Caden-only pilot has been run or a blocker is documented,
- paper outline exists,
- README/tasks/handoff are updated,
- `uv run ruff check .` passes,
- `uv run pytest` passes.

## Sprint 6 Gate

At the end of Sprint 5, choose:

### Path A: Study Evidence Is Good Enough

Proceed to paper writing and final polish:

- write report,
- clean figures/tables,
- polish demo script,
- freeze evaluated feature set.

### Path B: Prototype Needs One More Interaction Sprint

Refine before final study:

- improve recognizer,
- adjust gestures,
- add overlay feedback,
- repeat pilot.

### Path C: Product Direction Outruns Class Scope

Split work:

- finish class paper on command-mode pilot,
- keep cursor/control-center/product expansion as future work.
