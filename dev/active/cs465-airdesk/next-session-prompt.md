# AirDesk Next Session Prompt

You are working with Caden on AirDesk at:

`/home/caden/projects/AirDesk`

This repo is in CS465 submission/readiness cleanup. The current priority is a
fast grader-facing pass, with the root `README.md` as the main public entrypoint.
Do not collect new data, train models, or tune live hand-control thresholds
unless cleanup exposes an obvious broken command.

Hard boundaries:

- Preserve user changes.
- Do not touch `paper/` or any `/papers` path unless Caden explicitly changes
  that instruction.
- Do not delete raw data, logs, model checkpoints, or project provenance
  blindly.
- Keep `airdesk control run`, `airdesk gesture ...`, old `airdesk run`, and old
  `airdesk cursor run` command surfaces stable.
- Use `apply_patch` for manual edits.

Start with:

1. Run `git status --short`.
2. Read `README.md`, `pyproject.toml`, and these active docs:
   - `dev/active/cs465-airdesk/context.md`
   - `dev/active/cs465-airdesk/tasks.md`
   - `dev/active/cs465-airdesk/architecture.md`
3. Check whether dirty `paper/` files are still present; leave them unstaged and
   untouched.

Current project truth:

- AirDesk is a webcam-based mid-air desktop control prototype for Hyprland.
- It is designed with situationally impaired interaction in mind: dirty hands,
  gloves, limited reach, standing away from the desk, temporary pain, and
  presentation contexts.
- Do not claim validated accessibility benefits for populations not tested.
  Use "designed with" or "may be relevant to."
- The current live demo path is deterministic MediaPipe control:
  `airdesk control run`.
- Live actions are dry-run by default.
- The preferred real pointer path is `--execute --pointer-execute`, because
  `/dev/uinput` gives normal hover/click behavior.
- Learned TCN/IPN/DTW work is retained as research/diagnostic infrastructure,
  not as the live global desktop action recognizer.

What changed in the cleanup pass:

- Root `README.md` was rewritten as the grader-facing entrypoint.
- It now covers overview, what works, `uv` setup, safe dry-run demo, optional
  Hyprland/uinput execution, gesture cheat sheet, repo map, architecture,
  tests, limitations, and grader notes.
- `.gitignore`/submission surface was audited: generated `data/*` artifacts are
  ignored except `data/.gitkeep`; caches and virtualenvs are ignored/untracked.
- Tracked `dev/active/` and `dev/archive/` docs remain useful provenance but are
  not ideal for a skim-grade package.

Recommended submission strategy:

- Best near-term strategy: create a clean grader export/zip or cleaned
  submission branch that includes README, source, configs, tests, scripts, and
  selected study docs, while excluding internal planning/session docs and local
  artifacts.
- Do not delete `dev/` provenance from the working repo unless Caden explicitly
  chooses that branch/package strategy.

Verification to keep current:

```bash
uv run airdesk --help
uv run airdesk control run --help
uv run ruff check .
uv run pytest
```

If continuing cleanup, focus on packaging/readability, not live threshold
tuning.
