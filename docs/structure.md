# Repository Structure

Snapshot of the layout at session start (2026-05-25). The repo is a fresh scaffold for a ~6h onsite project on phage-host prediction in *E. coli* (multi-output regression: bacterial genome -> 96-dim phage interaction vector). Most code is empty; the substance is in `docs/`.

## Top-level layout

```
.
|-- CLAUDE.md              # Agent instructions (also visible as claude.md -- same file, case-insensitive FS)
|-- README.md              # Empty
|-- requirements-dev.txt   # Heavy ML stack: torch, transformers>=5.5, accelerate, peft, bitsandbytes, datasets, lm-eval, etc.
|-- .gitignore             # venv/, .vscode/, __pycache__/, .mypy_cache/
|-- .claude/               # Agent harness config (settings, hooks, rules, skills)
|-- .vscode/               # Empty (gitignored)
|-- docs/                  # Project briefs and reference material (see below)
|-- scripts/               # Dev workflow shell scripts
|-- src/                   # Python source -- currently just an empty __init__.py
|-- tests/                 # Pytest tests -- currently just an empty __init__.py
`-- venv/                  # Local Python 3.12 venv (gitignored, rebuilt by bootstrap.sh)
```

## docs/

Background and brief for the onsite. Read these before writing code.

- `tabula_workday.md` -- Onsite project brief: scope, goals, deliverables, dataset overview (Gaborieau et al. 2024, *Nature Microbiology*).
- `tabula_reference.md` -- Self-contained biology + ML reference for phage-host prediction in *E. coli*. Defines every term on first use. No prior bacterial-genomics knowledge assumed.
- `tabula_research.md` -- Companion document with decisions, corrections, and pitfalls not obvious from the paper or README. Distinguishes `[VERIFIED-FROM-CODE]` vs `[UNVERIFIED]` claims.
- `2023.11.22.567924v1.full.pdf` -- BioRxiv preprint of the Gaborieau et al. paper.
- `structure.md` -- This file.

## scripts/

All scripts `cd` to repo root and invoke binaries from `./venv/bin/` directly. The venv is NOT activated in-shell -- it's referenced by absolute path inside each script. No system Python is used.

- `bootstrap.sh` -- `python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt`
- `format.sh` -- isort (black profile, gitignore-aware) + black, on `src tests`
- `lint.sh` -- `mypy -- src tests`
- `test.sh` -- `pytest` (no `-v`, picks up both `tests/` and any other auto-discovered dirs)
- `watch.sh` -- `watchexec` on `.py` files in `src` + `tests`, re-runs lint + test on change

## src/ and tests/

Both empty (just `__init__.py`). All implementation work lives here once started.

## .claude/

- `settings.json` -- effortLevel xhigh, opus-4-7[1m], env vars to silence HF/tqdm progress bars, PreToolUse hooks (block `pip install`, require user approval on `git push|reset|checkout --|clean|rebase`).
- `hooks/check-bash.sh` -- The hook script the settings reference for bash gating.
- `rules/code-style.md` -- Type-annotation discipline (modern types, no `# type: ignore`, no string forward refs); tensor shape comments; ASCII-only "NASA C programmer charset" for code and `.md` (LaTeX for math symbols).
- `rules/testing.md` -- pytest conventions, no system Python, env vars to suppress HF noise.
- `skills/decompress/SKILL.md` -- User-invocable break-time skill.
