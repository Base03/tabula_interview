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
|-- paper-repo/            # Shallow clone of mdmparis/coli_phage_interactions_2023 (gitignored, ~210MB)
|-- src/                   # Python source -- currently just an empty __init__.py
|-- tests/                 # Pytest tests (see below)
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

- `bootstrap.sh` -- `python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt`, then clones `paper-repo/` if missing (echoes "skipping" if present; does not auto-update on subsequent runs).
- `format.sh` -- isort (black profile, gitignore-aware) + black, on `src tests`
- `lint.sh` -- `mypy -- src tests`
- `test.sh` -- `pytest` (no `-v`, picks up both `tests/` and any other auto-discovered dirs)
- `watch.sh` -- `watchexec` on `.py` files in `src` + `tests`, re-runs lint + test on change

## src/ and tests/

`src/` is empty (just `__init__.py`); all implementation work lives there once started.

`tests/` contains:
- `__init__.py` (empty)
- `test_paper_pipeline_structure.py` -- structural-contract test that traces `predict_all_phages.py` against the data in `paper-repo/` and asserts the pre/post-one-hot dims, the 402-vs-403 anomaly, the absence of `H_host`, the LF110 carve-out, and the NaN-handling of the `same_ABC_as_host` bug. If the paper repo's schema ever shifts, this test surfaces which doc claim went stale. Auto-clones `paper-repo/` on first run; skips with a clear message if cloning fails.

## paper-repo/

Shallow clone of [mdmparis/coli_phage_interactions_2023](https://github.com/mdmparis/coli_phage_interactions_2023) at the project root, used as read-only input by the structural-contract test in `tests/`. Gitignored (~210MB, mostly genome data we don't want to vendor).

`./scripts/bootstrap.sh` clones it on first run and prints "paper-repo already present, skipping clone." on subsequent runs. The test fixture in `tests/test_paper_pipeline_structure.py` also has an auto-clone fallback so `./scripts/test.sh` works on a fresh checkout without bootstrapping.

To refresh against the upstream repo:
```bash
rm -rf paper-repo && ./scripts/bootstrap.sh
```

## .claude/

- `settings.json` -- effortLevel xhigh, opus-4-7[1m], env vars to silence HF/tqdm progress bars, PreToolUse hooks (block `pip install`, require user approval on `git push|reset|checkout --|clean|rebase`).
- `hooks/check-bash.sh` -- The hook script the settings reference for bash gating.
- `rules/code-style.md` -- Type-annotation discipline (modern types, no `# type: ignore`, no string forward refs); tensor shape comments; ASCII-only "NASA C programmer charset" for code and `.md` (LaTeX for math symbols).
- `rules/testing.md` -- pytest conventions, no system Python, env vars to suppress HF noise.
- `skills/decompress/SKILL.md` -- User-invocable break-time skill.
