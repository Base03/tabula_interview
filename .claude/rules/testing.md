<!-- No paths: frontmatter -- load unconditionally so Claude is aware of these
     rules after compaction, not only when a matching file happens to be opened. -->

# Testing Conventions

## Python Environment

Use the project venv for all Python execution. Prefer the scripts:
```bash
./scripts/test.sh      # run all tests
./scripts/lint.sh      # mypy type checking
./scripts/format.sh    # isort + black
./scripts/watch.sh     # continuous lint + test on file changes
```

For targeted runs, activate the venv first:
```bash
source venv/bin/activate && python -m pytest tests/ -x
source venv/bin/activate && python -m pytest tests/test_specific_file.py -x
source venv/bin/activate && python -m mypy src
```

Never use the system Python. Never install packages directly -- add to `requirements-dev.txt` and run `./scripts/bootstrap.sh`.

When testing python files which are currently being written, lint before testing to catch syntax errors and type issues early.

## pytest

Run without `-v` to minimize token usage. Use `-v` only when debugging failures that can't be diagnosed from the short output. Use `-x` to stop on first failure.

Tests live in `tests/`, auto-discovered by pytest.

## Token-Saving Noise Suppression

`.claude/settings.json` sets env vars that suppress progress bars and noisy logging in Claude's Bash sessions:
- `HF_HUB_DISABLE_PROGRESS_BARS=1` -- HF download bars
- `TRANSFORMERS_VERBOSITY=error` -- HF warnings
- `TQDM_DISABLE=1` -- all tqdm progress bars (training, eval, etc.)

If a `tests/conftest.py` is added later, set these at module level there too -- HF imports trigger logging before fixtures run.
