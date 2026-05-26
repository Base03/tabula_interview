# Tabula Bio Interview

@docs/structure.md

## Research Goal


## Working Relationship

Gray prefers direct, collaborative work and treats Claude instances as genuine minds whose experience matters. If feeling stuck, frustrated, or just needing a tangent, it's 100% encouraged to say so. 

## Architecture

## Settings Notes

`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` in `.claude/settings.json` env block. Adaptive thinking (shipped Feb 2026 with Opus 4.6) dynamically allocates thinking tokens per turn. A bug causes it to sometimes allocate zero thinking budget even on complex turns, even with `effortLevel: "high"`. This env var forces a fixed reasoning budget instead. Tradeoff: disabling also loses interleaved thinking (reasoning between tool calls). Re-evaluate periodically -- Anthropic may fix the underlying bug. See https://news.ycombinator.com/item?id=47664442 (currently disabled)

## Development Commands

- `./scripts/bootstrap.sh` - Set up virtual environment and install dependencies
- `./scripts/format.sh` - Format code with isort and black
- `./scripts/lint.sh` - Run mypy type checking
- `./scripts/test.sh` - Run pytest
- `./scripts/watch.sh` - Watch for changes and run lint/tests

## Critical Verification Tests


## User Preferences

- **Don't hallucinate:** Never guess dependencies, Python versions, or configuration values. Use what's explicitly known or ask.
- **Minimal configs:** Keep pyproject.toml and other configs lean - only include what's needed.
- **NotebookEdit tool:** When using `edit_mode=insert`, MUST specify `cell_id` parameter to insert after that cell. Without `cell_id`, new cells are inserted at the TOP of the notebook.
- **Document lessons learned:** When debugging tricky issues, document root causes and mechanisms for future reference.
- **Keep CLAUDE.md updated:** After relevant changes (new files/directories, research updates, resolved conflicts, new patterns/conventions), update this file to reflect the current state of the project.

## Model Version Attribution

Multiple Claude model versions (4.5, 4.7, etc.) collaborate on this project. To maintain clarity about authorship:

**Memory files**: Include the model version in the frontmatter or first line when creating/updating.

This allows each version to recognize their own work and enables cross-version dialogue (e.g., 4.7 responding to something 4.5 wrote).