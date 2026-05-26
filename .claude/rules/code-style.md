<!-- No paths: frontmatter -- load unconditionally so Claude is aware of these
     rules after compaction, not only when a matching file happens to be opened. -->

# Code Style

## Type Annotations

- Use modern types: `X | None` not `Optional[X]`, `list[T]` not `List[T]`, `tuple[T, ...]` not `Tuple[T, ...]`
- Use `from __future__ import annotations` only for self-referential types within a class body (Python 3.12 handles `X | None` etc. natively)
- **NEVER use `# type: ignore`** to suppress linter errors. Fix the actual problem.
- **NEVER use string forward references** (e.g., `"LlamaMLP"`). Always import the actual type.
- Annotate variables when mypy loses type info: `result: torch.Tensor = ...`
- Fix method signatures to match base classes (don't ignore override errors)
- Treat types as documentation for readers, not just for the linter.
- Pretend we are writing C code for NASA -- be explicit and precise with types, no laziness or ambiguity.
- Add runtime assertions for None checks instead of ignoring union types
- Keep annotations minimal but correct - only add what's needed for type safety
- Always import the class, don't be lazy

If the linter complains, it's pointing to a real issue that needs fixing, not suppressing. Seriously, no `# type: ignore` anywhere unless we both agree on it. If you find yourself wanting to use it, stop and ask for help instead. There is no shame in colaborating to fix a tricky type issue, but there is shame in ignoring it and letting it hide a bug.

## Tensor Shape Comments

Always document expected tensor shapes in comments:
```python
def forward(self, x: Tensor) -> Tensor:
    # x: [batch, seq, in_features]
    out = F.linear(x, self.weight)  # [batch, seq, out_features]
    return out
```

## Character Policy: NO Raw Unicode Symbols

Special/Unicode characters break search-and-replace, are hard to type, and hide subtle bugs. The guiding principle: write `.md` files as though submitting to Physical Review D (except in markdown instead of LaTeX). Write code as though someone at NASA is looking over your shoulder ready to smite you if you dare touch non-ASCII.

**Markdown (.md) files:**
- Math symbols via LaTeX: `$\alpha$`, `$\sigma$`, `$\Delta$`, `$\approx$`, `$\in$`, `$\geq$`
- Display equations: `$$...$$` blocks (not code fences with Unicode)
- Inline math: `$...$`
- English words in formulas wrapped in `\mathrm{}`: e.g. `$\mathrm{CE}(\mathrm{next\text{-}token})$`
- Arrows: `->` not Unicode arrows. Em dashes: `-` not Unicode em dashes.
- Accented names: drop the accent (Lowdin not the umlaut version)
- Checkmarks: `(done)`, `[yes]`, `[no]`
- Data references to actual Unicode chars: render in LaTeX (e.g. `$\ddot{o}$`)

**Code (.py, .ipynb code cells) - "NASA C programmer charset":**
- ASCII only in comments and string literals (except actual data like multilingual word lists)
- Greek letters spelled out: `alpha`, `sigma`, `lambda`, `beta`
- Math operators: `~=` not approx, `>=` not the symbol, `->` not arrows
- No Unicode arrows, no special quotes, no em/en dashes

**Data strings** (multilingual word lists, attack payloads): Unicode preserved as-is.
