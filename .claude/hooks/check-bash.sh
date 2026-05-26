#!/bin/bash
# PreToolUse hook: block pip install, require supervision for git commands
CMD=$(jq -r '.tool_input.command' < /dev/stdin)

# Block pip install
if echo "$CMD" | grep -qE '(^|&&\s*|;\s*)pip install'; then
  echo 'Use requirements-dev.txt + ./scripts/bootstrap.sh instead of pip install.' >&2
  exit 2
fi

# Require supervision for git commands
if echo "$CMD" | grep -qE '(^|&&\s*|;\s*)git '; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"Git commands require user supervision."}}'
  exit 0
fi

exit 0
