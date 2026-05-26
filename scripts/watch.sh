#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

watchexec --exts py --restart --watch src --watch tests -- './scripts/lint.sh && ./scripts/test.sh'