#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

./venv/bin/isort --profile=black --gitignore -- src tests
./venv/bin/black -- src tests