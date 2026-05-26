#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python3 -m venv venv

./venv/bin/pip install -r requirements-dev.txt

# Paper repo (read-only data dependency for tests/ and reference).
# Clones only when missing; does not auto-update on subsequent runs.
PAPER_REPO_URL="https://github.com/mdmparis/coli_phage_interactions_2023.git"
if [ -f paper-repo/data/interactions/interaction_matrix.csv ]; then
    echo "paper-repo already present, skipping clone."
else
    echo "Cloning paper repo into ./paper-repo (~210 MB, shallow)..."
    rm -rf paper-repo
    git clone --depth 1 "$PAPER_REPO_URL" paper-repo
    echo "paper-repo cloned."
fi
