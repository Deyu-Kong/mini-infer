#!/usr/bin/env bash
# Install mini-infer SDK in development mode.
# Usage: bash sdk/install.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
pip install -e "$HERE" "$@"