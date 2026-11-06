#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_DIR/infra:${PYTHONPATH:-}"

echo "🧪 Running ScalarLM pytest unit tests"
echo "   Project: $PROJECT_DIR"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

cd "$PROJECT_DIR"

if [ $# -gt 0 ]; then
    echo "🎯 Running specific test: $1"
    python -m pytest "$1" -v
else
    echo "🏃 Running all pytest unit tests (excluding legacy and integration scripts)"
    python -m pytest test/ -v --ignore=test/legacy
fi