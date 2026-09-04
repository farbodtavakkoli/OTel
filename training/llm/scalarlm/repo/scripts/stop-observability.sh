#!/bin/bash
# Stop observability stack

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.observability.yaml"

echo "========================================"
echo "Stopping Observability Stack"
echo "========================================"

cd "$PROJECT_ROOT"

# Stop all services (including GPU profile if it was started)
docker-compose -f docker-compose.observability.yaml --profile gpu down

echo ""
echo "✅ Observability stack stopped"
echo ""
echo "To remove data volumes as well, run:"
echo "  docker-compose -f docker-compose.observability.yaml --profile gpu down -v"
