#!/bin/bash
# Start observability stack (auto-detects CPU vs NVIDIA GPU)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.observability.yaml"

# Allow override via command line argument
FORCE_MODE="${1:-}"

echo "========================================"
echo "ScalarLM Observability Stack Startup"
echo "========================================"
echo ""

# Detect if NVIDIA GPU is available (unless overridden)
if [ "$FORCE_MODE" = "gpu" ] || [ "$FORCE_MODE" = "nvidia" ]; then
    echo "✓ GPU mode (forced via argument)"
    echo "  Starting observability stack WITH GPU metrics (DCGM exporter)"
    echo ""
    PROFILE="--profile gpu"
    PROM_CONFIG="prometheus-gpu.yml"
elif [ "$FORCE_MODE" = "cpu" ]; then
    echo "✓ CPU mode (forced via argument)"
    echo "  Starting observability stack WITHOUT GPU metrics"
    echo ""
    PROFILE=""
    PROM_CONFIG="prometheus-cpu.yml"
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected (auto-detected)"
    echo "  Starting observability stack WITH GPU metrics (DCGM exporter)"
    echo "  To force CPU mode: ./scripts/start-observability.sh cpu"
    echo ""
    PROFILE="--profile gpu"
    PROM_CONFIG="prometheus-gpu.yml"
else
    echo "✓ CPU-only system detected (auto-detected)"
    echo "  Starting observability stack WITHOUT GPU metrics"
    echo "  To force GPU mode: ./scripts/start-observability.sh gpu"
    echo ""
    PROFILE=""
    PROM_CONFIG="prometheus-cpu.yml"
fi

# Copy the appropriate Prometheus config
echo "Configuring Prometheus with $PROM_CONFIG..."
cp "$PROJECT_ROOT/deployment/observability/$PROM_CONFIG" "$PROJECT_ROOT/deployment/observability/prometheus.yml"

# Start services
echo "Starting services..."
cd "$PROJECT_ROOT"
docker-compose -f docker-compose.observability.yaml $PROFILE up -d

echo ""
echo "========================================"
echo "Services Started"
echo "========================================"
echo ""
echo "Access points:"
echo "  - Prometheus:  http://localhost:9090"
echo "  - Grafana:     http://localhost:3001 (admin/admin)"
echo ""
echo "Waiting 10 seconds for services to initialize..."
sleep 10

# Verify services are running
echo ""
echo "Service Status:"
docker-compose -f docker-compose.observability.yaml ps

echo ""
echo "✅ Observability stack is ready!"
echo ""
echo "Next steps:"
echo "  1. Make sure ScalarLM is running: ./scalarlm up cpu|nvidia"
echo "  2. Run tests: python test/infra/test_observability.py"
echo "  3. Open Grafana: open http://localhost:3001"
