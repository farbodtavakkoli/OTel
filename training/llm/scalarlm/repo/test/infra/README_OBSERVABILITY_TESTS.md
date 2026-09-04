# Observability Test Harnesses

This directory contains automated test harnesses for the ScalarLM observability stack.

## Quick Start

```bash
# 1. Start observability stack (CPU or GPU mode)
./scripts/start-observability.sh cpu

# 2. Start ScalarLM
./scalarlm up cpu

# 3. Run full verification test
python test/infra/test_observability_data.py

# 4. Generate load and view in Grafana
python test/infra/generate_load.py --requests 20 --delay 1
open http://localhost:3001
```

## Test Files

### `test_observability.py`
Basic observability tests that verify core instrumentation works.

**Tests**:
- ✓ Health endpoint accessible
- ✓ Metrics endpoint returns Prometheus format
- ✓ Queue metrics are tracked
- ✓ Prometheus is scraping (if running)

**Usage**:
```bash
python test/infra/test_observability.py
```

**When to use**: Quick smoke test to verify observability is working

---

### `test_observability_data.py`
**Full data pipeline verification** - checks entire flow from ScalarLM → Prometheus → Grafana.

**What it does**:
1. Checks all services are running (ScalarLM, Prometheus, Grafana)
2. Verifies Grafana dashboard exists
3. Generates 10 test requests
4. Waits for Prometheus scrape (20s)
5. Queries Prometheus for 7 key metrics
6. Verifies Grafana can query all dashboard panels
7. Prints comprehensive summary

**Usage**:
```bash
python test/infra/test_observability_data.py
```

**Expected runtime**: ~30 seconds (includes 20s wait for Prometheus scrape)

**When to use**: Verify the complete observability stack after setup or changes

---

### `generate_load.py`
**Load generator** for creating realistic traffic to populate metrics.

**Features**:
- Sends chat completion requests to `/v1/chat/completions`
- Uses currently loaded model (auto-detected)
- Configurable request count and delay
- Shows success rate and latency per request

**Usage**:
```bash
# Basic: 20 requests, 1 second apart
python test/infra/generate_load.py --requests 20 --delay 1

# High load: 100 requests, 0.1s apart
python test/infra/generate_load.py --requests 100 --delay 0.1

# Sustained load: 1000 requests over ~16 minutes
python test/infra/generate_load.py --requests 1000 --delay 1
```

**When to use**:
- Testing dashboard displays live data
- Load testing the inference pipeline
- Generating data to verify alerting rules

---

## Typical Workflow

### First-Time Setup Verification
```bash
# 1. Start services
./scripts/start-observability.sh cpu
./scalarlm up cpu

# 2. Run comprehensive test
python test/infra/test_observability_data.py

# Expected: All 7 metrics working, all dashboard queries successful
```

### Viewing Live Metrics in Grafana
```bash
# Terminal 1: Generate continuous load
python test/infra/generate_load.py --requests 50 --delay 1

# Terminal 2: Open dashboard
open http://localhost:3001/d/scalarlm-overview/scalarlm-overview

# Watch metrics update in real-time
```

### Quick Smoke Test (After Code Changes)
```bash
# Quick test - no external dependencies needed
python test/infra/test_observability.py

# Expected: 3-4 tests pass (Prometheus test skipped if not running)
```

## Troubleshooting

### Test fails with "ScalarLM is NOT running"
```bash
# Start ScalarLM
./scalarlm up cpu
```

### Test fails with "Prometheus is NOT running"
```bash
# Start observability stack
./scripts/start-observability.sh cpu
```

### Generate load fails with "Model not found"
The load generator auto-detects the loaded model by querying `/v1/models`. If you see this error:

```bash
# Check what model is loaded
curl http://localhost:8001/v1/models | jq '.data[].id'

# Common models:
# - tiny-random/qwen3 (default for testing)
# - meta-llama/Llama-3.2-1B
# - your custom model
```

### Grafana shows "No data"
```bash
# 1. Generate traffic
python test/infra/generate_load.py --requests 10 --delay 1

# 2. Wait for Prometheus scrape (15-30s)
sleep 30

# 3. Change Grafana time range to "Last 5 minutes"

# 4. Verify Prometheus is scraping
open http://localhost:9090/targets
# All targets should show "UP"
```

## What Gets Tested

| Metric | test_observability.py | test_observability_data.py | generate_load.py |
|--------|:---------------------:|:--------------------------:|:----------------:|
| Metrics endpoint (`/v1/metrics`) | ✓ | ✓ | - |
| Queue depth | ✓ | ✓ | ✓ |
| Queue wait time | - | ✓ | ✓ |
| vLLM running requests | - | ✓ | ✓ |
| KV cache usage | - | ✓ | ✓ |
| Prompt tokens | - | ✓ | ✓ |
| Generated tokens | - | ✓ | ✓ |
| Prometheus scraping | ✓ | ✓ | - |
| Grafana dashboard | - | ✓ | - |
| Grafana queries | - | ✓ | - |
| **Full pipeline** | - | ✓ | - |

## See Also

- [TESTING.md](../../docs/observability/TESTING.md) - Step-by-step testing guide
- [GETTING_STARTED.md](../../docs/observability/GETTING_STARTED.md) - User guide
- [ARCHITECTURE.md](../../docs/observability/ARCHITECTURE.md) - Technical design
