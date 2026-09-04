# ScalarLM Observability

Production-grade observability for ScalarLM's distributed LLM inference and training platform.

## What You Get

- **📊 Metrics**: Prometheus + Grafana dashboard with 9 panels showing queue depth, latency (TTFT/TBT), throughput, GPU utilization
- **📝 Logs**: Structured JSON logging with trace IDs
- **🔍 Traces**: OpenTelemetry instrumentation (API server, work queue, vLLM)
- **🖥️ GPU Telemetry**: DCGM metrics for NVIDIA GPUs (optional)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Using the Dashboard](#using-the-dashboard)
4. [Testing & Verification](#testing--verification)
5. [Troubleshooting](#troubleshooting)
6. [Architecture](#architecture)

---

# Quick Start

5-minute setup for ScalarLM observability.

## 1. Start Stack

```bash
./scripts/start-observability.sh  # Auto-detects CPU/GPU
```

## 2. Start ScalarLM

```bash
./scalarlm up cpu  # or nvidia
```

## 3. Verify

```bash
python test/infra/test_observability_data.py
```

**Expected:**
```
✓ Working: 7 metrics
✓ All dashboard queries successful
```

## 4. Generate Traffic

```bash
python test/infra/generate_load.py --requests 20
```

## 5. View Dashboard

```bash
open http://localhost:3001  # admin/admin
```

## Key URLs

- **Dashboard**: http://localhost:3001/d/scalarlm-overview/scalarlm-overview
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/v1/metrics

---

# Setup

Complete setup guide for ScalarLM observability stack.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- ScalarLM installed

## Detailed Setup

### 1. Start Observability Stack

```bash
# Auto-detect hardware
./scripts/start-observability.sh

# Or force specific mode
./scripts/start-observability.sh cpu    # No GPU metrics
./scripts/start-observability.sh gpu    # Include DCGM exporter
```

**Services started:**
- **Prometheus** (port 9090) - Metrics database
- **Grafana** (port 3001) - Dashboards
- **Loki** (port 3100) - Log aggregation
- **Tempo** (port 3200) - Trace storage
- **OpenTelemetry Collector** (ports 4317, 4318) - Telemetry ingestion
- **DCGM Exporter** (port 9400, GPU only) - NVIDIA GPU metrics

**Expected output:**
```
========================================
ScalarLM Observability Stack Startup
========================================

✓ CPU mode (auto-detected)
  Starting observability stack WITHOUT GPU metrics

Configuring Prometheus with prometheus-cpu.yml...
Starting services...

✅ Observability stack is ready!

Access points:
  - Prometheus:  http://localhost:9090
  - Grafana:     http://localhost:3001 (admin/admin)
```

### 2. Start ScalarLM

```bash
./scalarlm up cpu  # or nvidia
```

ScalarLM automatically enables observability features:
- OpenTelemetry tracing
- Prometheus metrics at `/v1/metrics`
- Structured JSON logging

### 3. Verify Services

#### Check Prometheus Targets

```bash
# Open targets page
open http://localhost:9090/targets
```

All targets should show **UP**:
- `scalarlm-api` (localhost:8000/v1/metrics)
- `vllm` (localhost:8001/metrics)
- `dcgm` (localhost:9400/metrics, GPU only)

#### Run Verification Test

```bash
# Full pipeline test (checks all services + metrics)
python test/infra/test_observability_data.py
```

**Expected output:**
```
==> Checking services...
✓ ScalarLM API is running
✓ Prometheus is running
✓ Grafana is running

==> Verifying Prometheus metrics...
✓ Queue depth metric: 0
✓ Queue wait time metric: 0.004
✓ vLLM running requests: 0
✓ KV cache usage: 0%
✓ Prompt tokens: 93
✓ Generated tokens: 385

==> Verifying Grafana dashboard queries...
✓ All 5 dashboard queries successful

Prometheus Metrics:
  ✓ Working: 7
  ✗ Errors: 0
```

### 4. Generate Traffic

```bash
# Generate test requests to populate metrics
python test/infra/generate_load.py --requests 20 --delay 1
```

**Output:**
```
[1/20] Sending request... ✓ OK (0.85s, 89 tokens)
[2/20] Sending request... ✓ OK (1.06s, 104 tokens)
...
Load generation complete
  Successes: 20
  Success rate: 100.0%
```

### 5. View Dashboards

```bash
# Open Grafana
open http://localhost:3001
```

**Login:** admin / admin

The **ScalarLM Overview** dashboard loads automatically, showing:
- Queue Depth
- Queue Wait Time (p50/p95/p99)
- Time to First Token (p50/p95/p99)
- Time Between Tokens (p50/p95/p99)
- KV Cache Usage
- Running Requests
- Request Rate
- Token Throughput
- GPU Utilization (if GPU mode)

## Service Ports

| Service | Port | Endpoint |
|---------|------|----------|
| ScalarLM API | 8000 | `/v1/metrics` |
| vLLM | 8001 | `/metrics` |
| Grafana | 3001 | Web UI |
| Prometheus | 9090 | Web UI + API |
| Loki | 3100 | Log API |
| Tempo | 3200 | Trace API |
| OTLP gRPC | 4317 | Trace/metrics ingestion |
| OTLP HTTP | 4318 | Trace/metrics ingestion |
| DCGM Exporter | 9400 | GPU metrics (NVIDIA only) |

## Configuration Files

Located in `deployment/observability/`:

```
deployment/observability/
├── prometheus-cpu.yml           # Prometheus config (CPU)
├── prometheus-gpu.yml           # Prometheus config (GPU)
├── loki-config.yml             # Loki configuration
├── tempo-config.yml            # Tempo configuration
├── otel-collector.yml          # OpenTelemetry config
├── alerts.yml                  # Alerting rules
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasources.yml
    │   └── dashboards/dashboards.yml
    └── dashboards/
        └── scalarlm-overview.json
```

## Managing Services

### Stop Services

```bash
# Stop all services
./scripts/stop-observability.sh

# Or manually
docker-compose -f docker-compose.observability.yaml down
```

### Restart Services

```bash
# Restart specific service
docker-compose -f docker-compose.observability.yaml restart grafana

# Restart all
docker-compose -f docker-compose.observability.yaml restart
```

### View Logs

```bash
# View logs
docker logs scalarlm-grafana
docker logs scalarlm-prometheus
docker logs scalarlm-dcgm-exporter  # GPU only

# Follow logs
docker logs -f scalarlm-grafana
```

### Clean Up Data

```bash
# Remove all data (metrics, logs, traces)
docker-compose -f docker-compose.observability.yaml down -v

# Remove specific volume
docker volume rm scalarlm_grafana-data
docker volume rm scalarlm_prometheus-data
```

---

# Using the Dashboard

How to use ScalarLM's observability features.

## Access Grafana

```bash
open http://localhost:3001
# Login: admin / admin
```

The **ScalarLM Overview** dashboard loads automatically.

## Dashboard Panels

### 1. Queue Depth by Model
Shows real-time requests waiting in work queue per model.

**What to look for:**
- Sustained high depth = need more workers or faster inference
- Spikes during traffic bursts = normal
- Always zero = no requests or workers pulling immediately

### 2. Queue Wait Time (p50/p95/p99)
Time from request submission to worker pickup.

**Target SLOs:**
- p50 < 100ms
- p95 < 500ms
- p99 < 1s

**High wait time causes:**
- Too few generate workers
- Workers busy with long requests
- Model not loaded (adapter loading delay)

### 3. Time to First Token - TTFT (p50/p95/p99)
Latency from request start to first token returned.

**Includes:**
- Queue wait time
- Adapter loading (if trained model)
- Prefill time (prompt processing)

**Target SLOs:**
- p50 < 500ms
- p95 < 2s (red threshold line on panel)
- p99 < 5s

**High TTFT causes:**
- Large prompts (more tokens to process)
- Cold adapter loading
- High batch contention

### 4. Time Between Tokens - TBT (p50/p95/p99)
Inter-token latency during streaming generation.

**Target SLOs:**
- p50 < 50ms
- p95 < 100ms
- p99 < 200ms

**High TBT causes:**
- Low GPU utilization
- Memory bandwidth bottleneck
- Small batch size (inefficient)

### 5. GPU KV Cache Usage
Percentage of KV cache memory used.

**What to look for:**
- 0-50% = plenty of headroom
- 50-80% = good utilization
- 80-95% = near capacity, may limit batch size
- >95% = memory pressure, OOM risk

### 6. Running Requests
Current number of requests being processed by vLLM.

**What to look for:**
- Consistently zero = no traffic or vLLM not running
- Matches expected concurrency
- Sudden drops = possible vLLM restart

### 7. Request Rate
Requests per second over time.

**Use for:**
- Traffic pattern analysis
- Load testing verification
- Capacity planning

### 8. Token Throughput
Tokens generated per second (prompt + generated).

**What to look for:**
- Prompt tokens (blue) = input processing throughput
- Generated tokens (green) = output generation throughput
- Total throughput = system capacity

**Typical values:**
- CPU: 10-100 tokens/sec
- Single GPU: 500-2000 tokens/sec
- Multi-GPU: 2000-10000+ tokens/sec

### 9. GPU Utilization (NVIDIA only)
Hardware utilization percentage.

**Target utilization:**
- Training: 80-100% (memory bound)
- Inference: 50-90% (depends on batch size)

**Low utilization:**
- Small batch size
- Request starvation (not enough traffic)
- CPU bottleneck

## Querying Metrics

### Prometheus Web UI

```bash
open http://localhost:9090
```

**Example queries:**

```promql
# Queue depth by model
scalarlm_queue_depth

# Queue wait time p95 (5 minute window)
histogram_quantile(0.95, rate(scalarlm_queue_wait_time_seconds_bucket[5m]))

# TTFT p99
histogram_quantile(0.99, rate(vllm:request_prefill_time_seconds_bucket[5m]))

# TBT p50
histogram_quantile(0.50, rate(vllm:inter_token_latency_seconds_bucket[5m]))

# Request rate
rate(vllm:e2e_request_latency_seconds_count[1m])

# Token throughput
rate(vllm:prompt_tokens_total[1m]) + rate(vllm:generation_tokens_total[1m])

# KV cache usage
vllm:kv_cache_usage_perc

# GPU utilization (NVIDIA)
DCGM_FI_DEV_GPU_UTIL
```

### Command Line Queries

```bash
# Queue depth
curl -s 'http://localhost:9090/api/v1/query?query=scalarlm_queue_depth' | jq '.data.result'

# TTFT p95 (returns value in seconds)
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(vllm:request_prefill_time_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

# Request rate (req/sec)
curl -s 'http://localhost:9090/api/v1/query?query=rate(vllm:e2e_request_latency_seconds_count[1m])' | jq '.data.result[0].value[1]'
```

## Common Workflows

### Load Testing

```bash
# Generate sustained load
python test/infra/generate_load.py --requests 100 --delay 0.5

# Watch metrics in Grafana
open http://localhost:3001/d/scalarlm-overview/scalarlm-overview
```

### Debugging Slow Requests

1. Check **TTFT** panel - is prefill slow?
2. Check **Queue Wait Time** - are requests waiting?
3. Check **KV Cache Usage** - is memory full?
4. Check **GPU Utilization** - is GPU idle?
5. Search logs for trace ID to see request flow

### Capacity Planning

1. Generate load at target QPS
2. Observe:
   - Queue depth (should stay low)
   - TTFT/TBT (should meet SLOs)
   - GPU utilization (should be 70-90%)
   - KV cache usage (should have headroom)
3. If any metric fails, scale up:
   - High queue depth → more workers or GPUs
   - High latency → faster GPU or optimize
   - Full KV cache → larger GPU or shorter contexts

---

# Testing & Verification

Step-by-step testing instructions to verify the observability implementation works correctly.

## Quick Verification

```bash
# Quick test (no external dependencies)
python test/infra/test_observability.py

# Full pipeline test (requires observability stack)
python test/infra/test_observability_data.py
```

## Test Harnesses

### Basic Observability Test

**File:** `test/infra/test_observability.py`

**What it tests:**
- ✓ Health endpoint accessible
- ✓ Metrics endpoint returns Prometheus format
- ✓ Queue metrics are tracked
- ✓ Prometheus is scraping (if running)

**Usage:**
```bash
python test/infra/test_observability.py
```

**Expected output:**
```
=== Testing Health Endpoint ===
✓ Health endpoint accessible: 200

=== Testing Metrics Endpoint ===
✓ Metrics endpoint accessible: 200
✓ Found metric: scalarlm_queue_depth
✓ Found metric: scalarlm_queue_wait_time_seconds

=== Testing Prometheus Scraping ===
✓ Prometheus is scraping ScalarLM

Test Results Summary
✓ health_endpoint: PASSED
✓ metrics_endpoint: PASSED
✓ queue_metrics: PASSED
✓ prometheus_scraping: PASSED

Total: 4 passed, 0 failed, 0 skipped
```

### Full Pipeline Verification

**File:** `test/infra/test_observability_data.py`

**What it does:**
1. Checks all services running (ScalarLM, Prometheus, Grafana)
2. Verifies Grafana dashboard exists
3. Generates 10 test requests
4. Waits for Prometheus scrape (20s)
5. Queries Prometheus for 7 key metrics
6. Verifies Grafana can query all dashboard panels
7. Prints comprehensive summary

**Usage:**
```bash
python test/infra/test_observability_data.py
```

**Expected runtime:** ~30 seconds (includes 20s wait for Prometheus scrape)

### Load Generator

**File:** `test/infra/generate_load.py`

**Features:**
- Sends chat completion requests to `/v1/chat/completions`
- Uses currently loaded model (auto-detected)
- Configurable request count and delay
- Shows success rate and latency per request

**Usage:**
```bash
# Basic: 20 requests, 1 second apart
python test/infra/generate_load.py --requests 20 --delay 1

# High load: 100 requests, 0.1s apart
python test/infra/generate_load.py --requests 100 --delay 0.1

# Sustained load: 1000 requests over ~16 minutes
python test/infra/generate_load.py --requests 1000 --delay 1
```

**When to use:**
- Testing dashboard displays live data
- Load testing the inference pipeline
- Generating data to verify alerting rules

## Typical Testing Workflow

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

---

# Troubleshooting

Common issues and solutions.

## Services Won't Start

```bash
# Check port conflicts
lsof -i :3001  # Grafana
lsof -i :9090  # Prometheus

# View service status
docker-compose -f docker-compose.observability.yaml ps

# Check logs for errors
docker logs scalarlm-grafana
```

## Grafana Shows "No Data"

```bash
# 1. Verify Prometheus is scraping
curl 'http://localhost:9090/api/v1/query?query=up'

# 2. Generate traffic
python test/infra/generate_load.py --requests 10

# 3. Wait for Prometheus scrape interval (15-30s)
sleep 30

# 4. In Grafana, set time range to "Last 5 minutes"
```

## Prometheus Targets Down

```bash
# Check ScalarLM is running
curl http://localhost:8000/v1/health

# Check vLLM is running
curl http://localhost:8001/v1/models

# Check metrics endpoints
curl http://localhost:8000/v1/metrics | head
curl http://localhost:8001/metrics | head
```

## DCGM Exporter Fails (GPU Mode)

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check driver version (need 450.80.02+)
nvidia-smi

# Check Docker has GPU support
docker info | grep -i runtime
```

## Dashboard Queries Fail

```bash
# Check Grafana can reach Prometheus
docker exec scalarlm-grafana wget -O- http://prometheus:9090/-/healthy

# Verify datasource UID
curl -s -u admin:admin http://localhost:3001/api/datasources | jq '.[] | {name, uid}'

# Should show:
# {
#   "name": "Prometheus",
#   "uid": "scalarlm-prometheus"
# }
```

## Test Fails with "ScalarLM is NOT running"

```bash
# Start ScalarLM
./scalarlm up cpu
```

## Test Fails with "Prometheus is NOT running"

```bash
# Start observability stack
./scripts/start-observability.sh cpu
```

## Generate Load Fails with "Model not found"

The load generator auto-detects the loaded model by querying `/v1/models`. If you see this error:

```bash
# Check what model is loaded
curl http://localhost:8001/v1/models | jq '.data[].id'

# Common models:
# - tiny-random/qwen3 (default for testing)
# - meta-llama/Llama-3.2-1B
# - your custom model
```

---

# Architecture

Technical design of the observability system.

## Stack Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Grafana (3001)                      │
│                  Dashboards & Visualization              │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────┐
    │            │            │            │
┌───▼────┐  ┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│Prometh-│  │  Loki  │  │ Tempo  │  │  DCGM  │
│ eus    │  │ (3100) │  │ (3200) │  │ (9400) │
│ (9090) │  │        │  │        │  │ [GPU]  │
└───▲────┘  └───▲────┘  └───▲────┘  └───▲────┘
    │           │           │           │
    │       ┌───┴───────────┴───┐       │
    │       │  OTel Collector   │       │
    │       │   (4317, 4318)    │       │
    │       └───▲───────────▲───┘       │
    │           │           │           │
┌───┴───────────┴───┐   ┌───┴───────────┴───┐
│  ScalarLM API     │   │   vLLM Server     │
│  (8000)           │   │   (8001)          │
│  /v1/metrics      │   │   /metrics        │
└───────────────────┘   └───────────────────┘
```

## Components

### Prometheus
- **Purpose**: Time-series metrics database
- **Scrapes**:
  - ScalarLM API: `localhost:8000/v1/metrics` (every 15s)
  - vLLM: `localhost:8001/metrics` (every 15s)
  - DCGM: `localhost:9400/metrics` (every 15s, GPU only)
- **Retention**: 15 days
- **Config**: `deployment/observability/prometheus-{cpu,gpu}.yml`

### Grafana
- **Purpose**: Visualization and dashboards
- **Data sources**: Prometheus (metrics), Loki (logs), Tempo (traces)
- **Dashboards**: Auto-provisioned ScalarLM Overview
- **Config**: `deployment/observability/grafana/`

### Loki
- **Purpose**: Log aggregation
- **Retention**: 7 days
- **Config**: `deployment/observability/loki-config.yml`

### Tempo
- **Purpose**: Distributed trace storage
- **Retention**: 7 days
- **Config**: `deployment/observability/tempo-config.yml`

### OpenTelemetry Collector
- **Purpose**: Telemetry ingestion and forwarding
- **Endpoints**:
  - gRPC: `localhost:4317`
  - HTTP: `localhost:4318`
- **Exports to**: Prometheus, Loki, Tempo
- **Config**: `deployment/observability/otel-collector.yml`

### DCGM Exporter (GPU only)
- **Purpose**: NVIDIA GPU hardware metrics
- **Requirements**: NVIDIA driver 450.80.02+, nvidia-docker2
- **Metrics**: GPU utilization, memory, temperature, power, NVLink
- **Config**: Runs with `--profile gpu` in Docker Compose

## Metrics Flow

```
Request → ScalarLM → Queue → Worker → vLLM
   │                  │        │        │
   ├─ Emit metrics ──→│        │        │
   │                  ├─ Track depth/wait
   │                  │        │        │
   │                  │        ├─ Load adapter
   │                  │        │        │
   │                  │        │        ├─ Generate tokens
   │                  │        │        │
   └─────────────────→└────────┴────────┴→ /metrics
                                           │
                                      Prometheus
                                      scrapes ←─┐
                                           │    │
                                      Stores in │
                                      TSDB      │
                                           │    │
                                      Grafana ──┘
                                      queries
```

## Hardware Modes

### CPU Mode
- Prometheus scrapes: ScalarLM API, vLLM
- No DCGM exporter
- Config: `prometheus-cpu.yml`

### GPU Mode
- Prometheus scrapes: ScalarLM API, vLLM, DCGM
- DCGM exporter included
- Config: `prometheus-gpu.yml`
- Auto-detection: Checks `nvidia-smi` availability

## Key Design Principles

1. **Zero Coupling**: vLLM runs unmodified; ScalarLM scrapes its `/metrics` endpoint
2. **Graceful Degradation**: Missing observability libraries don't break ScalarLM
3. **Hardware Aware**: CPU and GPU modes with appropriate metric collection
4. **Production Ready**: Structured logs, distributed tracing, SLO tracking
5. **Consistent UIDs**: Datasource UIDs are predictable (`scalarlm-prometheus`)

## Metric Reference

### ScalarLM API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `scalarlm_queue_depth` | Gauge | Requests in work queue |
| `scalarlm_queue_wait_time_seconds` | Histogram | Time in queue before processing |
| `scalarlm_inference_requests_total` | Counter | Total inference requests |

### vLLM Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:request_prefill_time_seconds` | Histogram | Time to First Token (TTFT) |
| `vllm:inter_token_latency_seconds` | Histogram | Time Between Tokens (TBT) |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency |
| `vllm:kv_cache_usage_perc` | Gauge | KV cache utilization (%) |
| `vllm:num_requests_running` | Gauge | Currently running requests |
| `vllm:num_requests_waiting` | Gauge | Requests waiting for slot |
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed |
| `vllm:generation_tokens_total` | Counter | Total tokens generated |

### GPU Metrics (NVIDIA DCGM)

| Metric | Type | Description |
|--------|------|-------------|
| `DCGM_FI_DEV_GPU_UTIL` | Gauge | GPU utilization (%) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Gauge | Memory bandwidth utilization (%) |
| `DCGM_FI_DEV_GPU_TEMP` | Gauge | GPU temperature (°C) |
| `DCGM_FI_DEV_POWER_USAGE` | Gauge | Power consumption (W) |
| `DCGM_FI_DEV_FB_USED` | Gauge | GPU memory used (MB) |

## Data Retention

| Service | Retention | Location |
|---------|-----------|----------|
| Prometheus | 15 days | `prometheus-data` volume |
| Loki | 7 days | `loki-data` volume |
| Tempo | 7 days | `tempo-data` volume |
| Grafana | Persistent | `grafana-data` volume |
