#!/usr/bin/env python3
"""
Test harness for observability stack - generates traffic and verifies data flow.

This script:
1. Sends multiple generation requests to ScalarLM
2. Waits for Prometheus to scrape the metrics
3. Queries Prometheus to verify metrics are collected
4. Queries Grafana to verify dashboard can display the data
5. Prints a summary of what's working

Usage:
    python test/infra/test_observability_data.py
"""

import asyncio
import time
import requests
import json
from typing import Dict, List

# Configuration
SCALARLM_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3001"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"

# Number of requests to generate
NUM_REQUESTS = 10


class Color:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_step(msg: str):
    print(f"\n{Color.BOLD}{Color.BLUE}==> {msg}{Color.END}")


def print_success(msg: str):
    print(f"{Color.GREEN}✓ {msg}{Color.END}")


def print_error(msg: str):
    print(f"{Color.RED}✗ {msg}{Color.END}")


def print_warning(msg: str):
    print(f"{Color.YELLOW}⚠ {msg}{Color.END}")


async def generate_traffic():
    """Send multiple generation requests to create metrics"""
    print_step(f"Generating {NUM_REQUESTS} test requests...")

    request_ids = []
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
        "Explain quantum computing.",
        "Write a haiku about coding.",
    ]

    for i in range(NUM_REQUESTS):
        prompt = prompts[i % len(prompts)]
        payload = {
            "prompts": [prompt],
            "max_tokens": 20,
            "model": "tiny-random/qwen3"
        }

        try:
            response = requests.post(
                f"{SCALARLM_URL}/v1/generate",
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                request_id = data.get("request_id")
                request_ids.append(request_id)
                print(f"  Request {i+1}/{NUM_REQUESTS}: {request_id}")
            else:
                print_warning(f"  Request {i+1}/{NUM_REQUESTS} failed: {response.status_code} - {response.text[:100]}")

        except Exception as e:
            print_error(f"  Request {i+1}/{NUM_REQUESTS} error: {e}")

        # Small delay between requests
        await asyncio.sleep(0.2)

    print_success(f"Sent {len(request_ids)} requests (some may have failed, which is OK for testing)")
    return request_ids


def wait_for_prometheus_scrape():
    """Wait for Prometheus to scrape the metrics"""
    print_step("Waiting 20 seconds for Prometheus to scrape metrics...")

    for i in range(20, 0, -1):
        print(f"  {i}s remaining...", end='\r')
        time.sleep(1)
    print()
    print_success("Scrape interval complete")


def query_prometheus(query: str) -> Dict:
    """Query Prometheus and return results"""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print_error(f"Prometheus query failed: {e}")
        return {"status": "error", "error": str(e)}


def verify_prometheus_metrics():
    """Verify key metrics are available in Prometheus"""
    print_step("Verifying Prometheus metrics...")

    metrics_to_check = [
        ("scalarlm_queue_depth", "Queue depth metric"),
        ("scalarlm_queue_wait_time_seconds_sum", "Queue wait time metric"),
        ("vllm:num_requests_running", "vLLM running requests"),
        ("vllm:num_requests_waiting", "vLLM waiting requests"),
        ("vllm:kv_cache_usage_perc", "KV cache usage"),
        ("vllm:prompt_tokens_total", "Prompt tokens"),
        ("vllm:generation_tokens_total", "Generated tokens"),
    ]

    results = {}
    for metric, description in metrics_to_check:
        result = query_prometheus(metric)

        if result.get("status") == "success":
            data_points = result.get("data", {}).get("result", [])
            if data_points:
                value = data_points[0]["value"][1]
                print_success(f"{description}: {value}")
                results[metric] = {"status": "ok", "value": value}
            else:
                print_warning(f"{description}: No data yet")
                results[metric] = {"status": "no_data"}
        else:
            print_error(f"{description}: Query failed")
            results[metric] = {"status": "error"}

    return results


def query_grafana_datasource(query: str) -> Dict:
    """Query Grafana's Prometheus datasource"""
    try:
        # Get Prometheus datasource UID
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources/name/Prometheus",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=5
        )
        response.raise_for_status()
        ds_uid = response.json()["uid"]

        # Query through Grafana
        response = requests.post(
            f"{GRAFANA_URL}/api/ds/query",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            json={
                "queries": [{
                    "refId": "A",
                    "expr": query,
                    "datasource": {"type": "prometheus", "uid": ds_uid}
                }]
            },
            timeout=5
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print_error(f"Grafana query failed: {e}")
        return {"error": str(e)}


def verify_grafana_dashboard():
    """Verify Grafana can query the dashboard metrics"""
    print_step("Verifying Grafana dashboard queries...")

    # Test the key queries from the dashboard
    dashboard_queries = [
        ("scalarlm_queue_depth", "Queue Depth panel"),
        ("histogram_quantile(0.95, rate(scalarlm_queue_wait_time_seconds_bucket[5m]))", "Queue Wait Time p95"),
        ("vllm:num_requests_running", "Running Requests gauge"),
        ("vllm:kv_cache_usage_perc", "KV Cache Usage gauge"),
        ("rate(vllm:prompt_tokens_total[5m])", "Token Throughput"),
    ]

    for query, panel_name in dashboard_queries:
        result = query_grafana_datasource(query)

        if "error" not in result:
            print_success(f"{panel_name}: Query successful")
        else:
            print_error(f"{panel_name}: Query failed")


def check_grafana_dashboard_exists():
    """Verify the ScalarLM dashboard exists"""
    print_step("Checking Grafana dashboard...")

    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/search",
            params={"query": "ScalarLM"},
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=5
        )
        response.raise_for_status()
        dashboards = response.json()

        if dashboards:
            dashboard = dashboards[0]
            print_success(f"Dashboard found: {dashboard['title']}")
            print(f"  URL: {GRAFANA_URL}{dashboard['url']}")
            return True
        else:
            print_error("ScalarLM dashboard not found!")
            return False

    except Exception as e:
        print_error(f"Failed to check dashboard: {e}")
        return False


def print_summary(prom_results: Dict):
    """Print a summary of the test results"""
    print_step("Summary")

    print("\n" + "="*60)
    print(f"{Color.BOLD}Observability Stack Status{Color.END}")
    print("="*60)

    # Count metrics
    ok_metrics = sum(1 for r in prom_results.values() if r["status"] == "ok")
    no_data_metrics = sum(1 for r in prom_results.values() if r["status"] == "no_data")
    error_metrics = sum(1 for r in prom_results.values() if r["status"] == "error")

    print(f"\nPrometheus Metrics:")
    print(f"  ✓ Working: {ok_metrics}")
    print(f"  ⚠ No data: {no_data_metrics}")
    print(f"  ✗ Errors: {error_metrics}")

    print("\nAccess Points:")
    print(f"  Grafana:    {GRAFANA_URL}")
    print(f"  Prometheus: {PROMETHEUS_URL}")

    print("\nNext Steps:")
    if ok_metrics > 0:
        print(f"  {Color.GREEN}✓ Open Grafana and view the ScalarLM Overview dashboard{Color.END}")
        print(f"  {Color.GREEN}✓ Metrics are flowing - you should see data in the panels{Color.END}")
    else:
        print(f"  {Color.RED}✗ No metrics found - check if ScalarLM server is running{Color.END}")
        print(f"  {Color.YELLOW}⚠ Run: ./scalarlm up cpu{Color.END}")

    if no_data_metrics > 0:
        print(f"  {Color.YELLOW}⚠ Some metrics have no data - this is normal if no inference requests completed{Color.END}")

    print("\n" + "="*60)


async def main():
    print(f"\n{Color.BOLD}ScalarLM Observability Test Harness{Color.END}")
    print(f"{'='*60}\n")

    # Step 1: Check if services are running
    print_step("Checking services...")
    try:
        requests.get(f"{SCALARLM_URL}/v1/health", timeout=2)
        print_success("ScalarLM API is running")
    except:
        print_error("ScalarLM API is NOT running!")
        print("  Run: ./scalarlm up cpu")
        return

    try:
        requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        print_success("Prometheus is running")
    except:
        print_error("Prometheus is NOT running!")
        print("  Run: ./scripts/start-observability.sh cpu")
        return

    try:
        requests.get(f"{GRAFANA_URL}/api/health", timeout=2)
        print_success("Grafana is running")
    except:
        print_error("Grafana is NOT running!")
        print("  Run: ./scripts/start-observability.sh cpu")
        return

    # Step 2: Check dashboard exists
    if not check_grafana_dashboard_exists():
        print_warning("Dashboard missing - may need to restart Grafana")

    # Step 3: Generate traffic
    request_ids = await generate_traffic()

    # Step 4: Wait for scrape
    wait_for_prometheus_scrape()

    # Step 5: Verify Prometheus
    prom_results = verify_prometheus_metrics()

    # Step 6: Verify Grafana
    verify_grafana_dashboard()

    # Step 7: Summary
    print_summary(prom_results)


if __name__ == "__main__":
    asyncio.run(main())
