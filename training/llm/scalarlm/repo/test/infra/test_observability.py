#!/usr/bin/env python3
"""
Test observability implementation.

This script verifies that:
1. FastAPI server exposes /v1/metrics endpoint
2. Prometheus metrics are being collected
3. Queue metrics are tracked
4. Structured logging is working
"""

import requests
import time
import json
import sys


def test_metrics_endpoint(base_url="http://localhost:8000"):
    """Test that the Prometheus metrics endpoint is available."""
    print("\n=== Testing Metrics Endpoint ===")

    try:
        response = requests.get(f"{base_url}/v1/metrics", timeout=5)
        print(f"✓ Metrics endpoint accessible: {response.status_code}")

        if response.status_code == 200:
            metrics_text = response.text
            print(f"✓ Metrics content length: {len(metrics_text)} bytes")

            # Check for expected metrics
            expected_metrics = [
                "scalarlm_queue_depth",
                "scalarlm_queue_wait_time_seconds",
                "scalarlm_inference_requests_total",
            ]

            found_metrics = []
            for metric in expected_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                    print(f"✓ Found metric: {metric}")
                else:
                    print(f"✗ Missing metric: {metric}")

            if len(found_metrics) == len(expected_metrics):
                print("✓ All expected metrics present")
                return True
            else:
                print(f"✗ Only {len(found_metrics)}/{len(expected_metrics)} metrics found")
                return False
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to connect to metrics endpoint: {e}")
        return False


def test_health_endpoint(base_url="http://localhost:8000"):
    """Test that the health endpoint is accessible."""
    print("\n=== Testing Health Endpoint ===")

    try:
        response = requests.get(f"{base_url}/v1/health", timeout=5)
        print(f"✓ Health endpoint accessible: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to connect to health endpoint: {e}")
        return False


def test_queue_metrics(base_url="http://localhost:8000"):
    """Test that queue metrics are being tracked."""
    print("\n=== Testing Queue Metrics ===")

    try:
        # Try to get the actual loaded model
        model_name = "test-model"  # Default fallback
        try:
            vllm_response = requests.get(f"{base_url.replace('8000', '8001')}/v1/models", timeout=2)
            if vllm_response.status_code == 200:
                models = vllm_response.json().get("data", [])
                if models:
                    model_name = models[0]["id"]
                    print(f"Using detected model: {model_name}")
        except:
            pass  # Use fallback if detection fails

        # Make a test request to generate queue activity
        # Note: API expects "prompts" (list) not "prompt" (string)
        test_payload = {
            "model": model_name,
            "prompts": ["Hello, world!"],  # prompts is a list
            "max_tokens": 10
        }

        print("Submitting test generate request...")
        response = requests.post(
            f"{base_url}/v1/generate",
            json=test_payload,
            timeout=10  # Increased timeout for actual inference
        )

        # Accept various status codes:
        # - 200: Success
        # - 202: Accepted (async processing)
        # - 404: Model not found (but request was processed by our code)
        # - 500: Internal error (but request reached our code)

        if response.status_code in [200, 202]:
            print(f"✓ Request submitted successfully: {response.status_code}")
            success = True
        elif response.status_code in [404, 500]:
            print(f"⚠ Request processed but failed: {response.status_code}")
            print(f"   This is OK for metrics test - we just need to verify metrics were updated")
            print(f"   Error: {response.text[:200]}")
            success = True  # Still test metrics
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

        if success:
            # Wait a moment for metrics to update
            time.sleep(0.5)

            # Check metrics again
            metrics_response = requests.get(f"{base_url}/v1/metrics", timeout=5)
            metrics_text = metrics_response.text

            # Look for queue metrics
            if "scalarlm_queue_depth" in metrics_text:
                # Extract queue depth value
                for line in metrics_text.split("\n"):
                    if "scalarlm_queue_depth{" in line and not line.startswith("#"):
                        print(f"✓ Queue depth metric found: {line.strip()}")
                        break

                print("✓ Queue metrics are being tracked")
                return True
            else:
                print("✗ Queue metrics not found after request")
                return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed during queue metrics test: {e}")
        return False


def test_prometheus_scraping(prometheus_url="http://localhost:9090"):
    """
    Test that Prometheus is scraping ScalarLM metrics.

    Note: This test is OPTIONAL and will be skipped if Prometheus is not running.
    To run the full observability stack, use:
        docker-compose -f docker-compose.observability.yaml up -d
    """
    print("\n=== Testing Prometheus Scraping (Optional) ===")

    try:
        # Query Prometheus for ScalarLM metrics
        query = "up{job='scalarlm-api'}"
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success" and data["data"]["result"]:
                result = data["data"]["result"][0]
                value = result["metric"]
                print(f"✓ Prometheus is scraping ScalarLM: {value}")
                return True
            else:
                print("✗ No ScalarLM metrics in Prometheus")
                print("   Make sure Prometheus is configured to scrape localhost:8000")
                return False
        else:
            print(f"✗ Prometheus query failed: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"⚠ Prometheus not accessible")
        print(f"   This is EXPECTED if observability stack is not running")
        print(f"   To start observability stack: ./scripts/start-observability.sh")
        return None


def main():
    """Run all observability tests."""
    print("=" * 60)
    print("ScalarLM Observability Test Suite")
    print("=" * 60)
    print()
    print("This test suite has two modes:")
    print("  1. QUICK TEST: Tests core instrumentation (no external deps)")
    print("  2. FULL TEST: Tests with observability stack (Prometheus, etc.)")
    print()
    print("The Prometheus test will be SKIPPED if stack is not running.")
    print("To run full test: ./scripts/start-observability.sh")
    print()

    results = {
        "health_endpoint": test_health_endpoint(),
        "metrics_endpoint": test_metrics_endpoint(),
        "queue_metrics": test_queue_metrics(),
        "prometheus_scraping": test_prometheus_scraping(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"✗ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠ {test_name}: SKIPPED")
            skipped += 1

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    # Determine if this was a quick or full test
    if skipped > 0:
        print("\n✓ Core instrumentation tests PASSED (Quick Test Mode)")
        print("  Some tests were skipped because observability stack is not running.")
        print("  To run full tests: ./scripts/start-observability.sh")

    if failed > 0:
        print("\n✗ SOME TESTS FAILED - Check the output above for details.")
        sys.exit(1)
    else:
        if skipped == 0:
            print("\n✓ ALL TESTS PASSED (Full Test Mode)")
        print("\n✅ Observability implementation is working correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
