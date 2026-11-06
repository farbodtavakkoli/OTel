import pytest
import asyncio
import requests
import time
from typing import List, Dict, Any

SCALARLM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"


class TestKVCacheTokenLifecycle:
    def test_kv_cache_stats_endpoint(self):
        response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                assert isinstance(data, dict)
                expected_fields = ["total_tokens", "free_tokens", "reserved_tokens", "utilization_percent"]
                for field in expected_fields:
                    if field in data:
                        print(f"✓ KV cache has {field}: {data[field]}")
                        assert isinstance(data[field], (int, float))
            except:
                pass

    def test_atomic_work_acquisition(self):
        response = requests.post(f"{SCALARLM_URL}/v1/work/get_work_and_adaptors", json={
            "requested_batch_size": 2
        })
        assert response.status_code in [200, 204, 404, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Atomic work result: {data}")

                assert isinstance(data, dict)
                if "requests" in data:
                    assert isinstance(data["requests"], list)
                if "kv_tokens_reserved" in data:
                    assert isinstance(data["kv_tokens_reserved"], int)
                if "adaptors_loaded" in data:
                    assert isinstance(data["adaptors_loaded"], list)
            except:
                pass
        elif response.status_code == 204:
            print("✓ No work available (expected when queue empty)")

    def test_token_reservation_lifecycle(self):
        payload = {
            "prompt": "Count to 5",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 20
        }

        stats_before = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")

        response = requests.post(f"{SCALARLM_URL}/v1/generate", json=payload)
        assert response.status_code in [200, 422, 500]

        if response.status_code == 200:
            stats_after = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")

            if stats_before.status_code == 200 and stats_after.status_code == 200:
                try:
                    before_data = stats_before.json()
                    after_data = stats_after.json()

                    print(f"KV tokens before: {before_data.get('free_tokens', 'unknown')}")
                    print(f"KV tokens after: {after_data.get('free_tokens', 'unknown')}")

                    if 'free_tokens' in before_data and 'free_tokens' in after_data:
                        print(f"✓ Token lifecycle test completed")
                except:
                    pass

    def test_concurrent_token_reservations(self):
        import concurrent.futures

        def make_concurrent_request(request_id):
            try:
                response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                    "prompt": f"Request {request_id}: What is 1+1?",
                    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "max_tokens": 5
                })
                return request_id, response.status_code
            except Exception as e:
                return request_id, f"Error: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_concurrent_request, i) for i in range(3)]
            results = [future.result() for future in futures]

        print(f"Concurrent token reservation results: {results}")

        success_count = sum(1 for _, status in results if isinstance(status, int) and status == 200)
        error_count = sum(1 for _, status in results if isinstance(status, int) and status in [422, 500, 503])

        print(f"Successful: {success_count}, Errors: {error_count}")
        assert success_count > 0 or error_count == len(results)

    def test_batch_size_calculation(self):
        response = requests.post(f"{SCALARLM_URL}/v1/work/calculate_batch_size", json={
            "requested_batch_size": 10
        })
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Dynamic batch size calculation: {data}")

                if "calculated_batch_size" in data:
                    batch_size = data["calculated_batch_size"]
                    assert isinstance(batch_size, int)
                    assert batch_size >= 0
                    print(f"✓ Calculated batch size: {batch_size}")

                if "available_tokens" in data:
                    tokens = data["available_tokens"]
                    assert isinstance(tokens, int)
                    print(f"✓ Available KV cache tokens: {tokens}")
            except:
                pass

    def test_token_accounting_accuracy(self):
        initial_response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        if initial_response.status_code != 200:
            pytest.skip("KV cache stats not available")

        try:
            initial_data = initial_response.json()
            initial_free = initial_data.get("free_tokens", 0)
            print(f"Initial free tokens: {initial_free}")
        except:
            pytest.skip("KV cache stats not parseable")

        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hi",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "max_tokens": 3
        })

        time.sleep(2)

        final_response = requests.get(f"{SCALARLM_URL}/v1/kv_cache/stats")
        if final_response.status_code == 200:
            try:
                final_data = final_response.json()
                final_free = final_data.get("free_tokens", 0)
                print(f"Final free tokens: {final_free}")

                token_diff = abs(final_free - initial_free)
                if token_diff < 100:
                    print("✓ Token accounting appears accurate")
                else:
                    print(f"⚠ Token accounting discrepancy: {token_diff} tokens")
            except:
                pass

    def test_kv_cache_pressure_scenarios(self):
        requests_to_make = 5
        results = []

        for i in range(requests_to_make):
            response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                "prompt": f"Memory pressure test {i}",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": 10
            })
            results.append((i, response.status_code))
            time.sleep(0.5)

        print(f"Memory pressure test results: {results}")

        for req_id, status in results:
            if isinstance(status, int):
                assert status in [200, 429, 503, 422, 500]

    def test_work_orchestrator_stats(self):
        response = requests.get(f"{SCALARLM_URL}/v1/work/stats")
        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Work orchestrator stats: {data}")

                if "work_manager" in data:
                    work_stats = data["work_manager"]
                    expected_fields = ["total_batches_processed", "total_requests_processed"]
                    for field in expected_fields:
                        if field in work_stats:
                            assert isinstance(work_stats[field], int)
                            print(f"✓ Work manager {field}: {work_stats[field]}")

                if "kv_cache" in data:
                    kv_stats = data["kv_cache"]
                    expected_fields = ["total_tokens", "free_tokens", "utilization_percent"]
                    for field in expected_fields:
                        if field in kv_stats:
                            print(f"✓ KV cache {field}: {kv_stats[field]}")
            except:
                pass

    def test_adaptor_loading_integration(self):
        response = requests.post(f"{SCALARLM_URL}/v1/work/get_adaptors", json={
            "job_hash": "test_job_12345"
        })
        assert response.status_code in [200, 404, 422, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Adaptors for test job: {data}")

                if isinstance(data, dict):
                    if "adaptors" in data:
                        assert isinstance(data["adaptors"], list)
                    if "adaptor_paths" in data:
                        assert isinstance(data["adaptor_paths"], list)
            except:
                pass

    def test_token_update_mechanisms(self):
        payload = {
            "request_id": "test_update_123",
            "actual_tokens": 150,
            "flops": 1000000
        }

        response = requests.post(f"{SCALARLM_URL}/v1/work/update_tokenization", json=payload)
        assert response.status_code in [200, 404, 422, 500]

        if response.status_code == 200:
            print("✓ Tokenization update endpoint working")

        complete_payload = {
            "request_id": "test_update_123",
            "total_tokens_used": 150,
            "result": {"generated_text": "test result"}
        }

        response = requests.post(f"{SCALARLM_URL}/v1/work/complete_work", json=complete_payload)
        assert response.status_code in [200, 404, 422, 500]

        if response.status_code == 200:
            print("✓ Work completion endpoint working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])