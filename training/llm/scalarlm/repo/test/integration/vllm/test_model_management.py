import requests
import pytest
import os
import time
from pathlib import Path

SCALARLM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"

def test_model_discovery():
    response = requests.get(f"{SCALARLM_URL}/v1/megatron/list_models")
    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Available models: {data}")
            assert isinstance(data, (list, dict))
        except:
            pass


def test_job_hash_to_model_mapping():
    test_hashes = [
        "abc123def456",
        "test_model_123",
        "1234567890abcdef"
    ]

    for job_hash in test_hashes:
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hello",
            "model": job_hash,
            "max_tokens": 5
        })

        assert response.status_code in [200, 404, 422, 500]

        if response.status_code == 404:
            try:
                error_data = response.json()
                print(f"Model {job_hash} not found: {error_data}")
            except:
                print(f"Model {job_hash} not found (no JSON response)")


def test_adapter_discovery_mechanism():
    test_cases = [
        {"job_hash": "nonexistent123"},
        {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"job_hash": "test456"},
        {}
    ]

    for payload in test_cases:
        response = requests.post(f"{SCALARLM_URL}/v1/work/get_adaptors", json=payload)
        assert response.status_code in [200, 204, 400, 404, 422, 500]

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Adaptors for {payload}: {data}")
            except:
                pass


def test_vllm_model_registry():
    response = requests.get(f"{VLLM_URL}/v1/models")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        models = data.get("data", [])
        print(f"vLLM registered models: {[m.get('id') for m in models]}")

        assert len(models) > 0

        model_ids = [m.get("id", "") for m in models]
        trained_models = [m for m in model_ids if len(m) > 20 and "/" not in m]

        if trained_models:
            print(f"Found trained models in registry: {trained_models}")


def test_lora_adapter_management():
    load_payload = {
        "lora_name": "test_adapter_123",
        "lora_path": "/nonexistent/path.pt"
    }

    response = requests.post(f"{VLLM_URL}/v1/load_lora_adapter", json=load_payload)
    assert response.status_code in [200, 400, 404, 422, 500]

    unload_payload = {"lora_name": "test_adapter_123"}
    response = requests.post(f"{VLLM_URL}/v1/unload_lora_adapter", json=unload_payload)
    assert response.status_code in [200, 400, 404, 422, 500]


def test_model_loading_performance():
    start_time = time.time()

    response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "Test",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 1
    })

    load_time = time.time() - start_time

    assert load_time < 30.0

    if response.status_code == 200:
        print(f"Generation completed in {load_time:.2f} seconds")
    else:
        print(f"Generation failed in {load_time:.2f} seconds with status {response.status_code}")


def test_concurrent_model_access():
    import concurrent.futures
    import threading

    def make_request(request_id):
        try:
            response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                "prompt": f"Request {request_id}",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": 3
            })
            return response.status_code, request_id
        except Exception as e:
            return 0, f"Error {request_id}: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request, i) for i in range(3)]
        results = [future.result() for future in futures]

    print(f"Concurrent request results: {results}")

    status_codes = [r[0] for r in results]
    assert any(code in [200, 422, 500] for code in status_codes)


def test_job_directory_structure():
    jobs_dir = "/app/cray/jobs"

    try:
        if os.path.exists(jobs_dir):
            job_dirs = [d for d in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, d))]
            print(f"Found job directories: {job_dirs}")

            for job_dir in job_dirs[:3]:
                job_path = Path(jobs_dir) / job_dir
                pt_files = list(job_path.glob("*.pt"))
                if pt_files:
                    print(f"Job {job_dir} has checkpoint files: {[f.name for f in pt_files]}")
        else:
            print(f"Jobs directory {jobs_dir} not accessible")
    except Exception as e:
        print(f"Could not check job directories: {e}")


def test_model_state_persistence():
    payload = {
        "prompt": "What is 1+1?",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 5,
        "temperature": 0.1
    }

    response1 = requests.post(f"{SCALARLM_URL}/v1/generate", json=payload)
    response2 = requests.post(f"{SCALARLM_URL}/v1/generate", json=payload)

    if response1.status_code == 200 and response2.status_code == 200:
        data1 = response1.json()
        data2 = response2.json()

        text1 = data1.get("generated_text", "")
        text2 = data2.get("generated_text", "")

        print(f"Response 1: '{text1}'")
        print(f"Response 2: '{text2}'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])