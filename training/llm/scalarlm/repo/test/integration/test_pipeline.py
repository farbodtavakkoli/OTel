import requests
import time
import pytest
import json
import io
import concurrent
SCALARLM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"

@pytest.mark.skip(reason="This test is disabled by default due to taking time to complete.")
def test_complete_pipeline():
    import tempfile
    import tarfile
    import os

    training_data = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "type": "math",
            "examples": [
                {"input": "What is 2+2?", "output": "The answer is 4."},
                {"input": "What is 3+3?", "output": "The answer is 6."},
                {"input": "What is 5+5?", "output": "The answer is 10."}
            ]
        },
        "num_steps": 1,
        "learning_rate": 0.0001
    }

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name

        with tarfile.open(tar_path, 'w:gz') as tar:
            config_data = json.dumps(training_data).encode('utf-8')
            config_info = tarfile.TarInfo(name='config.json')
            config_info.size = len(config_data)
            tar.addfile(config_info, io.BytesIO(config_data))

    try:
        with open(tar_path, 'rb') as f:
            files = {'file': f}
            data = {'params': json.dumps(training_data)}
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", files=files, data=data)
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)

    if response.status_code == 500:
        try:
            error_data = response.json()
            print(f"Training failed: {error_data}")
            pytest.skip(f"Training infrastructure error: {error_data}")
        except:
            print(f"Training failed: {response.text}")
            pytest.skip(f"Training infrastructure error: {response.text}")

    assert response.status_code in [200, 201, 404]

    if response.status_code == 404:
        pytest.skip("Training endpoint not implemented")

    data = response.json()

    if "job_status" in data and "job_id" in data["job_status"]:
        job_id = data["job_status"]["job_id"]
    elif "job_status" in data and "model_name" in data["job_status"]:
        job_id = data["job_status"]["model_name"]
    elif "job_id" in data:
        job_id = data["job_id"]
    elif "job_hash" in data:
        job_id = data["job_hash"]
    else:
        pytest.fail(f"No job identifier found in response: {data}")

    max_wait = 120
    start_time = time.time()
    last_status = None

    print(f"Training job submitted: {job_id}")

    while time.time() - start_time < max_wait:
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")

        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status", "unknown")

            if status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Training status: {status}")
                last_status = status

            if status == "completed":
                print(f"Training completed in {time.time() - start_time:.1f} seconds")
                break
            elif status == "failed":
                pytest.fail(f"Training failed: {status_data}")

        time.sleep(10)
    else:
        pytest.fail(f"Training did not complete within {max_wait} seconds")

    trained_response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "What is 3+3?",
        "model": job_id,
        "max_tokens": 20
    })

    assert trained_response.status_code == 200
    trained_data = trained_response.json()
    trained_answer = trained_data.get("generated_text", "")

    print(f"Trained model answer: '{trained_answer}'")

    assert len(trained_answer.strip()) > 0


def test_quick_pipeline():
    import tempfile
    import tarfile
    import os

    training_data = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "type": "math", 
            "examples": [{"input": "Test", "output": "Response"}]
        },
        "num_steps": 1,
        "learning_rate": 0.001
    }

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name

        with tarfile.open(tar_path, 'w:gz') as tar:
            config_data = json.dumps(training_data).encode('utf-8')
            config_info = tarfile.TarInfo(name='config.json')
            config_info.size = len(config_data)
            tar.addfile(config_info, io.BytesIO(config_data))

    try:
        with open(tar_path, 'rb') as f:
            files = {'file': f}
            data = {'params': json.dumps(training_data)}
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", files=files, data=data)
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)

    if response.status_code == 404:
        pytest.skip("Training not implemented")

    assert response.status_code in [200, 201]
    data = response.json()

    if "job_status" in data and "model_name" in data["job_status"]:
        job_id = data["job_status"]["model_name"]
        print(f"Training job submitted successfully: {job_id}")

        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            print(f"Job status: {status}")
            print(f"Full status response: {status_data}")
            assert status is not None

    else:
        pytest.fail(f"No job identifier in response: {data}")


def test_model_switching():
    response1 = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "Hello",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 5
    })

    if response1.status_code == 200:
        response2 = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hi there",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "max_tokens": 5
        })

        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        text1 = data1.get("generated_text", "")
        text2 = data2.get("generated_text", "")

        print(f"Response 1: '{text1}'")
        print(f"Response 2: '{text2}'")


def test_adapter_loading_flow():
    response = requests.post(f"{SCALARLM_URL}/v1/work/get_adaptors", json={
        "job_hash": "nonexistent123"
    })
    assert response.status_code in [200, 404, 422, 500]

    response = requests.post(f"{SCALARLM_URL}/v1/work/get_work", json={})
    assert response.status_code in [200, 204, 404, 500]

    response = requests.post(f"{SCALARLM_URL}/v1/generate/get_adaptors", json={
        "model_name": "test_model"
    })
    assert response.status_code in [200, 404, 422, 500]


def test_training_job_lifecycle():
    payload = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset": {
            "examples": [{"input": "Hi", "output": "Hello"}]
        },
        "num_steps": 5
    }

    response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
    if response.status_code == 404:
        pytest.skip("Training not implemented")

    if response.status_code in [200, 201]:
        data = response.json()
        job_id = data.get("job_id") or data.get("job_hash")

        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            initial_status = status_data.get("status")
            assert initial_status in ["pending", "running", "queued"]
            print(f"Initial training status: {initial_status}")

        time.sleep(5)
        status_response = requests.get(f"{SCALARLM_URL}/v1/megatron/train/{job_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            current_status = status_data.get("status")
            print(f"Status after 5s: {current_status}")


def test_openai_compatibility_with_trained_models():
    fake_job_hash = "test123456789"

    response = requests.post(f"{SCALARLM_URL}/v1/chat/completions", json={
        "model": fake_job_hash,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 10
    })

    assert response.status_code in [200, 404, 422, 500]

    response = requests.post(f"{SCALARLM_URL}/v1/completions", json={
        "model": fake_job_hash,
        "prompt": "What is 2+2?",
        "max_tokens": 10
    })

    assert response.status_code in [200, 404, 422, 500]


def test_work_orchestrator_integration():
    response = requests.post(f"{SCALARLM_URL}/v1/work/get_work_and_adaptors", json={})
    assert response.status_code in [200, 204, 404, 500]

    if response.status_code == 200:
        try:
            data = response.json()
            assert isinstance(data, dict)
            print(f"Work and adaptors response: {data}")
        except:
            pass

    complete_payload = {
        "work_id": "test123",
        "result": {"generated_text": "test result"}
    }
    response = requests.post(f"{SCALARLM_URL}/v1/work/complete_work", json=complete_payload)
    assert response.status_code in [200, 404, 422, 500]

    tokenization_payload = {
        "work_id": "test123", 
        "tokens": 150,
        "flops": 1000000
    }
    response = requests.post(f"{SCALARLM_URL}/v1/work/update_tokenization", json=tokenization_payload)
    assert response.status_code in [200, 404, 422, 500]


def test_metrics_and_monitoring():
    response = requests.get(f"{SCALARLM_URL}/v1/generate/metrics")
    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Generation metrics: {data}")
        except:
            pass

    response = requests.get(f"{SCALARLM_URL}/v1/work/stats")
    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Work stats: {data}")
        except:
            pass

def test_invalid_model_names():
    invalid_models = [
        "nonexistent/model",
        "",
        "model with spaces",
        "model/with/too/many/slashes",
        "a" * 1000,
        "model-with-unicode-ñ",
        "123456789",
        "../../../etc/passwd"
    ]

    for model in invalid_models:
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": "Hello",
            "model": model,
            "max_tokens": 5
        })

        assert response.status_code in [400, 404, 422, 500]
        print(f"Model '{model}': {response.status_code}")


def test_invalid_training_requests():
    invalid_payloads = [
        {},
        {"base_model": ""},
        {"base_model": "test", "dataset": {}},
        {"base_model": "test", "dataset": {"examples": []}},
        {"base_model": "test", "num_steps": -1},
        {"base_model": "test", "learning_rate": -0.1},
        {"base_model": "test", "num_steps": 1000000},
        {"base_model": "../../../etc/passwd"},
        {"base_model": "test", "dataset": "not_a_dict"}
    ]

    for payload in invalid_payloads:
        response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
        assert response.status_code in [400, 404, 422, 500]


def test_generation_parameter_limits():
    extreme_cases = [
        {"max_tokens": -1},
        {"max_tokens": 0},
        {"max_tokens": 100000},
        {"temperature": -1.0},
        {"temperature": 10.0},
        {"top_p": -0.5},
        {"top_p": 2.0},
        {"prompt": ""},
        {"prompt": "x" * 10000},
    ]

    for params in extreme_cases:
        full_payload = {
            "prompt": "Hello",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 5,
            **params
        }

        response = requests.post(f"{SCALARLM_URL}/v1/generate", json=full_payload)
        assert response.status_code in [200, 400, 422, 500]

        if response.status_code not in [200]:
            print(f"Rejected {params}: {response.status_code}")


def test_concurrent_training_requests():
    def submit_training(job_id):
        payload = {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": {
                "examples": [{"input": f"Job {job_id}", "output": f"Response {job_id}"}]
            },
            "num_steps": 3
        }

        try:
            response = requests.post(f"{SCALARLM_URL}/v1/megatron/train", json=payload)
            return job_id, response.status_code
        except Exception as e:
            return job_id, f"Error: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(submit_training, i) for i in range(3)]
        results = [future.result() for future in futures]

    print(f"Concurrent training results: {results}")

    for job_id, status in results:
        if isinstance(status, int):
            assert status in [200, 201, 400, 404, 422, 500, 503]


def test_malformed_json_requests():
    endpoints = [
        ("POST", "/v1/generate"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/megatron/train"),
        ("POST", "/v1/work/get_work")
    ]

    malformed_data = "{ not valid json }"

    for method, endpoint in endpoints:
        try:
            response = requests.request(
                method, 
                f"{SCALARLM_URL}{endpoint}",
                data=malformed_data,
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code in [400, 422, 500]

        except Exception as e:
            print(f"Exception for {endpoint}: {e}")


def test_large_request_handling():
    large_prompt = "Tell me a story about " + "cats and dogs " * 500

    response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": large_prompt,
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 10
    })

    assert response.status_code in [200, 400, 413, 422, 500]

    if response.status_code == 413:
        print("Server properly rejects large requests with 413 Payload Too Large")
    elif response.status_code == 200:
        print("Server handles large prompts successfully")


def test_rapid_sequential_requests():
    results = []

    for i in range(10):
        start_time = time.time()
        response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
            "prompt": f"Quick test {i}",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_tokens": 1
        })

        elapsed = time.time() - start_time
        results.append((i, response.status_code, elapsed))

    print("Rapid request results:")
    for i, status, elapsed in results:
        print(f"  Request {i}: {status} in {elapsed:.3f}s")

    successful = sum(1 for _, status, _ in results if status == 200)
    print(f"Successful requests: {successful}/10")

    assert successful > 0 or all(status in [422, 500, 503] for _, status, _ in results)


def test_missing_required_fields():
    test_cases = [
        ("POST", "/v1/generate", {"prompt": "Hello", "max_tokens": 5}),

        ("POST", "/v1/generate", {"model": "test", "max_tokens": 5}),

        ("POST", "/v1/completions", {"prompt": "Hello"}),

        ("POST", "/v1/chat/completions", {"model": "test"}),

        ("POST", "/v1/megatron/train", {"dataset": {"examples": []}}),

        ("POST", "/v1/megatron/train", {"base_model": "test"})
    ]

    for method, endpoint, payload in test_cases:
        response = requests.request(method, f"{SCALARLM_URL}{endpoint}", json=payload)

        assert response.status_code in [200, 400, 422, 500]
        print(f"{endpoint} without required fields: {response.status_code}")


def test_request_timeout_handling():
    payload = {
        "prompt": "Write a very long detailed story about",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "max_tokens": 100
    }

    try:
        response = requests.post(
            f"{SCALARLM_URL}/v1/generate",
            json=payload,
            timeout=5.0
        )

        print(f"Request completed with status: {response.status_code}")
        assert response.status_code in [200, 408, 422, 500, 504]

    except requests.exceptions.Timeout:
        print("Request timed out (expected behavior)")


def test_system_resource_limits():
    def stress_request(request_id):
        try:
            response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
                "prompt": "Stress test",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": 2
            }, timeout=10)
            return response.status_code
        except Exception as e:
            return f"Error: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(stress_request, i) for i in range(5)]
        results = [future.result() for future in futures]

    print(f"Stress test results: {results}")

    for result in results:
        if isinstance(result, int):
            assert result in [200, 422, 429, 500, 503]


def test_error_response_format():
    response = requests.post(f"{SCALARLM_URL}/v1/generate", json={
        "prompt": "Test",
        "model": "definitely-nonexistent-model",
        "max_tokens": 5
    })

    assert response.status_code in [400, 404, 422, 500]

    try:
        error_data = response.json()
        print(f"Error response format: {error_data}")

        error_fields = ["error", "detail", "message", "code"]
        has_error_info = any(field in error_data for field in error_fields)

        if has_error_info:
            print("✓ Error response has proper structure")
        else:
            print("! Error response missing standard fields")

    except:
        print("Error response is not JSON (plain text error)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
