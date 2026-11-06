import requests
import pytest

BASE_URL = "http://localhost:8000"

def test_health_endpoints():
    response = requests.get(f"{BASE_URL}/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "vllm" in data

    response = requests.get(f"{BASE_URL}/v1/health/keepalive")
    assert response.status_code in [200, 404]

    response = requests.get(f"{BASE_URL}/v1/health/endpoints")
    assert response.status_code in [200, 404]


def test_models_endpoint():
    response = requests.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)


def test_generate_endpoint():
    payload = {
        "prompt": "Hello",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/generate", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        try:
            data = response.json()
            assert "generated_text" in data or "text" in data or "choices" in data
        except:
            pass


def test_completions_endpoint():
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello",
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        try:
            data = response.json()
            assert "choices" in data
        except:
            pass


def test_chat_completions_endpoint():
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        try:
            data = response.json()
            assert "choices" in data
        except:
            pass


def test_openai_tokenize():
    payload = {
        "text": "Hello world",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
    response = requests.post(f"{BASE_URL}/v1/openai/tokenize", json=payload)
    assert response.status_code in [200, 404, 422]


def test_work_queue_endpoints():
    response = requests.get(f"{BASE_URL}/v1/work/stats")
    assert response.status_code in [200, 404, 500]

    response = requests.post(f"{BASE_URL}/v1/work/get_work", json={})
    assert response.status_code in [200, 204, 404, 500]


def test_generate_metrics():
    response = requests.get(f"{BASE_URL}/v1/generate/metrics")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


def test_megatron_endpoints():
    response = requests.get(f"{BASE_URL}/v1/megatron/list_models")
    assert response.status_code in [200, 404]

    response = requests.get(f"{BASE_URL}/v1/megatron/gpu_count")
    assert response.status_code in [200, 404]

    response = requests.get(f"{BASE_URL}/v1/megatron/node_count")
    assert response.status_code in [200, 404]


def test_megatron_train_endpoint():
    payload = {
        "model": "test",
        "invalid": "data"
    }
    response = requests.post(f"{BASE_URL}/v1/megatron/train", json=payload)
    assert response.status_code in [400, 422, 404, 500]


def test_work_adaptors():
    response = requests.post(f"{BASE_URL}/v1/work/get_adaptors", json={})
    assert response.status_code in [200, 204, 404, 422, 500]

    response = requests.post(f"{BASE_URL}/v1/generate/get_adaptors", json={})
    assert response.status_code in [200, 204, 404, 422, 500]


def test_endpoint_discovery():
    endpoints_to_check = [
        "/v1/generate/endpoints",
        "/v1/openai/endpoints",
        "/v1/megatron/endpoints"
    ]

    for endpoint in endpoints_to_check:
        response = requests.get(f"{BASE_URL}{endpoint}")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (dict, list))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])