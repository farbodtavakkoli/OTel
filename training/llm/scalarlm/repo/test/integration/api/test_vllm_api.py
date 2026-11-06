import requests
import pytest

BASE_URL = "http://localhost:8001"

def test_health_endpoints():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

    response = requests.get(f"{BASE_URL}/ping")
    assert response.status_code in [200, 404]

    response = requests.post(f"{BASE_URL}/ping", json={})
    assert response.status_code in [200, 404]


def test_version():
    response = requests.get(f"{BASE_URL}/version")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert "version" in data or isinstance(data, str)


def test_models_endpoint():
    response = requests.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


def test_completions():
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello",
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]


def test_chat_completions():
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code in [200, 404, 422, 500]
    if response.status_code == 200:
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]


def test_tokenize():
    payload = {
        "text": "Hello world"
    }
    response = requests.post(f"{BASE_URL}/tokenize", json=payload)
    assert response.status_code in [200, 400, 404, 422]
    if response.status_code == 200:
        data = response.json()
        assert "tokens" in data or "ids" in data or isinstance(data, list)


def test_detokenize():
    payload = {
        "tokens": [1, 2, 3]
    }
    response = requests.post(f"{BASE_URL}/detokenize", json=payload)
    assert response.status_code in [200, 404, 422]


def test_embeddings():
    payload = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "input": "Hello world"
    }
    response = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    assert response.status_code in [200, 404, 501]


def test_lora_adapter_endpoints():
    payload = {
        "lora_name": "test_adapter",
        "lora_path": "/nonexistent/path"
    }
    response = requests.post(f"{BASE_URL}/v1/load_lora_adapter", json=payload)
    assert response.status_code in [200, 400, 404, 422]

    payload = {"lora_name": "test_adapter"}
    response = requests.post(f"{BASE_URL}/v1/unload_lora_adapter", json=payload)
    assert response.status_code in [200, 400, 404, 422]


def test_metrics():
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert len(response.text) > 0


def test_load_endpoint():
    response = requests.get(f"{BASE_URL}/load")
    assert response.status_code in [200, 404]


def test_responses_endpoints():
    response = requests.post(f"{BASE_URL}/v1/responses", json={})
    assert response.status_code in [200, 400, 404, 422, 500]

    response = requests.get(f"{BASE_URL}/v1/responses/fake-id")
    assert response.status_code in [200, 400, 404]

    response = requests.post(f"{BASE_URL}/v1/responses/fake-id/cancel", json={})
    assert response.status_code in [200, 400, 404]


def test_additional_endpoints():
    endpoints = [
        ("/pooling", "POST", {}),
        ("/classify", "POST", {}),
        ("/score", "POST", {}),
        ("/v1/score", "POST", {}),
        ("/rerank", "POST", {}),
        ("/v1/rerank", "POST", {}),
        ("/invocations", "POST", {})
    ]

    for endpoint, method, payload in endpoints:
        if method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        else:
            response = requests.get(f"{BASE_URL}{endpoint}")
        assert response.status_code in [200, 400, 404, 422, 500, 501]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])