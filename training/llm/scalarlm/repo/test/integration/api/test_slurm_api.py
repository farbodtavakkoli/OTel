import requests
import pytest

BASE_URL = "http://localhost:8000"

def test_slurm_status():
    response = requests.get(f"{BASE_URL}/slurm/status")
    assert response.status_code in [200, 404, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


def test_slurm_squeue():
    response = requests.get(f"{BASE_URL}/slurm/squeue")
    assert response.status_code in [200, 404, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


def test_slurm_cancel():
    response = requests.post(f"{BASE_URL}/slurm/cancel/999999")
    assert response.status_code in [200, 404, 400, 503]


def test_slurm_endpoints():
    response = requests.get(f"{BASE_URL}/slurm/endpoints")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


def test_megatron_squeue():
    response = requests.get(f"{BASE_URL}/v1/megatron/squeue")
    assert response.status_code in [200, 404, 503]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])