import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.web import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model" in data
    assert "index_size" in data


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Policy Assistant" in response.data


def test_chat_missing_question(client):
    response = client.post("/chat", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_chat_empty_question(client):
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_chat_no_body(client):
    response = client.post("/chat", content_type="application/json", data="")
    assert response.status_code == 400


def test_policy_document_served(client):
    response = client.get("/policies/pto_policy.md")
    assert response.status_code == 200
    assert b"PTO" in response.data


def test_policy_document_unknown(client):
    response = client.get("/policies/does_not_exist.md")
    assert response.status_code == 404


def test_policy_document_rejects_traversal(client):
    response = client.get("/policies/..%2F..%2F.env")
    assert response.status_code == 404
