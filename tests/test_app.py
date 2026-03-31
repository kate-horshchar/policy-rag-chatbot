import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.web import app


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
