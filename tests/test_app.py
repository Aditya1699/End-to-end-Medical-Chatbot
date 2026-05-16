from pathlib import Path

from app import create_app
from src.config import AppConfig


class DummyChain:
    def invoke(self, payload):
        return {"result": f"Answer for {payload['query']}"}


def make_app(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    config = AppConfig(base_dir=tmp_path, pinecone_api_key="test-key", model_path=model)
    flask_app = create_app(config)
    flask_app.qa_chain = DummyChain()
    flask_app.config.update(TESTING=True)
    return flask_app


def test_health_endpoint(tmp_path):
    client = make_app(tmp_path).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_chat_requires_message(tmp_path):
    client = make_app(tmp_path).test_client()

    response = client.post("/get", json={"msg": ""})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_chat_returns_json_answer(tmp_path):
    client = make_app(tmp_path).test_client()

    response = client.post("/get", json={"msg": "What is fever?"})

    assert response.status_code == 200
    assert response.get_json() == {"answer": "Answer for What is fever?"}
