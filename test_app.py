from pathlib import Path

import pandas as pd
import pytest

from app import app
from train import load_dataset


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_index(client):
    assert client.get("/").status_code == 200


def test_empty_prediction(client):
    response = client.post("/predict", json={"text": " "})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_prediction_requires_trained_model(client, tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "missing.joblib")
    app_module.model = None
    response = client.post("/predict", json={"text": "A news article."})
    assert response.status_code == 503
    app_module.model = None


def test_dataset_combines_title_and_body(tmp_path):
    path = tmp_path / "articles.csv"
    pd.DataFrame({
        "title": ["Fake title", "Real title", "Fake story", "Real story"],
        "text": ["body one", "body two", "body three", "body four"],
        "label": ["fake", "real", "fake", "real"],
    }).to_csv(path, index=False)
    texts, labels = load_dataset(path)
    assert texts.iloc[0] == "Fake title body one"
    assert set(labels) == {"fake", "real"}
