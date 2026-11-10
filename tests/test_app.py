import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Runs lifespan startup/shutdown, which loads frost_model.json
    with TestClient(app) as client:
        yield client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "git_sha" in data


def test_predict_returns_probability_and_threshold(client):
    response = client.post(
        "/predict",
        json={
            "air_temp_c": 0,
            "dew_point_c": -1,
            "wind_speed_ms": 3,
            "cloud_cover_pct": 50,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert 0.0 <= data["frost_probability"] <= 1.0
    assert data["threshold"] == 0.35
    assert isinstance(data["alert"], bool)
    assert "model_version" in data


def test_frosty_reading_scores_higher_than_mild(client):
    frosty = {
        "air_temp_c": -5,
        "dew_point_c": -6,
        "wind_speed_ms": 1,
        "cloud_cover_pct": 5,
    }

    mild = {
        "air_temp_c": 15,
        "dew_point_c": 10,
        "wind_speed_ms": 3,
        "cloud_cover_pct": 50,
    }

    frosty_response = client.post("/predict", json=frosty)
    mild_response = client.post("/predict", json=mild)

    assert frosty_response.status_code == 200
    assert mild_response.status_code == 200

    frosty_data = frosty_response.json()
    mild_data = mild_response.json()

    assert (
        frosty_data["frost_probability"]
        > mild_data["frost_probability"]
    )

    assert frosty_data["alert"] is True
    assert mild_data["alert"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {
            "air_temp_c": 0,
            "dew_point_c": 0,
            "wind_speed_ms": -1,
            "cloud_cover_pct": 50,
        },
        {
            "air_temp_c": 0,
            "dew_point_c": 0,
            "wind_speed_ms": 5,
            "cloud_cover_pct": 101,
        },
    ],
)
def test_invalid_readings_return_422(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_startup_loads_model(client):
    response = client.post(
        "/predict",
        json={
            "air_temp_c": 1,
            "dew_point_c": 0,
            "wind_speed_ms": 2,
            "cloud_cover_pct": 25,
        },
    )

    # If startup didn't load the model, this endpoint would fail.
    assert response.status_code == 200
