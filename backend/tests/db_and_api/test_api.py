from fastapi.testclient import TestClient
from backend.app.api import app

client = TestClient(app)

def test_registration_endpoint():
    response = client.post(
        "/registration",
        json={
            "username": "testuser_api",
            "email": "api_test@example.com",
            "password": "strongpassword123"
        }
    )

    assert response.status_code in [200, 201]
    data = response.json()
    assert data["username"] == "testuser_api"
    assert 'password' not in data