from app_flask import app

def test_flask_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200

def test_flask_home_get():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200