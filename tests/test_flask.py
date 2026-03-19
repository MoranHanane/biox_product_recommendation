from unittest.mock import patch, Mock
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["BIOX_API_KEY"] = "dummy_key"
os.environ["FLASK_SECRET_KEY"] = "test_secret"

from app_flask import app

def login_session(client, role="commercial"):
    """
    Initialise une session utilisateur simulée pour les tests Flask.

    Permet de bypasser la page de login en injectant directement
    les variables de session nécessaires.

    Paramètres :
    - client : client de test Flask
    - role (str) : rôle simulé ("commercial" par défaut)

    Effets :
    - session["username"] = "tester"
    - session["role"] = role
    """
    with client.session_transaction() as sess:
        sess["username"] = "tester"
        sess["role"] = role


def test_login_page_accessible():
    """
    Vérifie que la page de login est accessible sans authentification.

    Attendu :
    - Code HTTP 200
    """
    client = app.test_client()
    r = client.get("/login")
    assert r.status_code == 200


def test_index_requires_login():
    """
    Vérifie que la route "/" est protégée.

    Attendu :
    - Redirection (302) vers /login si non authentifié
    """
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 302
    assert "/login" in r.location

def test_index_access_with_role():
    """
    Vérifie qu’un utilisateur authentifié avec un rôle valide
    peut accéder à la page principale.

    Attendu :
    - Code HTTP 200
    """
    client = app.test_client()
    login_session(client, role="commercial")

    r = client.get("/")
    assert r.status_code == 200

@patch("app_flask.requests.post")
def test_flask_to_api_integration_success(mock_post):
    """
    Vérifie le bon fonctionnement du flux Flask → API en cas de succès.

    Simule une réponse API valide (200) et vérifie :
    - appel correct de requests.post
    - affichage des résultats côté Flask
    """
    client = app.test_client()
    login_session(client)

    fake_response = Mock()
    fake_response.status_code = 200
    fake_response.json.return_value = [{"Reference_A": "A", "Reference_B": "B"}]
    mock_post.return_value = fake_response

    r = client.post("/", data={
        "references": "[A] Produit A",
        "top_k": "1"
    })

    assert r.status_code == 200
    mock_post.assert_called_once()


@patch("app_flask.requests.post")
def test_flask_to_api_integration_api_error(mock_post):
    """
    Vérifie la gestion des erreurs retournées par l’API.

    Simule une réponse API 500 et vérifie :
    - affichage d’un message d’erreur utilisateur
    """
    client = app.test_client()
    login_session(client)

    fake_response = Mock()
    fake_response.status_code = 500
    fake_response.text = "Erreur interne"
    mock_post.return_value = fake_response

    r = client.post("/", data={
        "references": "[A] Produit A",
        "top_k": "1"
    })

    assert r.status_code == 200
    assert b"Erreur API 500" in r.data


@patch("app_flask.requests.post", side_effect=Exception("API indisponible"))
def test_flask_to_api_integration_connection_error(mock_post):
    """
    Vérifie la gestion des erreurs de connexion à l’API.

    Simule une exception réseau et vérifie :
    - affichage d’un message "Erreur de connexion"
    """
    client = app.test_client()
    login_session(client)

    r = client.post("/", data={
        "references": "[A] Produit A",
        "top_k": "1"
    })

    assert r.status_code == 200
    assert b"Erreur de connexion" in r.data


def test_top_k_validation_non_integer():
    client = app.test_client()
    login_session(client)

    r = client.post("/", data={
        "references": "[A] Produit A",
        "top_k": "abc"
    })

    assert r.status_code == 200
    assert b"top_k doit etre un entier" in r.data or b"top_k doit" in r.data


def test_top_k_validation_range():
    client = app.test_client()
    login_session(client)

    r = client.post("/", data={
        "references": "[A] Produit A",
        "top_k": "99"
    })

    assert r.status_code == 200
    assert b"compris entre 1 et 5" in r.data


def test_empty_references_validation():
    client = app.test_client()
    login_session(client)

    r = client.post("/", data={
        "references": "",
        "top_k": "1"
    })

    assert r.status_code == 200
    assert b"au moins une reference" in r.data or b"au moins une" in r.data

def test_flask_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 302
    assert "/login" in r.location

def test_flask_home_get():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 302
    assert "/login" in r.location
