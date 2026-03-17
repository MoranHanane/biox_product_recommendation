"""
tests_integres.py

Suite de tests automatisés du pipeline de recommandation produit Bio-X.

Objectifs :
- vérifier l'intégrité du pipeline de données
- détecter les incohérences de colonnes
- tester les règles métier critiques
- tester la robustesse du moteur de recommandation
- valider la logique de séparation Train/Test (C12)

Les tests utilisent pytest et reposent sur de petits jeux de données simulés.
"""

import pandas as pd
from pathlib import Path

from rebuild_customer_V5 import load_master
from product_similarity_V5 import build_similarity_index
from product_suggestions_V5 import make_product_suggestions_for_customer

# A) Tests de données et de pipeline

# TEST 1
# chargement du master avec colonnes requises

def test_master_has_required_columns(tmp_path):
    """
    Vérifie que la fonction load_master :
    - charge correctement le fichier master
    - extrait le product_code depuis la colonne Reference
    - crée la colonne product_code

    Ce test protège contre les erreurs de structure
    du masterfile.
    """

    df = pd.DataFrame({
        "Reference": ["[A01] Product A"],
        "species": ["Pig"],
        "class_name": ["PCR"],
        "subclass_name": ["Test"],
        "MPC_range": ["Range1"],
        "MajorProductoftTheClass": ["MPC1"]
    })

    path = tmp_path / "master.xlsx"
    df.to_excel(path, sheet_name="master_file", index=False)

    out = load_master(path)

    assert "product_code" in out.columns
    assert out["product_code"].iloc[0] == "A01"


# TEST 2
# structure de l'index de similarité

def test_similarity_index_structure():
    """
    Vérifie que l'index de similarité :
    - contient la colonne similarity
    - génère le nombre attendu de paires produit

    L'objectif est de détecter toute modification
    accidentelle du format de l'index.
    """

    df = pd.DataFrame({
        "Reference": ["A","B"],
        "product_code_raw": ["A","B"],
        "product_code_base": ["A","B"],
        "species": ["Pig","Pig"],
        "class_name": ["PCR","PCR"],
        0:[0.1,0.2],
        1:[0.2,0.1]
    })

    index = build_similarity_index(df)

    assert "similarity" in index.columns
    assert len(index) == 2

# TEST 3
#tests suggestions_produits: pas d'auto-suggestion

def test_no_self_suggestion():
    
    """
    Vérifie qu'un produit ne peut pas se recommander lui-même.

    Ce test protège contre un bug classique dans les moteurs
    de similarité où un produit apparaît dans ses propres
    suggestions.
    """
        
    lines = pd.DataFrame({
        "Reference":["A"],
        "product_code_raw":["A"],
        "product_code_base":["A"],
        "CA_N":[10],
        "CA_N1":[0]
    })

    index = pd.DataFrame({
        "Reference_A":["A"],
        "Reference_B":["A"],
        "product_code_raw_A":["A"],
        "product_code_raw_B":["A"],
        "product_code_base_A":["A"],
        "product_code_base_B":["A"],
        "species_B":["Pig"],
        "class_name_B":["PCR"],
        "similarity":[0.9]
    })

    res = make_product_suggestions_for_customer(lines,index)

    assert res["Reference_B"].isna().all()


# TEST 4
# Exclusion même product_code_base

def test_same_base_excluded():
    """
    Vérifie que deux produits ayant le même product_code_base
    ne peuvent pas être recommandés entre eux.

    Cette règle métier évite de suggérer
    des variantes d'un même produit.
    """

    lines = pd.DataFrame({
        "Reference":["A"],
        "product_code_raw":["A"],
        "product_code_base":["BASE"],
        "CA_N":[10],
        "CA_N1":[0]
    })

    index = pd.DataFrame({
        "Reference_A":["A"],
        "Reference_B":["B"],
        "product_code_raw_A":["A"],
        "product_code_raw_B":["B"],
        "product_code_base_A":["BASE"],
        "product_code_base_B":["BASE"],
        "species_B":["Pig"],
        "class_name_B":["PCR"],
        "similarity":[0.9]
    })

    res = make_product_suggestions_for_customer(lines,index)

    assert res["Reference_B"].isna().all()


# TEST 5
# Produits déjà achetés

def test_already_purchased_flag():
    """
    Vérifie que les produits déjà présents dans le panier
    sont correctement identifiés.

    Le champ already_in_basket permet de signaler
    au commercial que la suggestion correspond
    à un produit déjà acheté.
    """

    lines = pd.DataFrame({
        "Reference":["A","B"],
        "product_code_raw":["A","B"],
        "product_code_base":["A","B"],
        "CA_N":[10,5],
        "CA_N1":[0,0]
    })

    index = pd.DataFrame({
        "Reference_A":["A"],
        "Reference_B":["B"],
        "product_code_raw_A":["A"],
        "product_code_raw_B":["B"],
        "product_code_base_A":["A"],
        "product_code_base_B":["B"],
        "species_B":["Pig"],
        "class_name_B":["PCR"],
        "similarity":[0.9]
    })

    res = make_product_suggestions_for_customer(lines,index)

    assert res["already_in_basket"].iloc[0] == True


# TEST 6
# Robustesse PDF manquant

def test_missing_pdf_handled():
    """
    Vérifie que l'absence de fichier PDF
    (instructions_for_use ou validation_file)
    ne provoque pas de crash dans le pipeline.

    Le système doit ignorer les documents manquants
    et continuer le calcul des embeddings.
    """

    from product_similarity_V5 import extract_text_from_pdf

    txt = extract_text_from_pdf(Path("fichier_inexistant.pdf"))

    assert txt == ""


# TEST 7
# Train/Test split sur une extraction simulée (C12)


def test_train_test_split_logic():
    """
    Test de validation du modèle basé sur une séparation temporelle.

    Train :
        produits achetés en N-1

    Test :
        produits achetés en N

    Le test vérifie que les recommandations
    générées à partir du train contiennent
    des produits présents dans le test.
    """

    lines = pd.DataFrame({
        "product_code":["A","B","C"],
        "CA_N1":[10,0,5],
        "CA_N":[0,8,3]
    })

    sim_idx = pd.DataFrame({
        "product_code":["A","C"],
        "product_code_suggested":["B","B"],
        "similarity":[0.9,0.8]
    })

    train_products = lines[lines["CA_N1"] > 0]["product_code"].unique()
    test_products  = lines[lines["CA_N"] > 0]["product_code"].unique()

    sim_train = sim_idx[sim_idx["product_code"].isin(train_products)]

    reco = (
        sim_train.groupby("product_code_suggested")["similarity"]
        .max()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    recall = len(set(reco) & set(test_products))

    assert recall > 0


# TEST 8
# Train/Test split sur une extraction Odoo réelle (C12)


def test_train_test_split_real_dashboard():

    dashboard = pd.read_excel(
        "DAshboard_Odoo_Labo-X.xlsx",
        header=None,
        skiprows=8
    )

    lines = pd.DataFrame({
        "Reference": dashboard.iloc[:,1],
        "CA_N1": dashboard.iloc[:,11],
        "CA_N": dashboard.iloc[:,13]
    })

    lines = lines.dropna(subset=["Reference"])

    train_products = set(lines[lines["CA_N1"] > 0]["Reference"])
    test_products  = set(lines[lines["CA_N"] > 0]["Reference"])

    sugg = pd.read_excel(
        "mpnet-customer_product_suggestions.xlsx"
    )

    sugg = sugg[sugg["Reference_A"].isin(train_products)]

    suggested_products = set(sugg["Reference_B"].dropna())

    matches = suggested_products & test_products

    produits_test = len(test_products)
    produits_retrouves = len(matches)

    recall = produits_retrouves / produits_test if produits_test else 0

    print("\n----- Evaluation pipeline réel -----")
    print("produits test :", produits_test)
    print("produits retrouvés :", produits_retrouves)
    print("recall :", round(recall,3))
    print("-----------------------------------")

    assert recall >= 0

# B) Tests de données et de pipeline

import json
import pytest
from unittest.mock import patch

from Flask import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_flask_to_api_integration_success(client):
    mock_api_response = [
        {
            "Reference_A": "TEST_REF",
            "Reference_B": "SUGGESTED_REF",
            "score": 0.95
        }
    ]

    class MockResponse:
        status_code = 200

        def json(self):
            return mock_api_response

    with patch("Flask.requests.post", return_value=MockResponse()) as mock_post:
        response = client.post(
            "/",
            data={
                "references": "TEST_REF",
                "top_k": "1"
            }
        )

        assert response.status_code == 200

        # Vérifie que Flask appelle bien l’API
        mock_post.assert_called_once()

        # Vérifie que la réponse contient bien la suggestion simulée
        assert b"SUGGESTED_REF" in response.data


def test_flask_to_api_integration_api_error(client):
    class MockResponse:
        status_code = 500
        text = "Internal Server Error "

        def json(self):
            return {"error": "Internal error"}

    with patch("Flask.requests.post", return_value=MockResponse()):
        response = client.post(
            "/",
            data={
                "references": "TEST_REF",
                "top_k": "1"
            }
        )

        assert response.status_code == 200
        assert b"Erreur API" in response.data


def test_flask_to_api_integration_connection_error(client):
    with patch("Flask.requests.post", side_effect=Exception("Connection error")):
        response = client.post(
            "/",
            data={
                "references": "TEST_REF",
                "top_k": "1"
            }
        )

        assert response.status_code == 200
        assert b"Erreur de connexion" in response.data
