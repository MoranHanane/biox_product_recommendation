# Bio-X Product Recommendation

## Description
Application de recommandation de produits basée sur la similarité sémantique entre références produits.

## Structure
- API.py : API FastAPI (endpoint /recommendations)
- Flask.py : interface utilisateur
- product_similarity_V5.py : génération de l’index de similarité
- product_suggestions_V5.py : moteur de recommandation
- tests_integres.py : tests automatisés
- data/ : fichiers de données (index, diagnostics, outputs)

## Installation
pip install -r requirements.txt

## Lancement

### API
uvicorn API:app --reload  
Accès : http://127.0.0.1:8000/docs

Authentification : clé API via header `X-API-Key`

### Frontend
flask --app Flask.py run

## Données utilisées
- mpnet-product_similarity_index.csv : index de similarité
- mpnet-customer_product_suggestions.xlsx : résultats

## Tests
pytest tests/


## Frontend Flask

Le frontend utilise un template HTML situé dans /templates/index.html.