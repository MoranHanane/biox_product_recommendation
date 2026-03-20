from pathlib import Path
from typing import List, Optional

import os
import pandas as pd
import pytest

from dotenv import load_dotenv
from fastapi import FastAPI, Security, HTTPException, Query
from fastapi.security import APIKeyHeader
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import json
from datetime import datetime
from pathlib import Path


from product_suggestions_V5 import make_product_suggestions_for_customer

#LOGS et GESTIONS PATHS 
#########
LOG_FILE = Path("logs_api.jsonl")

def log_prediction(event_type, payload):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "payload": payload
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path

# Chargement .env
##################

load_dotenv()

API_KEY = os.getenv("BIOX_API_KEY")
MASTER_XLSX = os.getenv("BIOX_MASTER_XLSX", "product_catalog_V11.xlsx")
INDEX_CSV = os.getenv("BIOX_INDEX_CSV", "data/mpnet-product_similarity_index.csv")
MODEL_NAME = os.getenv("BIOX_MODEL_NAME", "mpnet")

if not API_KEY:
    raise RuntimeError("BIOX_API_KEY absent du fichier .env")


# App FastAPI
##############

app = FastAPI(
    title="Bio-X Recommendation API",
    version="2.0.0",
    description="API REST présentant le fonctionnement du moteur de recommandations de produits Bio-X."
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide ou absente.")
    return api_key



# Schémas
############

class RecommendationRequest(BaseModel):
    references: List[str] = Field(..., min_length=1)
    top_k: int = Field(default=1, ge=1, le=5)



# Helpers
############

def load_master_df() -> pd.DataFrame:
    path = Path(MASTER_XLSX)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Master introuvable: {path}")
    return pd.read_excel(path, sheet_name="master_file")


def load_similarity_index() -> pd.DataFrame:
    path = resolve_path(INDEX_CSV)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Index introuvable: {path}")
    return pd.read_csv(path, low_memory=False)


def build_lines_from_references(master_df: pd.DataFrame, references: List[str]) -> pd.DataFrame:
    needed = {"Reference", "product_code_raw", "product_code_base"}
    missing = needed - set(master_df.columns)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Colonnes manquantes dans master_file: {sorted(missing)}"
        )

    refs_clean = [r.strip() for r in references if isinstance(r, str) and r.strip()]
    lines = master_df[master_df["Reference"].astype(str).str.strip().isin(refs_clean)].copy()

    if lines.empty:
        raise HTTPException(
            status_code=404,
            detail="Aucune des références demandées n'a été trouvée dans le master."
        )

    lines["CA_N"] = 1.0
    lines["CA_N1"] = 0.0

    required_lines_cols = ["product_code_raw", "product_code_base", "Reference", "CA_N", "CA_N1"]
    return lines[required_lines_cols]



# Routes
############

@app.get("/health")
def health(_: str = Security(verify_key)):
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "master_file": MASTER_XLSX,
        "index_file": INDEX_CSV
    }


@app.get("/models")
def models(_: str = Security(verify_key)):
    return {
        "available_models": ["mpnet", "e5", "bge", "scibert"],
        "active_model": MODEL_NAME
    }


@app.post("/recommendations/by-reference")
def recommendations_by_reference(req: RecommendationRequest, _: str = Security(verify_key)):

    log_prediction("request", {
        "references": req.references,
        "top_k": req.top_k
    })

    try:
        master_df = load_master_df()
        sim_idx = load_similarity_index()
        lines = build_lines_from_references(master_df, req.references)

        result = make_product_suggestions_for_customer(
            lines=lines,
            similarity_index=sim_idx,
            master_df=master_df,
            top_k=req.top_k,
        )

        output = result.to_dict(orient="records")

        log_prediction("response", {
            "nb_results": len(output)
        })

        return output

    except Exception as e:
        log_prediction("error", {
            "error": str(e)
        })
        raise


# Tests
###############

client = TestClient(app)


def test_health_ok(monkeypatch):
    monkeypatch.setenv("BIOX_API_KEY", API_KEY)
    r = client.get("/health", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_ok():
    r = client.get("/models", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data
    assert "active_model" in data


def test_auth_forbidden():
    r = client.get("/health", headers={"X-API-Key": "wrong_key"})
    assert r.status_code == 403


def test_recommendations_by_reference():
    master_path = resolve_path(MASTER_XLSX)
    index_path = resolve_path(INDEX_CSV)

    if not master_path.exists() or not index_path.exists():
        pytest.skip("Master ou index absent localement pour ce test.")

    master_df = pd.read_excel(master_path, sheet_name="master_file")
    if master_df.empty:
        pytest.skip("Master vide.")

    sample_ref = str(master_df["Reference"].dropna().iloc[0]).strip()

    r = client.post(
        "/recommendations/by-reference",
        json={
            "references": [sample_ref],
            "top_k": 1
        },
        headers={"X-API-Key": API_KEY}
    )

    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)

    if len(data) > 0:
        assert "Reference_A" in data[0]
        assert data[0]["Reference_A"] == sample_ref