# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Product Similarity V5 — Multi-model + Diagnostics + Robust weights
# - Source unique : master_df (DataFrame)
# - Multi-modèles (mpnet/e5/bge/scibert) avec outputs préfixés
# - Features: catégorielles (species, class_name, MPC_range) + textuelles (ifu/validation/pathogen)
# - Redistribution non pénalisante si IFU/Validation/Pathogen manquants (presence-based scaling)
# - Diagnostics: summary + top nearest/farthest + latence
# - CSV index: "similarity" AVANT "Reference_B"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import fitz  # PyMuPDF


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# Colonnes attendues master_df
#############################

COL_REF = "Reference"
COL_CODE_RAW = "product_code_raw"
COL_CODE_BASE = "product_code_base"
COL_IFU = "instructions_for_use"
COL_VALID = "Validation_file"
COL_PATHO = "Pathogen/Target"

# Ajout master
COL_MPC_RANGE = "MPC_range"
COL_SPECIES = "species"
COL_CLASS = "class_name"


# Modèles (4)
#################
MODELS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "sentence_transformers"),
    "e5": ("intfloat/e5-large-v2", "sentence_transformers"),
    "bge": ("BAAI/bge-base-en-v1.5", "sentence_transformers"),
    "scibert": ("allenai/scibert_scivocab_uncased", "hf_transformers"),
}

CHUNK_CHARS = 2500
OVERLAP = 250

WEIGHTS = {
    "species": 4.0,
    "class_name": 2.0,
    "MPC_range": 3.0,
    "ifu": 2.0,
    "validation": 1.0,
    "pathogen": 2.0,
}


# Utils texte/pdf
##################

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def extract_text_from_pdf(path: Path) -> str:
    try:
        doc = fitz.open(path)
        txt = "\n".join(p.get_text() for p in doc)
        doc.close()
        return txt.strip()
    except Exception as e:
        logging.warning(f"PDF illisible: {path} ({e})")
        return ""



# Backends embedding
#####################

def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _load_hf_transformers(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl, torch


def _hf_encode_texts_mean_pool(texts: List[str], tok, mdl, torch, batch_size: int = 16) -> np.ndarray:
    """
    Encodage HF “mean pooling” (robuste pour SciBERT).
    """
    def mean_pool(last_hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            out = mdl(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_emb.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(all_emb)



# Embeddings: PDF links + text
###############################

def embed_pdfs_from_links(
    links: List[str],
    model_key: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Retourne embeddings + métriques latence.
    """
    t0 = time.perf_counter()

    if MODELS[model_key][1] == "sentence_transformers":
        model = _load_sentence_transformer(MODELS[model_key][0])
        dim = model.get_sentence_embedding_dimension()
        out = np.zeros((len(links), dim), dtype="float32")

        pdf_fail = 0
        for i, link in enumerate(links):
            if not isinstance(link, str) or not link.strip():
                continue
            p = Path(link)
            if not p.exists():
                pdf_fail += 1
                continue
            text = extract_text_from_pdf(p)
            chunks = chunk_text(text, CHUNK_CHARS, OVERLAP)
            if not chunks:
                pdf_fail += 1
                continue
            emb = model.encode(chunks, show_progress_bar=False)
            out[i] = emb.mean(axis=0)

        metrics = {
            "pdf_read_failures": float(pdf_fail),
            "embed_seconds_total": float(time.perf_counter() - t0),
        }
        return out, metrics

    # HF (SciBERT) : on encode les chunks avec mean-pooling
    tok, mdl, torch = _load_hf_transformers(MODELS[model_key][0])
    dim = mdl.config.hidden_size
    out = np.zeros((len(links), dim), dtype="float32")

    pdf_fail = 0
    for i, link in enumerate(links):
        if not isinstance(link, str) or not link.strip():
            continue
        p = Path(link)
        if not p.exists():
            pdf_fail += 1
            continue
        text = extract_text_from_pdf(p)
        chunks = chunk_text(text, CHUNK_CHARS, OVERLAP)
        if not chunks:
            pdf_fail += 1
            continue
        emb = _hf_encode_texts_mean_pool(chunks, tok, mdl, torch, batch_size=16)
        out[i] = emb.mean(axis=0)

    metrics = {
        "pdf_read_failures": float(pdf_fail),
        "embed_seconds_total": float(time.perf_counter() - t0),
    }
    return out, metrics


def embed_text_column(
    texts: List[str],
    model_key: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    texts = [t if isinstance(t, str) else "" for t in texts]

    if MODELS[model_key][1] == "sentence_transformers":
        model = _load_sentence_transformer(MODELS[model_key][0])
        X = model.encode(texts, show_progress_bar=True).astype("float32")
        return X, {"embed_seconds_total": float(time.perf_counter() - t0)}

    tok, mdl, torch = _load_hf_transformers(MODELS[model_key][0])
    X = _hf_encode_texts_mean_pool(texts, tok, mdl, torch, batch_size=32)
    return X, {"embed_seconds_total": float(time.perf_counter() - t0)}



# Features (one-hot + embeddings) + redistribution non pénalisante pour entrées avec doc manquante
###################################################################################################

def onehot_block(df: pd.DataFrame, col: str) -> np.ndarray:
    X = pd.get_dummies(df[col].astype("category"), prefix=col).values
    return normalize(X, axis=1)


def row_presence(X: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(X, axis=1) > 0).astype(float)


def build_feature_matrix(
    master: pd.DataFrame,
    emb_ifu: np.ndarray,
    emb_validation: np.ndarray,
    emb_pathogen: np.ndarray,
    weights: Dict[str, float],
) -> pd.DataFrame:

    df = master.copy()

    blocks, pres = {}, {}

    # catégorielles retenues
    for col in [COL_SPECIES, COL_CLASS, COL_MPC_RANGE]:
        if col in df.columns:
            X = onehot_block(df, col)
            blocks[col] = X
            pres[col] = row_presence(X)

    # embeddings
    for name, emb in {
        "ifu": emb_ifu,
        "validation": emb_validation,
        "pathogen": emb_pathogen,
    }.items():
        X = normalize(emb, axis=1)
        blocks[name] = X
        pres[name] = row_presence(X)

    keys = list(blocks.keys())
    W0 = np.array([weights[k] for k in keys], dtype="float32")
    P = np.vstack([pres[k] for k in keys]).T  # (n, nb_blocks)

    # Redistribution non pénalisante :
    # - denom = somme des poids présents
    # - scale = somme poids théoriques / denom
    # => si un produit n’a pas IFU/Validation/Pathogen, ses autres blocs reçoivent plus de poids
    denom = (P * W0).sum(axis=1, keepdims=True)
    scale = np.divide(W0.sum(), denom, out=np.ones_like(denom), where=denom != 0)

    Xs = [blocks[k] * (W0[i] * scale) for i, k in enumerate(keys)]
    feat = normalize(np.hstack(Xs), axis=1)

    meta = df[[COL_REF, COL_CODE_RAW, COL_CODE_BASE, COL_SPECIES, COL_CLASS]].copy()
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(feat)], axis=1)



# Similarity index 
# - CSV: similarity AVANT Reference_B
########################################

def build_similarity_index(feat_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [COL_REF, COL_CODE_RAW, COL_CODE_BASE, COL_SPECIES, COL_CLASS]
    meta = feat_df[meta_cols]
    X = feat_df.drop(columns=meta_cols).values

    cos = X @ X.T
    sim01 = (cos + 1.0) / 2.0

    n = len(feat_df)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows.append({
                "Reference_A": meta.iloc[i][COL_REF],
                "product_code_raw_A": meta.iloc[i][COL_CODE_RAW],
                "product_code_base_A": meta.iloc[i][COL_CODE_BASE],

                "similarity": float(sim01[i, j]),  # <-- avant Reference_B

                "Reference_B": meta.iloc[j][COL_REF],
                "product_code_raw_B": meta.iloc[j][COL_CODE_RAW],
                "product_code_base_B": meta.iloc[j][COL_CODE_BASE],

                "species_B": meta.iloc[j][COL_SPECIES],
                "class_name_B": meta.iloc[j][COL_CLASS],
            })
    return pd.DataFrame(rows)



# Diagnostics
#############

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = normalize(X, axis=1)
    return Xn @ Xn.T


def diagnostics_distribution(X: np.ndarray) -> dict:
    S = cosine_sim_matrix(X)
    np.fill_diagonal(S, np.nan)
    vals = S[(~np.isnan(S)) & (S > 0)]
    if len(vals) == 0:
        return dict(mean=np.nan, median=np.nan, min=np.nan,
                    p10=np.nan, p90=np.nan, max=np.nan)
    return dict(
        mean=float(vals.mean()),
        median=float(np.median(vals)),
        min=float(vals.min()),
        p10=float(np.percentile(vals, 10)),
        p90=float(np.percentile(vals, 90)),
        max=float(vals.max()),
    )


def diagnostics_top_pairs(X: np.ndarray, refs: List[str], field: str, topk: int, nearest: bool):
    S = cosine_sim_matrix(X)
    n = S.shape[0]
    rows = []
    for i in range(n):
        sims = S[i].copy()
        sims[i] = np.nan
        idx = np.where(sims > 0)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(-sims[idx])] if nearest else idx[np.argsort(sims[idx])]
        for j in order[:topk]:
            rows.append({
                "field": field,
                "Reference_A": refs[i],
                "Reference_B": refs[j],
                "similarity": float(sims[j]),
            })
    return pd.DataFrame(rows)



# ENTRY POINT
##############

from datetime import datetime
import json

output_dir= r"D:\Moran (CCI)\formation développeur IA\Moran\Bio-X\automatisation TB Odoo\recommendations_commerciales"

def check_field_discrimination(summary_df):
    """Vérifie que les champs sont bien discriminés par le modèle"""
    field_means = summary_df.groupby("field")["mean"].mean()
    field_ranges = field_means.max() - field_means.min()
    
    if field_ranges < 0.3:
        print(f"WARNING: Faible discrimination entre champs (range: {field_ranges:.3f})")
    elif field_ranges > 0.8:
        print(f"INFO: Forte discrimination entre champs (range: {field_ranges:.3f})")
    else:
        print(f"OK: Discrimination acceptable (range: {field_ranges:.3f})")


def check_extreme_stability(summary_df):
    """Vérifie la présence de valeurs aberrantes dans les extrêmes"""
    for idx, row in summary_df.iterrows():
        min_to_p10_ratio = row["p10"] / row["min"] if row["min"] > 0 else float('inf')
        max_to_p90_ratio = row["max"] / row["p90"] if row["p90"] > 0 else float('inf')
        
        if min_to_p10_ratio > 5:
            print(f"WARNING - {row['model']}/{row['field']}: Anomalie dans les valeurs minimales")
            print(f"  min ({row['min']:.3f}) très éloigné du p10 ({row['p10']:.3f})")
        
        if max_to_p90_ratio > 1.2:
            print(f"WARNING - {row['model']}/{row['field']}: Pic anormal dans les valeurs maximales")
            print(f"  max ({row['max']:.3f}) très éloigné du p90 ({row['p90']:.3f})")


def check_performance_efficiency(summary_df, latency_df):
    """Analyse le compromis qualité/performance"""
    merged = summary_df.merge(latency_df, on=["model", "field"], how="left")
    
    for idx, row in merged.iterrows():
        if pd.notna(row.get("embed_seconds_total")):
            quality_per_second = row["mean"] / row["embed_seconds_total"] if row["embed_seconds_total"] > 0 else 0
            
            if quality_per_second < 0.001:
                print(f"WARNING - {row['model']}/{row['field']}: Faible efficacité")
                print(f"  Qualité par seconde: {quality_per_second:.6f}")
            
            if row.get("pdf_read_failures", 0) > 0:
                print(f"INFO - {row['model']}/{row['field']}: {row['pdf_read_failures']} échecs de lecture PDF")

def main_build_index(
    master_df: pd.DataFrame,
    model_key: Optional[str] = None,
    out_index: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    prefix: Optional[str] = None,
) -> Path | list[Path]:
    """
    Si model_key est None: génère pour tous les modèles (comportement historique)
    Si model_key est défini: génère uniquement pour ce modèle

    Sorties (par modèle):
    - index : {model}-product_similarity_index.csv (ou out_index si fourni)
    - {model}-vector_diagnostics_summary.csv
    - {model}-vector_diagnostics_top_nearest.csv
    - {model}-vector_diagnostics_top_distant.csv
    - {model}-latency_metrics.csv

    out_dir: si fourni, écrit tous les fichiers dans ce dossier.
    prefix: si fourni, remplace le début des noms de fichiers (sinon model_key).
            Exemple: prefix="biox_index" => biox_index-vector_diagnostics_summary.csv, etc.
            (L'index peut aussi être forcé via out_index.)
    """

    required = {
        COL_REF, COL_CODE_RAW, COL_CODE_BASE, COL_IFU, COL_VALID,
        COL_PATHO, COL_MPC_RANGE, COL_SPECIES, COL_CLASS
    }
    missing = required - set(master_df.columns)
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans master_df : {sorted(missing)}")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    master = master_df.copy()

    # IMPORTANT : on ne supprime PAS les espaces internes des valeurs (matching Odoo)
    for c in [COL_REF, COL_CODE_RAW, COL_CODE_BASE]:
        master[c] = master[c].astype(str)

    # Déterminer la liste des modèles à exécuter
    if model_key is None:
        model_keys = list(MODELS.keys())
    else:
        if model_key not in MODELS:
            raise ValueError(f"model_key invalide: {model_key}. Choix: {list(MODELS.keys())}")
        model_keys = [model_key]

    def save_model_metadata(model_key, output_dir):
        meta = {
        "model": model_key,
        "timestamp": datetime.utcnow().isoformat(),
        "weights": WEIGHTS,
        "embedding_models": MODELS[model_key][0]
        }

        path = Path(output_dir) / f"{model_key}-model_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return path

    created_indexes: list[Path] = []

    for mk in model_keys:
        logging.info(f"=== BUILD INDEX MODEL: {mk} ({MODELS[mk][0]}) ===")
        t0 = time.perf_counter()

        # Base de nommage pour les fichiers de diagnostics
        base = prefix if prefix else mk

        def _p(name: str) -> Path:
            # Construit un chemin dans out_dir si fourni, sinon cwd
            return (out_dir / name) if out_dir else Path(name)

        # embeddings
        emb_ifu, m_ifu = embed_pdfs_from_links(master[COL_IFU].fillna("").tolist(), mk)
        emb_val, m_val = embed_pdfs_from_links(master[COL_VALID].fillna("").tolist(), mk)
        emb_pat, m_pat = embed_text_column(master[COL_PATHO].fillna("").tolist(), mk)

        # diagnostics
        refs = master[COL_REF].astype(str).tolist()

        summary = []
        latency = []

        for field, X, mm in [
            ("instructions_for_use", emb_ifu, m_ifu),
            ("Validation_file", emb_val, m_val),
            ("Pathogen/Target", emb_pat, m_pat),
        ]:
            row = {"model": mk, "field": field}
            row.update(diagnostics_distribution(X))
            summary.append(row)

            lat = {"model": mk, "field": field}
            lat.update(mm)
            latency.append(lat)

        _p(f"{base}-vector_diagnostics_summary.csv").write_text(
            pd.DataFrame(summary).to_csv(index=False), encoding="utf-8"
        )
        _p(f"{base}-latency_metrics.csv").write_text(
            pd.DataFrame(latency).to_csv(index=False), encoding="utf-8"
        )

        near = []
        far = []
        for field, X in [
            ("instructions_for_use", emb_ifu),
            ("Validation_file", emb_val),
            ("Pathogen/Target", emb_pat),
        ]:
            near.append(diagnostics_top_pairs(X, refs, field, 20, True))
            far.append(diagnostics_top_pairs(X, refs, field, 20, False))

        pd.concat(near, ignore_index=True).to_csv(_p(f"{base}-vector_diagnostics_top_nearest.csv"), index=False)
        pd.concat(far, ignore_index=True).to_csv(_p(f"{base}-vector_diagnostics_top_distant.csv"), index=False)

        # index
        feat_df = build_feature_matrix(master, emb_ifu, emb_val, emb_pat, WEIGHTS)
        index_df = build_similarity_index(feat_df)

        # Chemin de sortie de l'index
        if out_index is not None:
            # out_index ne s'applique que si on ne fait qu'un modèle
            if len(model_keys) != 1:
                raise ValueError("out_index est uniquement supporté quand model_key est fourni (un seul modèle).")
            out_index_path = Path(out_index)
            out_index_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_index_path = _p(f"{mk}-product_similarity_index.csv") if prefix is None else _p(f"{base}-product_similarity_index.csv")

        index_df.to_csv(out_index_path, index=False)
        save_model_metadata(mk, out_dir or Path.cwd())
        created_indexes.append(out_index_path)

        logging.info(f"MODEL {mk} OK — {time.perf_counter() - t0:.1f}s")

    summary_df = pd.DataFrame(summary)
    latency_df = pd.DataFrame(latency)

    check_field_discrimination(summary_df)
    check_performance_efficiency(summary_df, latency_df)
    check_extreme_stability(summary_df)

    # Retour: un Path si un seul modèle, sinon liste
    return created_indexes[0] if len(created_indexes) == 1 else created_indexes