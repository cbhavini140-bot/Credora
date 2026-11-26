"""Dataset-specific feature engineering matching the original Colab training pipelines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ULB_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

SPARKOV_RAW_REQUIRED = [
    "trans_date_trans_time",
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "category",
    "gender",
    "state",
]

SPARKOV_MODEL_FEATURES = [
    "amt",
    "city_pop",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_weekday",
    "merchant_distance_km",
    "category",
    "gender",
    "state",
]

PAYSIM_RAW_REQUIRED = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

PAYSIM_MODEL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
    "origin_error",
    "destination_error",
    "hour",
    "type",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometers between two lat/lon coordinates."""
    r = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def haversine_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized Haversine distance calculation in kilometers."""
    r = 6371.0
    lat1 = np.radians(pd.to_numeric(df["lat"], errors="coerce").astype(float))
    lon1 = np.radians(pd.to_numeric(df["long"], errors="coerce").astype(float))
    lat2 = np.radians(pd.to_numeric(df["merch_lat"], errors="coerce").astype(float))
    lon2 = np.radians(pd.to_numeric(df["merch_long"], errors="coerce").astype(float))
    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0) ** 2
    return pd.Series(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a))), index=df.index)


def prepare_ulb_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and order columns for ULB Credit Card Fraud model."""
    missing = [c for c in ULB_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"ULB schema mismatch. Missing required columns: {', '.join(missing[:10])}"
            + (f" (+{len(missing)-10} more)" if len(missing) > 10 else "")
        )
    out = df[ULB_FEATURES].copy()
    for col in ULB_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def prepare_sparkov_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Sparkov time and Haversine distance features or validate pre-engineered schema."""
    out = df.copy()
    has_engineered = all(c in out.columns for c in SPARKOV_MODEL_FEATURES)

    if not has_engineered:
        has_time = "trans_date_trans_time" in out.columns or all(
            c in out.columns
            for c in ["transaction_hour", "transaction_day", "transaction_month", "transaction_weekday"]
        )
        has_geo = "merchant_distance_km" in out.columns or all(
            c in out.columns for c in ["lat", "long", "merch_lat", "merch_long"]
        )
        missing_base = [c for c in ["amt", "city_pop", "category", "gender", "state"] if c not in out.columns]
        if not has_time:
            missing_base.append("trans_date_trans_time")
        if not has_geo:
            missing_base.extend([c for c in ["lat", "long", "merch_lat", "merch_long"] if c not in out.columns])
        if missing_base:
            raise ValueError(
                f"Sparkov schema mismatch. Missing required columns: {', '.join(missing_base)}"
            )

        if "trans_date_trans_time" in out.columns and not all(
            c in out.columns
            for c in ["transaction_hour", "transaction_day", "transaction_month", "transaction_weekday"]
        ):
            dt = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
            if dt.isna().any():
                raise ValueError("Invalid datetime values in 'trans_date_trans_time'.")
            out["transaction_hour"] = dt.dt.hour
            out["transaction_day"] = dt.dt.day
            out["transaction_month"] = dt.dt.month
            out["transaction_weekday"] = dt.dt.day_of_week

        if "merchant_distance_km" not in out.columns:
            out["merchant_distance_km"] = haversine_series(out)

    result = out[SPARKOV_MODEL_FEATURES].copy()
    numeric_cols = [
        "amt",
        "city_pop",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_weekday",
        "merchant_distance_km",
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    for col in ["category", "gender", "state"]:
        result[col] = result[col].astype(str)
    return result


def prepare_paysim_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer PaySim balance deltas, error terms, and hour feature matching Colab pipeline."""
    missing = [c for c in PAYSIM_RAW_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"PaySim schema mismatch. Missing required columns: {', '.join(missing)}"
        )

    out = df.copy()
    for col in ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "origin_balance_change" not in out.columns:
        out["origin_balance_change"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    if "destination_balance_change" not in out.columns:
        out["destination_balance_change"] = out["newbalanceDest"] - out["oldbalanceDest"]
    if "origin_error" not in out.columns:
        out["origin_error"] = out["origin_balance_change"] - out["amount"]
    if "destination_error" not in out.columns:
        out["destination_error"] = out["destination_balance_change"] - out["amount"]
    if "hour" not in out.columns:
        out["hour"] = out["step"] % 24

    result = out[PAYSIM_MODEL_FEATURES].copy()
    result["type"] = result["type"].astype(str)
    return result


def prepare_features_for_model(model_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Route DataFrame preparation strictly to the selected dataset schema."""
    key = model_key.lower().strip()
    if key == "ulb":
        return prepare_ulb_dataframe(df)
    if key == "sparkov":
        return prepare_sparkov_dataframe(df)
    if key == "paysim":
        return prepare_paysim_dataframe(df)
    raise ValueError(f"Unsupported dataset/model key '{model_key}'. Expected one of: ulb, sparkov, paysim.")


def extract_metadata_from_row(model_key: str, row: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized database metadata fields from a raw transaction dict without inventing PII."""
    key = model_key.lower().strip()
    if key == "ulb":
        return {
            "amount": float(row.get("Amount", 0.0)),
            "transaction_type": "CARD_PCA_ANONYMIZED",
            "merchant": None,
            "category": "pca_anonymized",
            "state": None,
            "city": None,
            "gender": None,
            "city_pop": None,
            "customer_ref": None,  # ULB has no customer identity fields
            "counterparty_ref": None,
            "ground_truth_label": bool(int(row["Class"])) if "Class" in row and pd.notna(row["Class"]) else None,
        }
    if key == "sparkov":
        cc_raw = str(row.get("cc_num", "")).strip()
        masked_cc = f"****{cc_raw[-4:]}" if len(cc_raw) >= 4 else None
        first = str(row.get("first", "")).strip()
        last = str(row.get("last", "")).strip()
        name_part = f"{first} {last}".strip()
        customer_ref = (
            f"{name_part} ({masked_cc})".strip()
            if name_part and masked_cc
            else (name_part or masked_cc)
        )
        return {
            "amount": float(row.get("amt", 0.0)),
            "transaction_type": str(row.get("category", "card_purchase")),
            "merchant": str(row["merchant"]) if row.get("merchant") not in (None, "", "nan") else None,
            "category": str(row.get("category", "")),
            "state": str(row.get("state", "")) if row.get("state") not in (None, "", "nan") else None,
            "city": str(row.get("city", "")) if row.get("city") not in (None, "", "nan") else None,
            "gender": str(row.get("gender", "")) if row.get("gender") not in (None, "", "nan") else None,
            "city_pop": int(float(row["city_pop"])) if row.get("city_pop") not in (None, "", "nan") else None,
            "customer_ref": customer_ref,
            "counterparty_ref": str(row["merchant"]) if row.get("merchant") not in (None, "", "nan") else None,
            "ground_truth_label": bool(int(row["is_fraud"])) if "is_fraud" in row and pd.notna(row["is_fraud"]) else None,
        }
    if key == "paysim":
        return {
            "amount": float(row.get("amount", 0.0)),
            "transaction_type": str(row.get("type", "TRANSFER")),
            "merchant": None,
            "category": str(row.get("type", "TRANSFER")),
            "state": None,
            "city": None,
            "gender": None,
            "city_pop": None,
            "customer_ref": str(row["nameOrig"]) if row.get("nameOrig") not in (None, "", "nan") else None,
            "counterparty_ref": str(row["nameDest"]) if row.get("nameDest") not in (None, "", "nan") else None,
            "ground_truth_label": bool(int(row["isFraud"])) if "isFraud" in row and pd.notna(row["isFraud"]) else None,
        }
    raise ValueError(f"Unsupported model_key: {model_key}")
