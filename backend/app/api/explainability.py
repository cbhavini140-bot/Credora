"""Explainable AI (SHAP) endpoints for global feature importance and per-transaction explanations."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.session import get_db
from app.ml.model_registry import get_model_registry
from app.models.entities import Transaction, User

router = APIRouter(prefix="/api/explainability", tags=["Explainable AI"])


class ExplainPayload(BaseModel):
    model_key: str
    features: dict[str, Any]


@router.get("/global/{model_key}")
def get_global_explainability(
    model_key: str,
    current_user: User = Depends(get_current_user),
):
    """Return genuine global feature importances from the selected trained model."""
    registry = get_model_registry()
    try:
        return registry.global_feature_importance(model_key)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/transaction/{transaction_id}")
def explain_stored_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compute real SHAP contributions for an existing database transaction."""
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")

    registry = get_model_registry()
    try:
        explanation = registry.explain_single_row(tx.dataset_source, tx.raw_features_json)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return {
        "transaction_id": tx.id,
        "transaction_ref": tx.transaction_ref,
        "dataset_source": tx.dataset_source,
        "amount": float(tx.amount),
        "stored_probability": tx.fraud_probability,
        "stored_risk_score": tx.risk_score,
        "stored_risk_level": tx.risk_level,
        **explanation,
    }


@router.post("/predict-explain")
def explain_adhoc_transaction(
    payload: ExplainPayload,
    current_user: User = Depends(get_current_user),
):
    """Compute SHAP explanation without saving the transaction."""
    registry = get_model_registry()
    try:
        return registry.explain_single_row(payload.model_key, payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
