"""Single-transaction prediction, CSV batch upload, schema inspection, and CSV export."""

import io
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import get_current_user
from app.database.session import get_db
from app.ml.feature_engineering import (
    PAYSIM_RAW_REQUIRED,
    SPARKOV_RAW_REQUIRED,
    ULB_FEATURES,
)
from app.ml.model_registry import get_model_registry
from app.models.entities import Transaction, User
from app.schemas.schemas import SinglePredictionRequest
from app.services.transaction_service import run_and_persist_transactions

router = APIRouter(prefix="/api/predictions", tags=["Fraud Detection"])


@router.get("/sample-schema/{model_key}")
def get_sample_schema(
    model_key: str,
    current_user: User = Depends(get_current_user),
):
    """Return expected schema and a genuine example payload from curated samples."""
    key = model_key.lower().strip()
    if key not in ("ulb", "sparkov", "paysim"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model_key.")

    settings = get_settings()
    registry = get_model_registry()
    bundle = registry.get_bundle(key)

    sample_map = {
        "ulb": settings.datasets_path / "samples" / "ulb_sample.csv",
        "sparkov": settings.datasets_path / "samples" / "sparkov_sample.csv",
        "paysim": settings.datasets_path / "samples" / "paysim_sample.csv",
    }
    required_map = {
        "ulb": ULB_FEATURES,
        "sparkov": SPARKOV_RAW_REQUIRED,
        "paysim": PAYSIM_RAW_REQUIRED,
    }

    example_legit: dict[str, Any] = {}
    example_fraud: dict[str, Any] = {}
    sample_file = sample_map[key]
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        label_col = {"ulb": "Class", "sparkov": "is_fraud", "paysim": "isFraud"}[key]
        if label_col in df.columns:
            legit_rows = df[df[label_col] == 0]
            fraud_rows = df[df[label_col] == 1]
            if not legit_rows.empty:
                example_legit = {
                    k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                    for k, v in legit_rows.iloc[0].drop(labels=[label_col], errors="ignore").to_dict().items()
                }
            if not fraud_rows.empty:
                example_fraud = {
                    k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                    for k, v in fraud_rows.iloc[0].drop(labels=[label_col], errors="ignore").to_dict().items()
                }

    return {
        "model_key": key,
        "model_name": bundle.model_name,
        "dataset_name": bundle.dataset_name,
        "threshold": bundle.threshold,
        "required_raw_columns": required_map[key],
        "model_features": bundle.expected_features,
        "explainability_note": bundle.explainability_note,
        "example_legit": example_legit,
        "example_fraud": example_fraud,
    }


@router.post("/single")
def predict_single_transaction(
    payload: SinglePredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run real ML prediction, risk scoring, optional SHAP explanation, and DB persistence."""
    try:
        outcome = run_and_persist_transactions(
            db=db,
            model_key=payload.model_key,
            raw_rows=[payload.features],
            actor=current_user,
            save_to_db=payload.save_transaction,
            include_shap=payload.include_shap,
            custom_ref=payload.transaction_ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return {
        "saved": payload.save_transaction,
        "prediction": outcome["previews"][0],
        "alerts_generated": len(outcome["alerts"]),
    }


@router.post("/batch")
async def predict_batch_csv(
    model_key: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Validate and process a CSV batch using the dataset-specific ML model and persist results."""
    key = model_key.lower().strip()
    if key not in ("ulb", "sparkov", "paysim"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported model_key. Choose one of: ulb, sparkov, paysim.",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only .csv files are accepted for batch transaction analysis.",
        )

    settings = get_settings()
    max_bytes = settings.MAX_CSV_UPLOAD_MB * 1024 * 1024
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded CSV file is empty.",
        )
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV file exceeds maximum allowed size of {settings.MAX_CSV_UPLOAD_MB} MB.",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to parse CSV file: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded CSV contains headers but zero transaction rows.",
        )

    if len(df) > 5000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch upload limit is 5,000 rows per file to ensure responsive scoring.",
        )

    batch_id = f"BATCH-{key.upper()}-{uuid.uuid4().hex[:8].upper()}"
    raw_rows = df.to_dict(orient="records")

    try:
        outcome = run_and_persist_transactions(
            db=db,
            model_key=key,
            raw_rows=raw_rows,
            actor=current_user,
            save_to_db=True,
            include_shap=False,
            batch_id=batch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    txs = outcome["transactions"]
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for tx in txs:
        risk_counts[tx.risk_level] = risk_counts.get(tx.risk_level, 0) + 1

    return {
        "batch_id": batch_id,
        "filename": file.filename,
        "model_key": key,
        "model_name": outcome["bundle"].model_name,
        "threshold": outcome["bundle"].threshold,
        "total_processed": len(txs),
        "predicted_fraud_count": sum(1 for t in txs if t.predicted_fraud),
        "alerts_generated": len(outcome["alerts"]),
        "risk_distribution": risk_counts,
        "download_url": f"/api/predictions/batch/{batch_id}/export",
        "results": outcome["previews"],
    }


@router.get("/batch/{batch_id}/export")
def export_batch_csv(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export actual processed predictions for a batch from PostgreSQL as CSV."""
    rows = (
        db.query(Transaction)
        .filter(Transaction.batch_id == batch_id)
        .order_by(Transaction.id.asc())
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transactions found for batch '{batch_id}'.",
        )

    export_records = [
        {
            "transaction_ref": tx.transaction_ref,
            "dataset_source": tx.dataset_source,
            "batch_id": tx.batch_id,
            "amount": float(tx.amount),
            "transaction_type": tx.transaction_type,
            "merchant": tx.merchant,
            "category": tx.category,
            "state": tx.state,
            "customer_ref": tx.customer_ref,
            "fraud_probability": tx.fraud_probability,
            "risk_score": tx.risk_score,
            "risk_level": tx.risk_level,
            "predicted_fraud": int(tx.predicted_fraud),
            "analyst_decision": tx.analyst_decision,
            "ground_truth_label": int(tx.ground_truth_label) if tx.ground_truth_label is not None else "",
            "transaction_time": tx.transaction_time.isoformat(),
        }
        for tx in rows
    ]

    out_df = pd.DataFrame(export_records)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    return StreamingResponse(
        iter([csv_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{batch_id}_results.csv"'},
    )
