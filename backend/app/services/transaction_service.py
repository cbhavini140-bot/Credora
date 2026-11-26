"""Transaction inference, persistence, risk scoring, and automated alert creation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.feature_engineering import extract_metadata_from_row
from app.ml.model_registry import CredoraModelRegistry, get_model_registry
from app.models.entities import (
    Alert,
    ModelVersion,
    SystemSetting,
    Transaction,
    User,
)
from app.services.audit_service import record_audit


def get_risk_thresholds(db: Session) -> tuple[float, float]:
    """Read persisted risk category thresholds from system_settings, or fallback to config."""
    settings = get_settings()
    row = db.query(SystemSetting).filter(SystemSetting.key == "risk_thresholds").first()
    if row and isinstance(row.value_json, dict):
        med = float(row.value_json.get("medium_threshold", settings.DEFAULT_RISK_MEDIUM_THRESHOLD))
        high = float(row.value_json.get("high_threshold", settings.DEFAULT_RISK_HIGH_THRESHOLD))
        return med, high
    return settings.DEFAULT_RISK_MEDIUM_THRESHOLD, settings.DEFAULT_RISK_HIGH_THRESHOLD


def determine_alert_priority(risk_score: float, predicted_fraud: bool) -> Optional[str]:
    """Assign alert priority strictly based on risk score and model prediction."""
    if predicted_fraud and risk_score >= 85.0:
        return "CRITICAL"
    if predicted_fraud or risk_score >= 70.0:
        return "HIGH"
    if risk_score >= 30.0:
        return "MEDIUM"
    return None


def _sanitize_json_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert pandas/numpy scalars or NaN values into JSON-safe Python primitives."""
    clean: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            clean[str(k)] = None
        elif pd.isna(v):
            clean[str(k)] = None
        elif hasattr(v, "item"):
            clean[str(k)] = v.item()
        else:
            clean[str(k)] = v
    return clean


def _parse_transaction_time(model_key: str, row: dict[str, Any]) -> datetime:
    if model_key == "sparkov" and row.get("trans_date_trans_time"):
        parsed = pd.to_datetime(row["trans_date_trans_time"], errors="coerce")
        if pd.notna(parsed):
            dt = parsed.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    return datetime.now(timezone.utc)


def run_and_persist_transactions(
    db: Session,
    model_key: str,
    raw_rows: list[dict[str, Any]],
    actor: Optional[User] = None,
    save_to_db: bool = True,
    include_shap: bool = False,
    batch_id: Optional[str] = None,
    is_simulated: bool = False,
    custom_ref: Optional[str] = None,
) -> dict[str, Any]:
    """Run ML inference on 1..N transactions and persist transactions + alerts if requested."""
    registry: CredoraModelRegistry = get_model_registry()
    bundle = registry.get_bundle(model_key)
    medium_th, high_th = get_risk_thresholds(db)

    raw_df = pd.DataFrame(raw_rows)
    predictions, prepared_df, _ = registry.predict_dataframe(
        model_key=bundle.model_key,
        raw_df=raw_df,
        medium_threshold=medium_th,
        high_threshold=high_th,
        include_shap=include_shap,
    )

    model_version = (
        db.query(ModelVersion).filter(ModelVersion.model_key == bundle.model_key).first()
    )

    saved_transactions: list[Transaction] = []
    created_alerts: list[Alert] = []
    preview_results: list[dict[str, Any]] = []

    for idx, (raw_row, pred) in enumerate(zip(raw_rows, predictions)):
        meta = extract_metadata_from_row(bundle.model_key, raw_row)
        merged_features = {**raw_row, **prepared_df.iloc[idx].to_dict()}
        # Mask raw credit card number if present in Sparkov input
        if "cc_num" in merged_features and merged_features["cc_num"] is not None:
            cc_str = str(merged_features["cc_num"])
            merged_features["cc_num"] = f"****{cc_str[-4:]}" if len(cc_str) >= 4 else "****"

        tx_ref = (
            custom_ref
            if (custom_ref and len(raw_rows) == 1)
            else f"TXN-{bundle.model_key.upper()}-{uuid.uuid4().hex[:10].upper()}"
        )
        tx_time = _parse_transaction_time(bundle.model_key, raw_row)

        preview_item = {
            "transaction_ref": tx_ref,
            "dataset_source": bundle.model_key,
            "model_name": bundle.model_name,
            "is_simulated": is_simulated,
            "amount": meta["amount"],
            "transaction_time": tx_time,
            "transaction_type": meta["transaction_type"],
            "merchant": meta["merchant"],
            "category": meta["category"],
            "state": meta["state"],
            "customer_ref": meta["customer_ref"],
            "counterparty_ref": meta["counterparty_ref"],
            "ground_truth_label": meta["ground_truth_label"],
            **pred,
        }

        if save_to_db:
            tx = Transaction(
                transaction_ref=tx_ref,
                dataset_source=bundle.model_key,
                batch_id=batch_id,
                is_simulated=is_simulated,
                amount=meta["amount"],
                transaction_time=tx_time,
                transaction_type=meta["transaction_type"],
                merchant=meta["merchant"],
                category=meta["category"],
                state=meta["state"],
                city=meta["city"],
                gender=meta["gender"],
                city_pop=meta["city_pop"],
                customer_ref=meta["customer_ref"],
                counterparty_ref=meta["counterparty_ref"],
                raw_features_json=_sanitize_json_dict(merged_features),
                model_version_id=model_version.id if model_version else None,
                fraud_probability=pred["fraud_probability"],
                risk_score=pred["risk_score"],
                risk_level=pred["risk_level"],
                predicted_fraud=pred["predicted_fraud"],
                ground_truth_label=meta["ground_truth_label"],
                analyst_decision="PENDING",
                created_by_id=actor.id if actor else None,
            )
            db.add(tx)
            db.flush()
            saved_transactions.append(tx)
            preview_item["id"] = tx.id

            priority = determine_alert_priority(pred["risk_score"], pred["predicted_fraud"])
            if priority is not None:
                reason = (
                    f"{bundle.model_name} flagged transaction ({bundle.dataset_name}): "
                    f"probability={pred['fraud_probability']:.4f} (threshold={bundle.threshold:.2f}), "
                    f"risk_score={pred['risk_score']:.1f} [{pred['risk_level']}]."
                )
                if is_simulated:
                    reason = f"[SIMULATED STREAM] {reason}"
                alert = Alert(
                    alert_code=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                    transaction_id=tx.id,
                    priority=priority,
                    status="OPEN",
                    reason=reason,
                    risk_score=pred["risk_score"],
                )
                db.add(alert)
                db.flush()
                created_alerts.append(alert)
                preview_item["alert_code"] = alert.alert_code
                preview_item["alert_priority"] = priority

        preview_results.append(preview_item)

    if save_to_db:
        if batch_id:
            record_audit(
                db=db,
                actor=actor,
                action="BATCH_UPLOAD",
                entity_type="batch",
                entity_id=batch_id,
                details={
                    "model_key": bundle.model_key,
                    "processed_count": len(saved_transactions),
                    "predicted_fraud_count": sum(1 for t in saved_transactions if t.predicted_fraud),
                    "alerts_generated": len(created_alerts),
                },
            )
        elif not is_simulated and len(saved_transactions) == 1:
            tx = saved_transactions[0]
            record_audit(
                db=db,
                actor=actor,
                action="PREDICT_SINGLE",
                entity_type="transaction",
                entity_id=tx.transaction_ref,
                details={
                    "model_key": bundle.model_key,
                    "fraud_probability": tx.fraud_probability,
                    "risk_score": tx.risk_score,
                    "predicted_fraud": tx.predicted_fraud,
                    "alerts_generated": len(created_alerts),
                },
            )
        db.commit()
        for tx in saved_transactions:
            db.refresh(tx)
        for alert in created_alerts:
            db.refresh(alert)

    return {
        "transactions": saved_transactions,
        "alerts": created_alerts,
        "previews": preview_results,
        "bundle": bundle,
    }
