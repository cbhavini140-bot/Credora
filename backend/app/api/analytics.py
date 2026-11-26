"""Analytics and model performance endpoints derived from PostgreSQL records."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.session import get_db
from app.ml.model_registry import get_model_registry
from app.models.entities import Alert, Investigation, ModelVersion, Transaction, User

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/overview")
def get_overview_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compute real-time KPIs, trends, distributions, and recent items from PostgreSQL."""
    total_tx = db.query(func.count(Transaction.id)).scalar() or 0
    predicted_fraud = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.predicted_fraud.is_(True))
        .scalar()
        or 0
    )
    confirmed_fraud = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.analyst_decision == "CONFIRMED_FRAUD")
        .scalar()
        or 0
    )
    legitimate_tx = (
        db.query(func.count(Transaction.id))
        .filter(
            (Transaction.analyst_decision == "CONFIRMED_LEGIT")
            | (
                (Transaction.analyst_decision == "PENDING")
                & (Transaction.predicted_fraud.is_(False))
            )
        )
        .scalar()
        or 0
    )
    high_risk_tx = (
        db.query(func.count(Transaction.id))
        .filter(Transaction.risk_level == "HIGH")
        .scalar()
        or 0
    )
    total_value = db.query(func.coalesce(func.sum(Transaction.amount), 0)).scalar() or 0
    flagged_value = (
        db.query(func.coalesce(func.sum(Transaction.amount), 0))
        .filter((Transaction.predicted_fraud.is_(True)) | (Transaction.risk_level == "HIGH"))
        .scalar()
        or 0
    )
    fraud_rate = round((predicted_fraud / total_tx) * 100.0, 2) if total_tx > 0 else 0.0

    # Risk distribution
    risk_rows = (
        db.query(Transaction.risk_level, func.count(Transaction.id))
        .group_by(Transaction.risk_level)
        .all()
    )
    risk_map = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for level, cnt in risk_rows:
        risk_map[level] = int(cnt)

    # Alert priority distribution
    alert_priority_rows = (
        db.query(Alert.priority, func.count(Alert.id))
        .group_by(Alert.priority)
        .all()
    )
    alert_priority_map = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for prio, cnt in alert_priority_rows:
        alert_priority_map[prio] = int(cnt)

    # Fraud by category (top 8)
    cat_rows = (
        db.query(
            func.coalesce(Transaction.category, Transaction.transaction_type).label("cat"),
            func.count(Transaction.id).label("total"),
            func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("fraud_count"),
        )
        .group_by("cat")
        .order_by(desc("fraud_count"), desc("total"))
        .limit(8)
        .all()
    )

    # Daily volume & fraud trend
    day_expr = func.date(Transaction.transaction_time)
    trend_rows = (
        db.query(
            day_expr.label("day"),
            func.count(Transaction.id).label("total"),
            func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("predicted_fraud"),
            func.sum(case((Transaction.analyst_decision == "CONFIRMED_FRAUD", 1), else_=0)).label(
                "confirmed_fraud"
            ),
        )
        .group_by(day_expr)
        .order_by(day_expr.asc())
        .limit(30)
        .all()
    )

    recent_transactions = (
        db.query(Transaction)
        .order_by(desc(Transaction.created_at), desc(Transaction.id))
        .limit(6)
        .all()
    )
    recent_alerts = (
        db.query(Alert)
        .order_by(desc(Alert.created_at), desc(Alert.id))
        .limit(6)
        .all()
    )
    pending_investigations = (
        db.query(Investigation)
        .filter(Investigation.status.in_(["OPEN", "IN_PROGRESS", "ESCALATED"]))
        .order_by(desc(Investigation.updated_at))
        .limit(5)
        .all()
    )

    open_alerts_count = (
        db.query(func.count(Alert.id))
        .filter(Alert.status.in_(["OPEN", "UNDER_REVIEW", "ESCALATED"]))
        .scalar()
        or 0
    )
    open_investigations_count = (
        db.query(func.count(Investigation.id))
        .filter(Investigation.status.in_(["OPEN", "IN_PROGRESS", "ESCALATED"]))
        .scalar()
        or 0
    )

    registry = get_model_registry()

    return {
        "kpis": {
            "total_transactions": int(total_tx),
            "predicted_fraud": int(predicted_fraud),
            "confirmed_fraud": int(confirmed_fraud),
            "legitimate_transactions": int(legitimate_tx),
            "high_risk_transactions": int(high_risk_tx),
            "fraud_rate_percent": fraud_rate,
            "total_transaction_value": round(float(total_value), 2),
            "flagged_transaction_value": round(float(flagged_value), 2),
            "open_alerts": int(open_alerts_count),
            "pending_investigations": int(open_investigations_count),
        },
        "risk_distribution": [
            {"level": k, "count": v} for k, v in risk_map.items()
        ],
        "alert_priority_distribution": [
            {"priority": k, "count": v} for k, v in alert_priority_map.items()
        ],
        "fraud_by_category": [
            {
                "category": r.cat or "uncategorized",
                "total": int(r.total),
                "fraud_count": int(r.fraud_count or 0),
            }
            for r in cat_rows
        ],
        "volume_and_fraud_trend": [
            {
                "date": str(r.day),
                "total": int(r.total),
                "predicted_fraud": int(r.predicted_fraud or 0),
                "confirmed_fraud": int(r.confirmed_fraud or 0),
            }
            for r in trend_rows
        ],
        "recent_transactions": [
            {
                "id": t.id,
                "transaction_ref": t.transaction_ref,
                "dataset_source": t.dataset_source,
                "amount": float(t.amount),
                "risk_score": t.risk_score,
                "risk_level": t.risk_level,
                "predicted_fraud": t.predicted_fraud,
                "analyst_decision": t.analyst_decision,
                "is_simulated": t.is_simulated,
                "created_at": t.created_at,
            }
            for t in recent_transactions
        ],
        "recent_alerts": [
            {
                "id": a.id,
                "alert_code": a.alert_code,
                "priority": a.priority,
                "status": a.status,
                "risk_score": a.risk_score,
                "transaction_id": a.transaction_id,
                "created_at": a.created_at,
            }
            for a in recent_alerts
        ],
        "pending_investigations": [
            {
                "id": inv.id,
                "case_code": inv.case_code,
                "title": inv.title,
                "priority": inv.priority,
                "status": inv.status,
                "updated_at": inv.updated_at,
            }
            for inv in pending_investigations
        ],
        "model_status": registry.status_summary(),
    }


@router.get("/risk")
def get_risk_analytics(
    dataset_source: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deep risk analytics: merchants, categories, states (Sparkov), dataset breakdown, investigations."""
    base_q = db.query(Transaction)
    if dataset_source:
        base_q = base_q.filter(Transaction.dataset_source == dataset_source.lower())

    def apply_ds(q):
        return q.filter(Transaction.dataset_source == dataset_source.lower()) if dataset_source else q

    dataset_rows = (
        apply_ds(
            db.query(
                Transaction.dataset_source,
                func.count(Transaction.id).label("total"),
                func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("fraud_count"),
                func.avg(Transaction.risk_score).label("avg_risk_score"),
                func.sum(Transaction.amount).label("total_value"),
            )
        )
        .group_by(Transaction.dataset_source)
        .all()
    )

    merchant_rows = (
        apply_ds(
            db.query(
                Transaction.merchant,
                func.count(Transaction.id).label("total"),
                func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("fraud_count"),
                func.avg(Transaction.risk_score).label("avg_risk"),
                func.sum(Transaction.amount).label("total_amount"),
            ).filter(Transaction.merchant.isnot(None))
        )
        .group_by(Transaction.merchant)
        .order_by(desc("fraud_count"), desc("avg_risk"))
        .limit(12)
        .all()
    )

    category_rows = (
        apply_ds(
            db.query(
                func.coalesce(Transaction.category, Transaction.transaction_type).label("cat"),
                func.count(Transaction.id).label("total"),
                func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("fraud_count"),
                func.avg(Transaction.risk_score).label("avg_risk"),
                func.sum(Transaction.amount).label("total_amount"),
            )
        )
        .group_by("cat")
        .order_by(desc("fraud_count"), desc("total"))
        .limit(15)
        .all()
    )

    state_rows = (
        apply_ds(
            db.query(
                Transaction.state,
                func.count(Transaction.id).label("total"),
                func.sum(case((Transaction.predicted_fraud.is_(True), 1), else_=0)).label("fraud_count"),
                func.avg(Transaction.risk_score).label("avg_risk"),
                func.sum(Transaction.amount).label("total_amount"),
            ).filter(Transaction.state.isnot(None))
        )
        .group_by(Transaction.state)
        .order_by(desc("fraud_count"), desc("total"))
        .limit(20)
        .all()
    )

    inv_stats = (
        db.query(Investigation.status, func.count(Investigation.id))
        .group_by(Investigation.status)
        .all()
    )

    # Risk score histogram buckets
    bucket_defs = [
        ("0-19", 0, 20),
        ("20-39", 20, 40),
        ("40-59", 40, 60),
        ("60-79", 60, 80),
        ("80-100", 80, 101),
    ]
    all_scores = [r[0] for r in base_q.with_entities(Transaction.risk_score).all()]
    histogram = [
        {
            "bucket": label,
            "count": sum(1 for s in all_scores if lo <= s < hi),
        }
        for label, lo, hi in bucket_defs
    ]

    return {
        "dataset_breakdown": [
            {
                "dataset_source": r.dataset_source,
                "total": int(r.total),
                "fraud_count": int(r.fraud_count or 0),
                "avg_risk_score": round(float(r.avg_risk_score or 0.0), 2),
                "total_value": round(float(r.total_value or 0.0), 2),
            }
            for r in dataset_rows
        ],
        "merchant_analytics": [
            {
                "merchant": r.merchant,
                "total": int(r.total),
                "fraud_count": int(r.fraud_count or 0),
                "avg_risk_score": round(float(r.avg_risk or 0.0), 2),
                "total_amount": round(float(r.total_amount or 0.0), 2),
            }
            for r in merchant_rows
        ],
        "category_analytics": [
            {
                "category": r.cat,
                "total": int(r.total),
                "fraud_count": int(r.fraud_count or 0),
                "avg_risk_score": round(float(r.avg_risk or 0.0), 2),
                "total_amount": round(float(r.total_amount or 0.0), 2),
            }
            for r in category_rows
        ],
        "geographic_analytics": [
            {
                "state": r.state,
                "total": int(r.total),
                "fraud_count": int(r.fraud_count or 0),
                "avg_risk_score": round(float(r.avg_risk or 0.0), 2),
                "total_amount": round(float(r.total_amount or 0.0), 2),
            }
            for r in state_rows
        ],
        "geographic_note": (
            "Geographic state analytics are derived exclusively from Sparkov transactions containing "
            "US state metadata. ULB and PaySim datasets do not provide geographic state attributes."
        ),
        "risk_score_histogram": histogram,
        "investigation_statistics": [
            {"status": s, "count": int(c)} for s, c in inv_stats
        ],
    }


@router.get("/models")
def get_model_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return genuine saved model evaluation metrics and live artifact load status."""
    registry = get_model_registry()
    status_map = registry.status_summary()
    versions = db.query(ModelVersion).order_by(ModelVersion.id.asc()).all()

    return {
        "models": [
            {
                "id": mv.id,
                "model_key": mv.model_key,
                "model_name": mv.model_name,
                "dataset_name": mv.dataset_name,
                "version": mv.version,
                "algorithm": mv.algorithm,
                "artifact_filename": mv.artifact_filename,
                "classification_threshold": mv.classification_threshold,
                "accuracy": mv.accuracy,
                "precision": mv.precision,
                "recall": mv.recall,
                "f1_score": mv.f1_score,
                "roc_auc": mv.roc_auc,
                "pr_auc": mv.pr_auc,
                "evaluation_dataset": mv.evaluation_dataset,
                "evaluation_notes": mv.evaluation_notes,
                "confusion_matrix": mv.confusion_matrix_json,
                "comparison_metrics": mv.comparison_metrics_json,
                "evaluated_at": mv.evaluated_at,
                "is_active": mv.is_active,
                "runtime_status": status_map.get(mv.model_key, {}),
            }
            for mv in versions
        ]
    }
