"""Transaction listing, search, filtering, customer summaries, and analyst decisions."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import asc, desc, func, or_
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.database.session import get_db
from app.models.entities import Alert, Investigation, InvestigationTransaction, Transaction, User
from app.schemas.schemas import (
    AnalystDecisionRequest,
    PaginatedTransactionsResponse,
    TransactionResponse,
)
from app.services.audit_service import record_audit

router = APIRouter(prefix="/api/transactions", tags=["Transactions"])


@router.get("", response_model=PaginatedTransactionsResponse)
def list_transactions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    search: Optional[str] = None,
    dataset_source: Optional[str] = None,
    risk_level: Optional[str] = None,
    transaction_type: Optional[str] = None,
    analyst_decision: Optional[str] = None,
    predicted_fraud: Optional[bool] = None,
    is_simulated: Optional[bool] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Transaction)

    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Transaction.transaction_ref.ilike(pattern),
                Transaction.merchant.ilike(pattern),
                Transaction.category.ilike(pattern),
                Transaction.customer_ref.ilike(pattern),
                Transaction.counterparty_ref.ilike(pattern),
                Transaction.state.ilike(pattern),
            )
        )
    if dataset_source:
        query = query.filter(Transaction.dataset_source == dataset_source.lower())
    if risk_level:
        query = query.filter(Transaction.risk_level == risk_level.upper())
    if transaction_type:
        query = query.filter(Transaction.transaction_type.ilike(f"%{transaction_type.strip()}%"))
    if analyst_decision:
        query = query.filter(Transaction.analyst_decision == analyst_decision.upper())
    if predicted_fraud is not None:
        query = query.filter(Transaction.predicted_fraud == predicted_fraud)
    if is_simulated is not None:
        query = query.filter(Transaction.is_simulated == is_simulated)
    if min_amount is not None:
        query = query.filter(Transaction.amount >= min_amount)
    if max_amount is not None:
        query = query.filter(Transaction.amount <= max_amount)
    if start_date is not None:
        query = query.filter(Transaction.transaction_time >= start_date)
    if end_date is not None:
        query = query.filter(Transaction.transaction_time <= end_date)

    total = query.count()

    sort_map = {
        "created_at": Transaction.created_at,
        "transaction_time": Transaction.transaction_time,
        "amount": Transaction.amount,
        "risk_score": Transaction.risk_score,
        "fraud_probability": Transaction.fraud_probability,
    }
    sort_col = sort_map.get(sort_by, Transaction.created_at)
    order_fn = asc if sort_dir == "asc" else desc

    rows = (
        query.order_by(order_fn(sort_col), desc(Transaction.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    total_pages = max(1, (total + page_size - 1) // page_size)
    return PaginatedTransactionsResponse(
        items=[TransactionResponse.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/customers")
def list_customers_from_datasets(
    dataset_source: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Aggregate customer/account profiles strictly from datasets that carry customer identifiers (Sparkov & PaySim)."""
    query = db.query(
        Transaction.customer_ref,
        Transaction.dataset_source,
        func.max(Transaction.gender).label("gender"),
        func.max(Transaction.city).label("city"),
        func.max(Transaction.state).label("state"),
        func.max(Transaction.city_pop).label("city_pop"),
        func.count(Transaction.id).label("transaction_count"),
        func.sum(Transaction.amount).label("total_amount"),
        func.max(Transaction.risk_score).label("max_risk_score"),
        func.avg(Transaction.risk_score).label("avg_risk_score"),
    ).filter(Transaction.customer_ref.isnot(None))

    if dataset_source:
        query = query.filter(Transaction.dataset_source == dataset_source.lower())
    if search:
        query = query.filter(Transaction.customer_ref.ilike(f"%{search.strip()}%"))

    rows = (
        query.group_by(Transaction.customer_ref, Transaction.dataset_source)
        .order_by(desc("max_risk_score"))
        .limit(limit)
        .all()
    )

    ulb_count = db.query(func.count(Transaction.id)).filter(Transaction.dataset_source == "ulb").scalar() or 0

    return {
        "customers": [
            {
                "customer_ref": r.customer_ref,
                "dataset_source": r.dataset_source,
                "gender": r.gender,
                "city": r.city,
                "state": r.state,
                "city_pop": r.city_pop,
                "transaction_count": int(r.transaction_count),
                "total_amount": round(float(r.total_amount or 0.0), 2),
                "max_risk_score": round(float(r.max_risk_score or 0.0), 2),
                "avg_risk_score": round(float(r.avg_risk_score or 0.0), 2),
            }
            for r in rows
        ],
        "ulb_anonymized_count": int(ulb_count),
        "privacy_notice": (
            "ULB European Credit Card transactions contain only anonymized PCA components (V1-V28) "
            "and carry no customer names, card numbers, or demographic identifiers. "
            "Only Sparkov (masked cardholder profiles) and PaySim (synthetic origin accounts) appear here."
        ),
    }


@router.get("/{transaction_id}")
def get_transaction_detail(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")

    alerts = db.query(Alert).filter(Alert.transaction_id == tx.id).all()
    links = (
        db.query(Investigation)
        .join(InvestigationTransaction, InvestigationTransaction.investigation_id == Investigation.id)
        .filter(InvestigationTransaction.transaction_id == tx.id)
        .all()
    )
    return {
        "transaction": TransactionResponse.model_validate(tx),
        "alerts": [
            {
                "id": a.id,
                "alert_code": a.alert_code,
                "priority": a.priority,
                "status": a.status,
                "reason": a.reason,
                "created_at": a.created_at,
            }
            for a in alerts
        ],
        "investigations": [
            {
                "id": inv.id,
                "case_code": inv.case_code,
                "title": inv.title,
                "status": inv.status,
                "final_decision": inv.final_decision,
            }
            for inv in links
        ],
    }


@router.patch("/{transaction_id}/decision", response_model=TransactionResponse)
def update_analyst_decision(
    transaction_id: int,
    payload: AnalystDecisionRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    """Record a human analyst's determination separately from the ML model's prediction."""
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")

    old_decision = tx.analyst_decision
    tx.analyst_decision = payload.decision.upper()
    tx.decided_by_id = analyst.id
    tx.decided_at = datetime.now(timezone.utc)

    record_audit(
        db=db,
        actor=analyst,
        action="ANALYST_DECISION",
        entity_type="transaction",
        entity_id=tx.transaction_ref,
        details={
            "transaction_id": tx.id,
            "old_decision": old_decision,
            "new_decision": tx.analyst_decision,
            "note": payload.note,
            "model_predicted_fraud": tx.predicted_fraud,
        },
    )
    db.commit()
    db.refresh(tx)
    return tx
