"""Alert management endpoints: list, filter, inspect, assign, and status updates."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session, joinedload

from app.core.security import get_current_user, require_roles
from app.database.session import get_db
from app.models.entities import Alert, Transaction, User
from app.schemas.schemas import AlertResponse, AlertUpdateRequest
from app.services.audit_service import record_audit

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


def _serialize_alert(a: Alert) -> AlertResponse:
    return AlertResponse(
        id=a.id,
        alert_code=a.alert_code,
        transaction_id=a.transaction_id,
        priority=a.priority,
        status=a.status,
        reason=a.reason,
        risk_score=a.risk_score,
        assigned_to_id=a.assigned_to_id,
        assigned_to_name=a.assigned_to.full_name if a.assigned_to else None,
        transaction_ref=a.transaction.transaction_ref if a.transaction else None,
        transaction_amount=float(a.transaction.amount) if a.transaction else None,
        dataset_source=a.transaction.dataset_source if a.transaction else None,
        predicted_fraud=a.transaction.predicted_fraud if a.transaction else None,
        created_at=a.created_at,
        updated_at=a.updated_at,
        closed_at=a.closed_at,
    )


@router.get("")
def list_alerts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    priority: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    assigned_to_id: Optional[int] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Alert).options(
        joinedload(Alert.transaction), joinedload(Alert.assigned_to)
    )
    if priority:
        query = query.filter(Alert.priority == priority.upper())
    if status_filter:
        query = query.filter(Alert.status == status_filter.upper())
    if assigned_to_id is not None:
        query = query.filter(Alert.assigned_to_id == assigned_to_id)
    if search:
        pat = f"%{search.strip()}%"
        query = query.join(Transaction, Alert.transaction_id == Transaction.id).filter(
            or_(
                Alert.alert_code.ilike(pat),
                Alert.reason.ilike(pat),
                Transaction.transaction_ref.ilike(pat),
            )
        )

    total = query.count()
    rows = (
        query.order_by(desc(Alert.created_at), desc(Alert.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [_serialize_alert(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.get("/{alert_id}")
def get_alert_detail(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    alert = (
        db.query(Alert)
        .options(joinedload(Alert.transaction), joinedload(Alert.assigned_to))
        .filter(Alert.id == alert_id)
        .first()
    )
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")

    return {
        "alert": _serialize_alert(alert),
        "transaction": alert.transaction,
    }


@router.patch("/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: int,
    payload: AlertUpdateRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    alert = (
        db.query(Alert)
        .options(joinedload(Alert.transaction), joinedload(Alert.assigned_to))
        .filter(Alert.id == alert_id)
        .first()
    )
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found.")

    changes: dict = {}
    if payload.status is not None:
        new_status = payload.status.upper()
        changes["status"] = {"from": alert.status, "to": new_status}
        alert.status = new_status
        if new_status in ("RESOLVED", "FALSE_POSITIVE"):
            alert.closed_at = datetime.now(timezone.utc)
        else:
            alert.closed_at = None

    if payload.priority is not None:
        new_priority = payload.priority.upper()
        changes["priority"] = {"from": alert.priority, "to": new_priority}
        alert.priority = new_priority

    if payload.assigned_to_id is not None:
        assignee = db.query(User).filter(User.id == payload.assigned_to_id).first()
        if assignee is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assigned user not found.")
        changes["assigned_to_id"] = {"from": alert.assigned_to_id, "to": assignee.id}
        alert.assigned_to_id = assignee.id

    action = "ANALYST_ASSIGNED" if "assigned_to_id" in changes and len(changes) == 1 else "ALERT_UPDATED"
    record_audit(
        db=db,
        actor=analyst,
        action=action,
        entity_type="alert",
        entity_id=alert.alert_code,
        details=changes,
    )
    db.commit()
    db.refresh(alert)
    return _serialize_alert(alert)
