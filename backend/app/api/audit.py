"""Audit log querying endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.session import get_db
from app.models.entities import AuditLog, User

router = APIRouter(prefix="/api/audit", tags=["Audit Logs"])


@router.get("")
def list_audit_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1, le=200),
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action.upper())
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type.lower())
    if search:
        pat = f"%{search.strip()}%"
        query = query.filter(
            or_(
                AuditLog.actor_email.ilike(pat),
                AuditLog.entity_id.ilike(pat),
                AuditLog.action.ilike(pat),
            )
        )

    total = query.count()
    rows = (
        query.order_by(desc(AuditLog.created_at), desc(AuditLog.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": [
            {
                "id": r.id,
                "actor_id": r.actor_id,
                "actor_email": r.actor_email,
                "actor_role": r.actor_role,
                "action": r.action,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "details": r.details_json,
                "ip_address": r.ip_address,
                "created_at": r.created_at,
            }
            for r in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }
