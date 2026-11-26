"""Audit trail helper service."""

from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models.entities import AuditLog, User


def record_audit(
    db: Session,
    action: str,
    actor: Optional[User] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """Persist an immutable audit log entry in the current database session."""
    entry = AuditLog(
        actor_id=actor.id if actor else None,
        actor_email=actor.email if actor else None,
        actor_role=actor.role if actor else "SYSTEM",
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        details_json=details or {},
        ip_address=ip_address,
    )
    db.add(entry)
    return entry
