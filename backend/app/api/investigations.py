"""Investigation case management, linked transactions, analyst notes, and final decisions."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc
from sqlalchemy.orm import Session, joinedload

from app.core.security import get_current_user, require_roles
from app.database.session import get_db
from app.models.entities import (
    AnalystNote,
    Investigation,
    InvestigationTransaction,
    Transaction,
    User,
)
from app.schemas.schemas import (
    AnalystNoteResponse,
    InvestigationCreateRequest,
    InvestigationNoteRequest,
    InvestigationResponse,
    InvestigationUpdateRequest,
    LinkTransactionRequest,
    TransactionResponse,
)
from app.services.audit_service import record_audit

router = APIRouter(prefix="/api/investigations", tags=["Investigations"])


def _serialize_investigation(inv: Investigation) -> InvestigationResponse:
    return InvestigationResponse(
        id=inv.id,
        case_code=inv.case_code,
        title=inv.title,
        summary=inv.summary,
        status=inv.status,
        priority=inv.priority,
        final_decision=inv.final_decision,
        assigned_to_id=inv.assigned_to_id,
        assigned_to_name=inv.assigned_to.full_name if inv.assigned_to else None,
        created_by_id=inv.created_by_id,
        created_at=inv.created_at,
        updated_at=inv.updated_at,
        closed_at=inv.closed_at,
        transactions=[
            TransactionResponse.model_validate(link.transaction)
            for link in inv.linked_transactions
            if link.transaction is not None
        ],
        notes=[
            AnalystNoteResponse(
                id=n.id,
                investigation_id=n.investigation_id,
                author_id=n.author_id,
                author_name=n.author.full_name if n.author else "System",
                content=n.content,
                created_at=n.created_at,
            )
            for n in inv.notes
        ],
    )


@router.get("")
def list_investigations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    priority: Optional[str] = None,
    assigned_to_id: Optional[int] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Investigation).options(
        joinedload(Investigation.assigned_to),
        joinedload(Investigation.linked_transactions).joinedload(InvestigationTransaction.transaction),
        joinedload(Investigation.notes).joinedload(AnalystNote.author),
    )
    if status_filter:
        query = query.filter(Investigation.status == status_filter.upper())
    if priority:
        query = query.filter(Investigation.priority == priority.upper())
    if assigned_to_id is not None:
        query = query.filter(Investigation.assigned_to_id == assigned_to_id)
    if search:
        pat = f"%{search.strip()}%"
        query = query.filter(Investigation.title.ilike(pat) | Investigation.case_code.ilike(pat))

    total = query.count()
    rows = (
        query.order_by(desc(Investigation.updated_at), desc(Investigation.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "items": [_serialize_investigation(inv) for inv in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.post("", response_model=InvestigationResponse, status_code=status.HTTP_201_CREATED)
def create_investigation(
    payload: InvestigationCreateRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    if payload.assigned_to_id is not None:
        assignee = db.query(User).filter(User.id == payload.assigned_to_id).first()
        if assignee is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignee user not found.")

    inv = Investigation(
        case_code=f"INV-{uuid.uuid4().hex[:8].upper()}",
        title=payload.title.strip(),
        summary=payload.summary,
        priority=payload.priority.upper(),
        status="OPEN",
        final_decision="PENDING",
        assigned_to_id=payload.assigned_to_id,
        created_by_id=analyst.id,
    )
    db.add(inv)
    db.flush()

    for tx_id in set(payload.transaction_ids):
        tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
        if tx is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction ID {tx_id} not found.",
            )
        db.add(InvestigationTransaction(investigation_id=inv.id, transaction_id=tx.id))

    if payload.initial_note and payload.initial_note.strip():
        db.add(
            AnalystNote(
                investigation_id=inv.id,
                author_id=analyst.id,
                content=payload.initial_note.strip(),
            )
        )

    record_audit(
        db=db,
        actor=analyst,
        action="INVESTIGATION_CREATED",
        entity_type="investigation",
        entity_id=inv.case_code,
        details={"title": inv.title, "priority": inv.priority, "transaction_ids": payload.transaction_ids},
    )
    db.commit()
    return get_investigation_detail(inv.id, analyst, db)


@router.get("/{investigation_id}", response_model=InvestigationResponse)
def get_investigation_detail(
    investigation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inv = (
        db.query(Investigation)
        .options(
            joinedload(Investigation.assigned_to),
            joinedload(Investigation.linked_transactions).joinedload(InvestigationTransaction.transaction),
            joinedload(Investigation.notes).joinedload(AnalystNote.author),
        )
        .filter(Investigation.id == investigation_id)
        .first()
    )
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found.")
    return _serialize_investigation(inv)


@router.patch("/{investigation_id}", response_model=InvestigationResponse)
def update_investigation(
    investigation_id: int,
    payload: InvestigationUpdateRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    inv = (
        db.query(Investigation)
        .options(
            joinedload(Investigation.linked_transactions).joinedload(InvestigationTransaction.transaction)
        )
        .filter(Investigation.id == investigation_id)
        .first()
    )
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found.")

    changes: dict = {}
    if payload.title is not None:
        inv.title = payload.title.strip()
        changes["title"] = inv.title
    if payload.summary is not None:
        inv.summary = payload.summary
        changes["summary_updated"] = True
    if payload.priority is not None:
        inv.priority = payload.priority.upper()
        changes["priority"] = inv.priority
    if payload.assigned_to_id is not None:
        assignee = db.query(User).filter(User.id == payload.assigned_to_id).first()
        if assignee is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignee user not found.")
        inv.assigned_to_id = assignee.id
        changes["assigned_to_id"] = assignee.id
    if payload.status is not None:
        inv.status = payload.status.upper()
        changes["status"] = inv.status
        if inv.status in ("CLOSED_FRAUD", "CLOSED_LEGIT"):
            inv.closed_at = datetime.now(timezone.utc)
        else:
            inv.closed_at = None

    if payload.final_decision is not None:
        decision = payload.final_decision.upper()
        inv.final_decision = decision
        changes["final_decision"] = decision
        if decision == "CONFIRMED_FRAUD":
            inv.status = "CLOSED_FRAUD"
            inv.closed_at = datetime.now(timezone.utc)
        elif decision == "CONFIRMED_LEGIT":
            inv.status = "CLOSED_LEGIT"
            inv.closed_at = datetime.now(timezone.utc)

        # Propagate confirmed decision to linked transactions so dashboard analytics stay consistent
        if decision in ("CONFIRMED_FRAUD", "CONFIRMED_LEGIT"):
            now = datetime.now(timezone.utc)
            for link in inv.linked_transactions:
                if link.transaction is not None:
                    link.transaction.analyst_decision = decision
                    link.transaction.decided_by_id = analyst.id
                    link.transaction.decided_at = now
                    for alert in link.transaction.alerts:
                        if alert.status in ("OPEN", "UNDER_REVIEW", "ESCALATED"):
                            alert.status = "RESOLVED" if decision == "CONFIRMED_FRAUD" else "FALSE_POSITIVE"
                            alert.closed_at = now

        record_audit(
            db=db,
            actor=analyst,
            action="ANALYST_DECISION",
            entity_type="investigation",
            entity_id=inv.case_code,
            details={"final_decision": decision, "linked_transaction_count": len(inv.linked_transactions)},
        )

    record_audit(
        db=db,
        actor=analyst,
        action="INVESTIGATION_UPDATED",
        entity_type="investigation",
        entity_id=inv.case_code,
        details=changes,
    )
    db.commit()
    return get_investigation_detail(inv.id, analyst, db)


@router.post("/{investigation_id}/notes", response_model=AnalystNoteResponse, status_code=status.HTTP_201_CREATED)
def add_investigation_note(
    investigation_id: int,
    payload: InvestigationNoteRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found.")

    note = AnalystNote(
        investigation_id=inv.id,
        author_id=analyst.id,
        content=payload.content.strip(),
    )
    db.add(note)
    db.flush()

    record_audit(
        db=db,
        actor=analyst,
        action="NOTE_ADDED",
        entity_type="investigation",
        entity_id=inv.case_code,
        details={"note_id": note.id, "preview": note.content[:120]},
    )
    db.commit()
    db.refresh(note)
    return AnalystNoteResponse(
        id=note.id,
        investigation_id=note.investigation_id,
        author_id=note.author_id,
        author_name=analyst.full_name,
        content=note.content,
        created_at=note.created_at,
    )


@router.post("/{investigation_id}/transactions", response_model=InvestigationResponse)
def link_transaction_to_investigation(
    investigation_id: int,
    payload: LinkTransactionRequest,
    analyst: User = Depends(require_roles(["ADMIN", "ANALYST", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    inv = db.query(Investigation).filter(Investigation.id == investigation_id).first()
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found.")
    tx = db.query(Transaction).filter(Transaction.id == payload.transaction_id).first()
    if tx is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found.")

    existing = (
        db.query(InvestigationTransaction)
        .filter(
            InvestigationTransaction.investigation_id == inv.id,
            InvestigationTransaction.transaction_id == tx.id,
        )
        .first()
    )
    if existing is None:
        db.add(InvestigationTransaction(investigation_id=inv.id, transaction_id=tx.id))
        record_audit(
            db=db,
            actor=analyst,
            action="INVESTIGATION_UPDATED",
            entity_type="investigation",
            entity_id=inv.case_code,
            details={"linked_transaction_ref": tx.transaction_ref},
        )
        db.commit()

    return get_investigation_detail(inv.id, analyst, db)
