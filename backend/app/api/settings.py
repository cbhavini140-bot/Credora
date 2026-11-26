"""System settings, risk threshold configuration, and RBAC permission matrix endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import get_current_user, require_roles
from app.database.session import get_db
from app.ml.model_registry import get_model_registry
from app.models.entities import SystemSetting, User
from app.schemas.schemas import RiskThresholdSettings
from app.services.audit_service import record_audit
from app.services.transaction_service import get_risk_thresholds

router = APIRouter(prefix="/api/settings", tags=["Settings"])

ROLE_PERMISSIONS = {
    "ADMIN": [
        "Manage user roles and account status",
        "Start, pause, resume, and stop simulated PaySim live stream",
        "Configure risk category boundaries",
        "Investigate transactions, manage alerts, and record analyst decisions",
        "Run single & batch ML predictions and view SHAP explanations",
    ],
    "RISK_MANAGER": [
        "Configure risk category boundaries",
        "Review portfolio risk analytics and model performance",
        "Investigate transactions, manage alerts, and record analyst decisions",
        "Run single & batch ML predictions and view SHAP explanations",
    ],
    "ANALYST": [
        "Run single & batch ML fraud detection",
        "Investigate transactions, manage alerts, and record analyst decisions",
        "Create and manage investigation cases and analyst notes",
        "View SHAP explanations, analytics, and audit logs",
    ],
}


@router.get("")
def get_system_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    med, high = get_risk_thresholds(db)
    registry = get_model_registry()

    return {
        "system": {
            "app_name": settings.APP_NAME,
            "tagline": settings.APP_TAGLINE,
            "environment": settings.APP_ENV,
            "database_backend": "PostgreSQL" if "postgresql" in settings.DATABASE_URL else "SQLite",
            "max_csv_upload_mb": settings.MAX_CSV_UPLOAD_MB,
            "academic_disclaimer": (
                "CREDORA is an academic BTech CSE AI-ML prototype. Simulated transaction monitoring "
                "uses sampled PaySim synthetic data and is not connected to live banking infrastructure."
            ),
        },
        "risk_thresholds": {
            "medium_threshold": med,
            "high_threshold": high,
        },
        "models": registry.status_summary(),
        "role_permissions": ROLE_PERMISSIONS,
    }


@router.put("/risk-thresholds")
def update_risk_thresholds(
    payload: RiskThresholdSettings,
    user: User = Depends(require_roles(["ADMIN", "RISK_MANAGER"])),
    db: Session = Depends(get_db),
):
    if payload.medium_threshold >= payload.high_threshold:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="medium_threshold must be strictly less than high_threshold.",
        )

    row = db.query(SystemSetting).filter(SystemSetting.key == "risk_thresholds").first()
    new_val = {
        "medium_threshold": payload.medium_threshold,
        "high_threshold": payload.high_threshold,
    }
    if row is None:
        row = SystemSetting(key="risk_thresholds", value_json=new_val, updated_by_id=user.id)
        db.add(row)
    else:
        row.value_json = new_val
        row.updated_by_id = user.id

    record_audit(
        db=db,
        actor=user,
        action="SETTINGS_UPDATED",
        entity_type="system_settings",
        entity_id="risk_thresholds",
        details=new_val,
    )
    db.commit()
    return {"risk_thresholds": new_val}
