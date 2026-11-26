"""Authentication, user profile, and RBAC user-management endpoints."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_roles,
    verify_password,
)
from app.database.session import get_db
from app.models.entities import RevokedToken, User
from app.schemas.schemas import (
    ProfileUpdateRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    UserRoleUpdateRequest,
)
from app.services.audit_service import record_audit

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(
    payload: UserRegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Register a new user account (self-registration is restricted to ANALYST/RISK_MANAGER)."""
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    requested_role = payload.role.upper()
    if requested_role == "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration as ADMIN is not permitted. Contact a system administrator.",
        )

    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name.strip(),
        hashed_password=hash_password(payload.password),
        role=requested_role,
        is_active=True,
    )
    db.add(user)
    db.flush()

    record_audit(
        db=db,
        actor=user,
        action="USER_REGISTERED",
        entity_type="user",
        entity_id=str(user.id),
        details={"email": user.email, "role": user.role},
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login_user(
    payload: UserLoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Authenticate credentials and issue a signed JWT access token."""
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    user.last_login_at = datetime.now(timezone.utc)
    token, _, expires_at = create_access_token(subject=str(user.id), role=user.role)

    record_audit(
        db=db,
        actor=user,
        action="LOGIN",
        entity_type="user",
        entity_id=str(user.id),
        details={"email": user.email, "role": user.role},
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=token,
        expires_at=expires_at,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout")
def logout_user(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Invalidate the current JWT token by persisting its JTI in revoked_tokens."""
    jti: Optional[str] = getattr(current_user, "_token_jti", None)
    exp: Optional[datetime] = getattr(current_user, "_token_exp", None)
    if jti and exp:
        if db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is None:
            db.add(RevokedToken(jti=jti, expires_at=exp))

    record_audit(
        db=db,
        actor=current_user,
        action="LOGOUT",
        entity_type="user",
        entity_id=str(current_user.id),
        details={"email": current_user.email},
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    return {"message": "Successfully logged out and token invalidated."}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserResponse)
def update_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.full_name is not None:
        current_user.full_name = payload.full_name.strip()
    if payload.new_password is not None:
        if not payload.current_password or not verify_password(
            payload.current_password, current_user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect.",
            )
        current_user.hashed_password = hash_password(payload.new_password)

    record_audit(
        db=db,
        actor=current_user,
        action="PROFILE_UPDATED",
        entity_type="user",
        entity_id=str(current_user.id),
        details={"updated_name": payload.full_name is not None, "updated_password": payload.new_password is not None},
    )
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/users", response_model=list[UserResponse])
def list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List active users for alert/investigation assignment and admin management."""
    return db.query(User).order_by(User.full_name.asc()).all()


@router.patch("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    payload: UserRoleUpdateRequest,
    admin_user: User = Depends(require_roles(["ADMIN"])),
    db: Session = Depends(get_db),
):
    """Administrator-only endpoint to update a user's RBAC role or active status."""
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    old_role = target.role
    target.role = payload.role.upper()
    if payload.is_active is not None:
        target.is_active = payload.is_active

    record_audit(
        db=db,
        actor=admin_user,
        action="SETTINGS_UPDATED",
        entity_type="user",
        entity_id=str(target.id),
        details={
            "target_email": target.email,
            "old_role": old_role,
            "new_role": target.role,
            "is_active": target.is_active,
        },
    )
    db.commit()
    db.refresh(target)
    return target
