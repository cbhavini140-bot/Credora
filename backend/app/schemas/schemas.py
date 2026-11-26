"""Pydantic request and response schemas for CREDORA APIs."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Auth & Users ---
class UserRegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=150)
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="ANALYST", pattern="^(ADMIN|ANALYST|RISK_MANAGER)$")


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse


class UserRoleUpdateRequest(BaseModel):
    role: str = Field(pattern="^(ADMIN|ANALYST|RISK_MANAGER)$")
    is_active: Optional[bool] = None


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=150)
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(default=None, min_length=8, max_length=128)


# --- Predictions & Transactions ---
class SinglePredictionRequest(BaseModel):
    model_key: str = Field(pattern="^(ulb|sparkov|paysim)$")
    features: dict[str, Any]
    save_transaction: bool = True
    include_shap: bool = True
    transaction_ref: Optional[str] = None


class AnalystDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(PENDING|CONFIRMED_FRAUD|CONFIRMED_LEGIT)$")
    note: Optional[str] = None


class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_ref: str
    dataset_source: str
    batch_id: Optional[str] = None
    is_simulated: bool
    amount: float
    transaction_time: datetime
    transaction_type: str
    merchant: Optional[str] = None
    category: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    gender: Optional[str] = None
    city_pop: Optional[int] = None
    customer_ref: Optional[str] = None
    counterparty_ref: Optional[str] = None
    raw_features_json: dict[str, Any]
    fraud_probability: float
    risk_score: float
    risk_level: str
    predicted_fraud: bool
    ground_truth_label: Optional[bool] = None
    analyst_decision: str
    decided_by_id: Optional[int] = None
    decided_at: Optional[datetime] = None
    created_at: datetime


class PaginatedTransactionsResponse(BaseModel):
    items: list[TransactionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# --- Alerts ---
class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    alert_code: str
    transaction_id: int
    priority: str
    status: str
    reason: str
    risk_score: float
    assigned_to_id: Optional[int] = None
    assigned_to_name: Optional[str] = None
    transaction_ref: Optional[str] = None
    transaction_amount: Optional[float] = None
    dataset_source: Optional[str] = None
    predicted_fraud: Optional[bool] = None
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None


class AlertUpdateRequest(BaseModel):
    status: Optional[str] = Field(
        default=None, pattern="^(OPEN|UNDER_REVIEW|ESCALATED|RESOLVED|FALSE_POSITIVE)$"
    )
    priority: Optional[str] = Field(default=None, pattern="^(CRITICAL|HIGH|MEDIUM|LOW)$")
    assigned_to_id: Optional[int] = None


# --- Investigations ---
class InvestigationCreateRequest(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    summary: Optional[str] = None
    priority: str = Field(default="HIGH", pattern="^(CRITICAL|HIGH|MEDIUM|LOW)$")
    assigned_to_id: Optional[int] = None
    transaction_ids: list[int] = Field(default_factory=list)
    initial_note: Optional[str] = None


class InvestigationUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=3, max_length=200)
    summary: Optional[str] = None
    status: Optional[str] = Field(
        default=None, pattern="^(OPEN|IN_PROGRESS|ESCALATED|CLOSED_FRAUD|CLOSED_LEGIT)$"
    )
    priority: Optional[str] = Field(default=None, pattern="^(CRITICAL|HIGH|MEDIUM|LOW)$")
    assigned_to_id: Optional[int] = None
    final_decision: Optional[str] = Field(
        default=None, pattern="^(PENDING|CONFIRMED_FRAUD|CONFIRMED_LEGIT)$"
    )


class InvestigationNoteRequest(BaseModel):
    content: str = Field(min_length=2, max_length=4000)


class LinkTransactionRequest(BaseModel):
    transaction_id: int


class AnalystNoteResponse(BaseModel):
    id: int
    investigation_id: int
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    content: str
    created_at: datetime


class InvestigationResponse(BaseModel):
    id: int
    case_code: str
    title: str
    summary: Optional[str] = None
    status: str
    priority: str
    final_decision: str
    assigned_to_id: Optional[int] = None
    assigned_to_name: Optional[str] = None
    created_by_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    transactions: list[TransactionResponse] = Field(default_factory=list)
    notes: list[AnalystNoteResponse] = Field(default_factory=list)


# --- Settings ---
class RiskThresholdSettings(BaseModel):
    medium_threshold: float = Field(ge=1.0, le=98.0)
    high_threshold: float = Field(ge=2.0, le=99.0)
