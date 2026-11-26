"""Database seed logic for default accounts, genuine model metrics, and initial settings."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.ml.model_registry import get_model_registry
from app.models.entities import ModelVersion, SystemSetting, User

# Recorded comparison metrics from the ULB Colab notebook (strictly as documented)
ULB_COMPARISON_METRICS = [
    {
        "model": "Logistic Regression",
        "precision": 0.0610,
        "recall": 0.9184,
        "f1_score": 0.1144,
        "roc_auc": 0.9722,
        "pr_auc": 0.7159,
        "scope": "ULB comparison split",
    },
    {
        "model": "Random Forest (Selected)",
        "precision": 0.8953,
        "recall": 0.7857,
        "f1_score": 0.8370,
        "roc_auc": 0.9593,
        "pr_auc": 0.8523,
        "scope": "ULB comparison split",
    },
    {
        "model": "XGBoost",
        "precision": 0.4780,
        "recall": 0.8860,
        "f1_score": 0.6190,
        "roc_auc": None,
        "pr_auc": 0.8480,
        "scope": "ULB comparison split",
    },
]


def _load_report_metrics(datasets_dir: Path) -> dict[str, dict[str, float]]:
    report_path = datasets_dir / "model_performance_report.csv"
    metrics_map: dict[str, dict[str, float]] = {}
    if not report_path.exists():
        return metrics_map
    with report_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Model") or "").strip()
            if not name:
                continue
            metrics_map[name] = {
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1_score": float(row["F1 Score"]),
                "roc_auc": float(row["ROC-AUC"]),
                "pr_auc": float(row["PR-AUC"]),
            }
    return metrics_map


def seed_database(db: Session) -> None:
    """Idempotently seed default RBAC users, genuine model version metadata, and risk thresholds."""
    settings = get_settings()

    # 1. Default RBAC accounts (demo credentials clearly documented in README)
    default_users = [
        ("admin@credora.dev", "Arjun Mehta (System Admin)", "CredoraAdmin!2026", "ADMIN"),
        ("analyst@credora.dev", "Priya Nair (Lead Fraud Analyst)", "CredoraAnalyst!2026", "ANALYST"),
        ("risk@credora.dev", "Vikram Rao (Risk Manager)", "CredoraRisk!2026", "RISK_MANAGER"),
    ]
    for email, full_name, raw_pw, role in default_users:
        existing = db.query(User).filter(User.email == email).first()
        if existing is None:
            db.add(
                User(
                    email=email,
                    full_name=full_name,
                    hashed_password=hash_password(raw_pw),
                    role=role,
                    is_active=True,
                )
            )

    # 2. Risk threshold settings
    if db.query(SystemSetting).filter(SystemSetting.key == "risk_thresholds").first() is None:
        db.add(
            SystemSetting(
                key="risk_thresholds",
                value_json={
                    "medium_threshold": settings.DEFAULT_RISK_MEDIUM_THRESHOLD,
                    "high_threshold": settings.DEFAULT_RISK_HIGH_THRESHOLD,
                },
            )
        )

    # 3. Model versions from genuine artifacts + model_performance_report.csv
    report_metrics = _load_report_metrics(settings.datasets_path)
    registry = get_model_registry()

    model_specs = [
        {
            "model_key": "ulb",
            "report_key": "ULB Random Forest",
            "model_name": "CREDORA Random Forest",
            "dataset_name": "ULB European Credit Card Fraud",
            "algorithm": "RandomForestClassifier (n_estimators=50, max_depth=10, balanced_subsample)",
            "artifact_filename": "credora_ulb_model.joblib",
            "default_threshold": 0.6100393414870652,
            "evaluation_dataset": "ULB creditcard.csv (Holdout Evaluation Split)",
            "evaluation_notes": (
                "Metrics loaded from model_performance_report.csv. Validation split recorded "
                "Precision=0.8841, Recall=0.7722, F1=0.8243, PR-AUC=0.7728 at threshold ~0.61. "
                "Features V1-V28 are anonymized PCA components."
            ),
            "comparison": ULB_COMPARISON_METRICS,
        },
        {
            "model_key": "sparkov",
            "report_key": "Sparkov XGBoost",
            "model_name": "CREDORA Sparkov XGBoost",
            "dataset_name": "Sparkov Credit Card Fraud Detection",
            "algorithm": "XGBClassifier (n_estimators=100, max_depth=4, lr=0.08, hist)",
            "artifact_filename": "credora_sparkov_model.joblib",
            "default_threshold": 0.5,
            "evaluation_dataset": "Sparkov fraudTest.csv (Official Independent Test Dataset)",
            "evaluation_notes": (
                "Metrics loaded from model_performance_report.csv evaluated against official "
                "independent fraudTest.csv. Engineered features include temporal attributes and "
                "Haversine merchant_distance_km."
            ),
            "comparison": None,
        },
        {
            "model_key": "paysim",
            "report_key": "PaySim XGBoost",
            "model_name": "CREDORA PaySim XGBoost",
            "dataset_name": "PaySim Synthetic Financial Transactions",
            "algorithm": "XGBClassifier (n_estimators=100, max_depth=4, lr=0.08, hist)",
            "artifact_filename": "credora_paysim_model.joblib",
            "default_threshold": 0.5,
            "evaluation_dataset": "PaySim PS_20174392719_1491204439457_log.csv (Synthetic Holdout Split)",
            "evaluation_notes": (
                "Metrics loaded from model_performance_report.csv. Dataset is synthetic mobile-money "
                "simulation data, not genuine banking transactions."
            ),
            "comparison": None,
        },
    ]

    for spec in model_specs:
        m = report_metrics.get(spec["report_key"], {})
        confusion_matrix_val = None
        if spec["model_key"] == "sparkov":
            # Official independent fraudTest.csv evaluation (555,719 rows)
            m = {
                "accuracy": 0.9748002137771068,
                "precision": 0.12864032066136408,
                "recall": 0.9575757575757575,
                "f1_score": 0.22681095406360424,
                "roc_auc": 0.9941339735603917,
                "pr_auc": 0.6978946049214898,
            }
            confusion_matrix_val = {
                "tn": 539661,
                "fp": 13913,
                "fn": 91,
                "tp": 2054,
                "total_samples": 555719,
                "evaluation_dataset": "fraudTest.csv",
            }
            spec["evaluation_notes"] = (
                "Evaluated directly against official independent fraudTest.csv (555,719 rows). "
                "Earlier Colab model_performance_report.csv recorded split metrics: "
                "Precision=0.1784, Recall=0.9647, F1=0.3012, ROC-AUC=0.9958, PR-AUC=0.7809."
            )

        threshold = spec["default_threshold"]
        if spec["model_key"] in registry.bundles:
            threshold = registry.bundles[spec["model_key"]].threshold

        existing_mv = (
            db.query(ModelVersion).filter(ModelVersion.model_key == spec["model_key"]).first()
        )
        if existing_mv is None:
            db.add(
                ModelVersion(
                    model_key=spec["model_key"],
                    model_name=spec["model_name"],
                    dataset_name=spec["dataset_name"],
                    version="1.0.0",
                    algorithm=spec["algorithm"],
                    artifact_filename=spec["artifact_filename"],
                    classification_threshold=threshold,
                    accuracy=m.get("accuracy"),
                    precision=m.get("precision"),
                    recall=m.get("recall"),
                    f1_score=m.get("f1_score"),
                    roc_auc=m.get("roc_auc"),
                    pr_auc=m.get("pr_auc"),
                    evaluation_dataset=spec["evaluation_dataset"],
                    evaluation_notes=spec["evaluation_notes"],
                    confusion_matrix_json=confusion_matrix_val,
                    comparison_metrics_json=spec["comparison"],
                    evaluated_at=datetime.now(timezone.utc),
                    is_active=True,
                )
            )
        else:
            existing_mv.accuracy = m.get("accuracy")
            existing_mv.precision = m.get("precision")
            existing_mv.recall = m.get("recall")
            existing_mv.f1_score = m.get("f1_score")
            existing_mv.roc_auc = m.get("roc_auc")
            existing_mv.pr_auc = m.get("pr_auc")
            existing_mv.evaluation_notes = spec["evaluation_notes"]
            existing_mv.confusion_matrix_json = confusion_matrix_val

    db.commit()
