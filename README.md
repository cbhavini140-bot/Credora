# CREDORA — AI-Powered Credit Card Fraud Detection & Financial Transaction Monitoring

> **Academic Prototype** | BTech CSE AI-ML Final Year Project
> 
> ⚠ This is an academic prototype. It is **not** connected to live banking infrastructure or real financial data feeds. The PaySim transaction stream is explicitly simulated. All transaction data used in development is either anonymised, publicly available on Kaggle, or synthetically generated.

---

## Project Overview

**CREDORA** (Credit Risk & Detection Operations for Real-time Analysis) is a full-stack financial fraud detection and transaction monitoring platform built as a BTech final year AI-ML project.

The platform integrates **three independently trained machine learning models** across three different fraud datasets, provides real-time fraud scoring, SHAP-based explainable AI, batch CSV analysis, investigation workflows, and a live simulated transaction stream — all within a professional enterprise-grade interface.

**Tagline:** *Trust in Every Transaction.*

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CREDORA Platform                   │
├─────────────────┬───────────────────────────────────┤
│    Frontend      │         Backend (FastAPI)          │
│  React + Vite   │  Python 3.12 + SQLAlchemy + SHAP  │
│  TypeScript     │  3 × ML Models (.joblib)           │
│  Tailwind CSS   │  WebSocket simulation              │
│  Recharts       │  RBAC auth (JWT)                   │
│  Lucide Icons   │  Alembic migrations               │
└─────────────────┴────────────────┬────────────────--─┘
                                   │
                        ┌──────────▼──────────┐
                        │   PostgreSQL 17.12   │
                        │    credora_db        │
                        │   10 ORM tables      │
                        └─────────────────────┘
```

**Frontend:** React 19 · Vite 6 · TypeScript · Tailwind CSS v4 · React Router v7 · Axios · Recharts · Lucide React  
**Backend:** Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 · Uvicorn · Alembic  
**ML Stack:** Scikit-learn · XGBoost · SHAP · Pandas · NumPy · Joblib  
**Database:** PostgreSQL 17.12

---

## Machine Learning Models

Three completely independent models, each with its own preprocessing pipeline. They are **never** combined or cross-applied.

### Model 1 — ULB Random Forest (`credora_ulb_model.joblib`)

| Attribute | Detail |
|-----------|--------|
| **Dataset** | ULB European Credit Card Fraud Detection (Kaggle) |
| **Transactions** | 284,807 (492 fraud, 284,315 legitimate) |
| **Algorithm** | RandomForestClassifier |
| **Configuration** | n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, class_weight=balanced_subsample |
| **Features** | Time, Amount, V1–V28 (28 anonymised PCA components) |
| **Classification Threshold** | ~0.61 |

> ⚠ **V1–V28 are anonymised PCA-transformed features.** Their names have no business meaning and are never relabelled in this platform.

**Recorded comparison metrics (Colab training split):**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------:|-------:|---:|--------:|-------:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.9722 | 0.7159 |
| **Random Forest (selected)** | **0.8953** | **0.7857** | **0.8370** | **0.9593** | **0.8523** |
| XGBoost | 0.4780 | 0.8860 | 0.6190 | — | 0.8480 |

**Validation metrics at threshold ~0.61:** Precision 0.8841 · Recall 0.7722 · F1 0.8243 · PR-AUC 0.7728

---

### Model 2 — Sparkov XGBoost (`credora_sparkov_model.joblib`)

| Attribute | Detail |
|-----------|--------|
| **Dataset** | Sparkov Credit Card Fraud Detection (Kaggle) |
| **Train/Test files** | fraudTrain.csv / fraudTest.csv |
| **Algorithm** | XGBClassifier |
| **Configuration** | n_estimators=100, max_depth=4, lr=0.08, subsample=0.8, colsample_bytree=0.8, hist |
| **Features** | amt, city_pop, transaction_hour/day/month/weekday, merchant_distance_km, category, gender, state |
| **Classification Threshold** | 0.50 |

**Live evaluation on official `fraudTest.csv` (555,719 rows — independent test set):**

| Metric | Value |
|--------|------:|
| Precision | 0.1286 |
| Recall | 0.9576 |
| F1 | 0.2268 |
| ROC-AUC | 0.9941 |
| PR-AUC | 0.6979 |

**Confusion matrix:** TN=539,661 · FP=13,913 · FN=91 · TP=2,054

> The high recall / lower precision reflects the class-imbalanced nature of credit card fraud detection. The model prioritises catching fraud (low false negatives).

---

### Model 3 — PaySim XGBoost (`credora_paysim_model.joblib`)

| Attribute | Detail |
|-----------|--------|
| **Dataset** | PaySim Synthetic Financial Transactions (Kaggle) |
| **Algorithm** | XGBClassifier |
| **Configuration** | n_estimators=100, max_depth=4, lr=0.08, subsample=0.8, colsample_bytree=0.8, hist |
| **Features** | step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, origin_balance_change, destination_balance_change, origin_error, destination_error, hour, type |
| **Classification Threshold** | 0.50 |

> ⚠ PaySim is a **synthetic simulation** of mobile-money transactions. It is not genuine bank data.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔐 **RBAC Authentication** | JWT-based login with Admin, Analyst, Risk Manager roles |
| 📊 **Overview Dashboard** | Live KPIs, fraud rate, alert summary, model status |
| 💳 **Transaction Management** | Full transaction table with filtering, pagination, risk badges |
| 🤖 **Fraud Detection** | Single transaction scoring across all 3 models with risk explanation |
| 📁 **Batch CSV Analysis** | Upload CSV files for bulk fraud scoring |
| 🔴 **Alerts** | Auto-generated fraud alerts with priority levels and analyst assignment |
| 🔍 **Investigations** | Full investigation lifecycle — open, in-progress, closed, analyst notes |
| 👥 **Customer Risk Profiles** | Customer-level aggregated risk metrics |
| 📈 **Risk Analytics** | Charts for fraud by category, time, geography, amount distribution |
| 🧠 **Model Performance** | Per-model metrics, confusion matrix, comparison charts |
| 💡 **Explainable AI (SHAP)** | SHAP feature contributions for individual predictions |
| 🌊 **Live Simulation** | Simulated PaySim transaction stream via WebSocket (clearly labelled) |
| 📋 **Audit Logs** | Full audit trail of analyst actions |
| ⚙ **Settings** | Risk threshold configuration, model management |

---

## Project Structure

```
credora/
├── start_backend.ps1          # Backend startup script (PowerShell)
├── start_frontend.ps1         # Frontend startup script (PowerShell)
├── ml_models/                 # Trained .joblib model artifacts (3 files)
├── datasets/
│   ├── model_performance_report.csv
│   ├── analysed_transactions.csv
│   └── samples/               # 100-row CSV samples (ULB, Sparkov, PaySim)
├── backend/
│   ├── .env                   # Local dev config (not committed to VCS)
│   ├── .env.example           # Template
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── alembic/versions/      # DB migration: ae790ad5a718_initial_schema.py
│   └── app/
│       ├── main.py            # FastAPI app entrypoint
│       ├── core/              # Config, security (JWT, RBAC)
│       ├── database/          # SQLAlchemy session, Alembic seed
│       ├── models/            # 10 ORM entity models
│       ├── schemas/           # Pydantic request/response schemas
│       ├── ml/                # Feature engineering, model registry, SHAP
│       ├── services/          # Business logic (transactions, alerts, simulation)
│       └── api/               # 11 FastAPI routers (auth, transactions, predictions,
│                              #   alerts, investigations, analytics, explainability,
│                              #   audit, settings, simulation)
├── frontend/
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── pages/             # 13 application pages
│       ├── components/        # Reusable UI components
│       ├── layouts/           # DashboardLayout with sidebar navigation
│       ├── services/          # Axios API client
│       ├── context/           # React AuthContext
│       └── types/             # TypeScript type definitions
└── documentation/             # Architecture and design docs
```

---

## Setup & Installation

### Prerequisites

| Requirement | Minimum Version |
|-------------|----------------|
| Python | 3.12+ |
| Node.js | 20+ |
| PostgreSQL | 14+ (17.12 portable included in dev setup) |
| pip | 23+ |
| npm | 10+ |

---

### 1. PostgreSQL Setup

A portable PostgreSQL 17.12 installation was used in development. If you have a system PostgreSQL:

```sql
-- Connect as superuser and run:
CREATE ROLE credora WITH LOGIN PASSWORD 'credora_dev_password';
CREATE DATABASE credora_db OWNER credora;
GRANT ALL PRIVILEGES ON DATABASE credora_db TO credora;
```

If using the portable installation included in development:

```powershell
$pgBin  = "$env:LOCALAPPDATA\Programs\pgsql\pgsql\bin"
$pgData = "$env:LOCALAPPDATA\Programs\pgsql\pgdata"
# Start PostgreSQL (run this in a persistent PowerShell window or as a service)
& "$pgBin\postgres.exe" -D $pgData -p 5432
```

---

### 2. Backend Setup

```powershell
cd credora\backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env to set your DATABASE_URL if different from the default

# Apply database migrations
.\.venv\Scripts\alembic.exe upgrade head

# Start the backend
.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend seeds default accounts automatically on first startup.

**Verify:** `http://localhost:8000/api/health` — all 3 models should show `"loaded": true`

---

### 3. Frontend Setup

```powershell
cd credora\frontend

npm install
npm run dev
```

**Access:** `http://localhost:5173`

---

### Quick Start (PowerShell scripts)

```powershell
# Terminal 1 — Backend (also starts PostgreSQL)
.\credora\start_backend.ps1

# Terminal 2 — Frontend
.\credora\start_frontend.ps1
```

---

## Demo Credentials

| Role | Email | Password | Capabilities |
|------|-------|----------|--------------|
| **Admin** | `admin@credora.dev` | `CredoraAdmin!2026` | Full access — all features, settings, user management |
| **Analyst** | `analyst@credora.dev` | `CredoraAnalyst!2026` | Transactions, alerts, investigations, predictions, SHAP |
| **Risk Manager** | `risk@credora.dev` | `CredoraRisk!2026` | Analytics, model performance, read-only investigations |

---

## API Reference (Key Endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Obtain JWT access token |
| `GET` | `/api/health` | Health check — model load status |
| `POST` | `/api/predictions/score` | Score a single transaction |
| `POST` | `/api/predictions/batch` | Upload CSV for bulk scoring |
| `GET` | `/api/transactions` | List transactions (paginated, filtered) |
| `GET` | `/api/alerts` | List fraud alerts |
| `POST` | `/api/alerts/{id}/assign` | Assign alert to analyst |
| `GET` | `/api/investigations` | List investigations |
| `POST` | `/api/explainability/shap/{tx_id}` | SHAP explanation for a transaction |
| `GET` | `/api/analytics/overview` | Aggregated fraud KPIs |
| `GET` | `/api/models/performance` | Per-model performance metrics |
| `WS` | `/api/simulation/ws` | WebSocket simulated PaySim stream |

Full interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Running Tests

```powershell
# Backend (5 pytest tests — requires PostgreSQL running)
cd credora\backend
.\.venv\Scripts\pytest.exe -v

# Frontend (1 Vitest test)
cd credora\frontend
npm run test
```

**Expected results:** 5/5 backend tests passing · 1/1 frontend test passing

---

## Risk Score Interpretation

| Score Range | Category | Meaning |
|-------------|----------|---------|
| 0–29 | 🟢 LOW | Transaction is likely legitimate |
| 30–69 | 🟡 MEDIUM | Requires analyst review |
| 70–100 | 🔴 HIGH | High probability of fraud — alert generated |

Risk scores are derived from model fraud probabilities (0–100 scale). They are distinct from the binary classification threshold used by the model.

---

## Known Limitations

1. **Academic prototype** — not connected to real banking infrastructure or live data feeds.
2. **ULB V1–V28 features** are anonymised PCA components. The SHAP explainability page correctly presents them as such — no fabricated business names.
3. **PaySim simulation** is explicitly synthetic. The WebSocket stream samples from the PaySim dataset and labels all transactions as "SIMULATED TRANSACTION STREAM".
4. **Sparkov model** was trained and evaluated on the official Kaggle `fraudTrain.csv`/`fraudTest.csv` split. The high recall / lower precision is expected given severe class imbalance.
5. **XGBoost cross-platform compatibility** — model files were trained in Google Colab (Linux) and deployed on Windows. A UBJSON-compatible `compat_setstate` loader is applied automatically at startup. This is lossless (probability difference < 4×10⁻¹⁰).
6. **Frontend bundle size** — The production bundle (~700 kB gzipped) is acceptable for an academic prototype. Code-splitting is configured via `manualChunks` in `vite.config.ts`.

---

## Datasets (Kaggle Sources)

| Dataset | Kaggle URL |
|---------|-----------|
| ULB Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Sparkov Fraud Detection | https://www.kaggle.com/datasets/kartik2112/fraud-detection |
| PaySim Synthetic Transactions | https://www.kaggle.com/datasets/ealaxi/paysim1 |

> Dataset files are **not** included in this repository due to size. Download them from Kaggle and place them in `datasets/` as described in the ML notebooks.

---

## Academic Declaration

This project was developed as a BTech CSE final year academic submission. All machine learning models were trained independently by the student using publicly available Kaggle datasets. No proprietary or confidential financial data was used. The platform is a demonstration of AI/ML integration capabilities and is not intended for production deployment.

---

*CREDORA · BTech CSE AI-ML Final Year Project · 2025–2026*
