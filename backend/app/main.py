"""CREDORA FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    alerts,
    analytics,
    audit,
    auth,
    explainability,
    investigations,
    predictions,
    settings as settings_api,
    simulation,
    transactions,
)
from app.core.config import get_settings
from app.database.seed import seed_database
from app.database.session import Base, SessionLocal, engine
from app.ml.model_registry import get_model_registry
from app.services.simulation_service import simulation_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure schema exists (migrations remain the canonical workflow, create_all ensures test/local safety)
    Base.metadata.create_all(bind=engine)
    # Load the 3 trained .joblib models once on startup (never retrained on startup)
    get_model_registry()
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
    yield
    await simulation_manager.stop()


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="CREDORA Fraud Detection & Transaction Monitoring API",
        description="TRUST IN EVERY TRANSACTION. Academic AI-ML fraud detection backend.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(transactions.router)
    app.include_router(predictions.router)
    app.include_router(alerts.router)
    app.include_router(investigations.router)
    app.include_router(analytics.router)
    app.include_router(explainability.router)
    app.include_router(audit.router)
    app.include_router(settings_api.router)
    app.include_router(simulation.router)

    @app.get("/api/health", tags=["Health"])
    def health_check():
        registry = get_model_registry()
        return {
            "status": "ok",
            "app": cfg.APP_NAME,
            "tagline": cfg.APP_TAGLINE,
            "models": registry.status_summary(),
        }

    return app


app = create_app()
