"""Simulated PaySim live transaction stream manager with WebSocket broadcast."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from fastapi import WebSocket

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.models.entities import User
from app.services.audit_service import record_audit
from app.services.transaction_service import run_and_persist_transactions


class SimulationManager:
    """Manages the SIMULATED TRANSACTION STREAM backed by sampled PaySim records."""

    def __init__(self) -> None:
        self.status: str = "STOPPED"  # STOPPED, RUNNING, PAUSED
        self.interval_seconds: float = 1.5
        self.processed_count: int = 0
        self.flagged_count: int = 0
        self.alerts_generated: int = 0
        self.cursor_index: int = 0
        self.total_pool_size: int = 0
        self.last_event_at: Optional[datetime] = None
        self.recent_events: list[dict[str, Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._clients: set[WebSocket] = set()
        self._rows: list[dict[str, Any]] = []

    def _load_sample_pool(self) -> None:
        if self._rows:
            return
        settings = get_settings()
        sample_path = settings.datasets_path / "samples" / "paysim_sample.csv"
        fallback_path = settings.datasets_path / "analysed_transactions.csv"
        target = sample_path if sample_path.exists() else fallback_path
        if not target.exists():
            raise FileNotFoundError(
                "No PaySim simulation source CSV found in credora/datasets/samples/paysim_sample.csv."
            )
        df = pd.read_csv(target)
        keep_cols = [
            c
            for c in [
                "step",
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFraud",
                "isFlaggedFraud",
            ]
            if c in df.columns
        ]
        self._rows = df[keep_cols].to_dict(orient="records")
        self.total_pool_size = len(self._rows)

    def get_state(self) -> dict[str, Any]:
        return {
            "feed_label": "SIMULATED TRANSACTION STREAM",
            "disclaimer": (
                "Simulated stream using sampled PaySim synthetic financial transactions "
                "(not connected to live bank feeds)."
            ),
            "status": self.status,
            "interval_seconds": self.interval_seconds,
            "processed_count": self.processed_count,
            "flagged_count": self.flagged_count,
            "alerts_generated": self.alerts_generated,
            "cursor_index": self.cursor_index,
            "total_pool_size": self.total_pool_size,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
            "recent_events": self.recent_events[:25],
        }

    async def connect_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        await ws.send_json({"type": "state", "data": self.get_state()})

    def disconnect_ws(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    def _log_control_audit(self, actor: Optional[User], action_detail: str) -> None:
        db = SessionLocal()
        try:
            record_audit(
                db=db,
                actor=actor,
                action="SIMULATION_CONTROL",
                entity_type="simulation",
                entity_id="paysim_stream",
                details={"event": action_detail, "interval_seconds": self.interval_seconds},
            )
            db.commit()
        finally:
            db.close()

    async def start(self, actor: Optional[User] = None) -> dict[str, Any]:
        self._load_sample_pool()
        if self.status == "RUNNING":
            return self.get_state()
        self.status = "RUNNING"
        self._log_control_audit(actor, "STARTED")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())
        await self.broadcast({"type": "state", "data": self.get_state()})
        return self.get_state()

    async def pause(self, actor: Optional[User] = None) -> dict[str, Any]:
        if self.status == "RUNNING":
            self.status = "PAUSED"
            self._log_control_audit(actor, "PAUSED")
            await self.broadcast({"type": "state", "data": self.get_state()})
        return self.get_state()

    async def resume(self, actor: Optional[User] = None) -> dict[str, Any]:
        self._load_sample_pool()
        if self.status == "PAUSED":
            self.status = "RUNNING"
            self._log_control_audit(actor, "RESUMED")
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run_loop())
            await self.broadcast({"type": "state", "data": self.get_state()})
        return self.get_state()

    async def stop(self, actor: Optional[User] = None) -> dict[str, Any]:
        self.status = "STOPPED"
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self._log_control_audit(actor, "STOPPED")
        await self.broadcast({"type": "state", "data": self.get_state()})
        return self.get_state()

    async def set_speed(self, interval_seconds: float, actor: Optional[User] = None) -> dict[str, Any]:
        self.interval_seconds = max(0.3, min(10.0, float(interval_seconds)))
        self._log_control_audit(actor, f"SPEED_SET_{self.interval_seconds}s")
        await self.broadcast({"type": "state", "data": self.get_state()})
        return self.get_state()

    def process_next_step(self) -> dict[str, Any]:
        """Synchronously process the next sampled PaySim row through the ML model and persist to DB."""
        self._load_sample_pool()
        row = self._rows[self.cursor_index % len(self._rows)]
        self.cursor_index += 1

        db = SessionLocal()
        try:
            outcome = run_and_persist_transactions(
                db=db,
                model_key="paysim",
                raw_rows=[row],
                actor=None,
                save_to_db=True,
                include_shap=False,
                is_simulated=True,
            )
        finally:
            db.close()

        preview = outcome["previews"][0]
        # Serialize datetime cleanly for JSON/WebSocket
        event_payload = {
            **preview,
            "transaction_time": preview["transaction_time"].isoformat()
            if hasattr(preview["transaction_time"], "isoformat")
            else str(preview["transaction_time"]),
        }

        self.processed_count += 1
        if preview["predicted_fraud"]:
            self.flagged_count += 1
        self.alerts_generated += len(outcome["alerts"])
        self.last_event_at = datetime.now(timezone.utc)
        self.recent_events.insert(0, event_payload)
        self.recent_events = self.recent_events[:25]
        return event_payload

    async def _run_loop(self) -> None:
        try:
            while self.status in ("RUNNING", "PAUSED"):
                if self.status == "PAUSED":
                    await asyncio.sleep(0.3)
                    continue
                event = await asyncio.to_thread(self.process_next_step)
                await self.broadcast(
                    {
                        "type": "transaction",
                        "data": event,
                        "state": self.get_state(),
                    }
                )
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            return


simulation_manager = SimulationManager()
