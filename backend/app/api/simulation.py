"""Simulated live PaySim monitoring REST controls and WebSocket endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from app.core.security import decode_access_token, get_current_user, require_roles
from app.database.session import SessionLocal
from app.models.entities import RevokedToken, User
from app.services.simulation_service import simulation_manager

router = APIRouter(prefix="/api/simulation", tags=["Simulated Live Monitoring"])


class SpeedUpdateRequest(BaseModel):
    interval_seconds: float = Field(ge=0.3, le=10.0)


@router.get("/status")
def get_simulation_status(current_user: User = Depends(get_current_user)):
    return simulation_manager.get_state()


@router.post("/start")
async def start_simulation(admin_user: User = Depends(require_roles(["ADMIN"]))):
    try:
        return await simulation_manager.start(actor=admin_user)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/pause")
async def pause_simulation(admin_user: User = Depends(require_roles(["ADMIN"]))):
    return await simulation_manager.pause(actor=admin_user)


@router.post("/resume")
async def resume_simulation(admin_user: User = Depends(require_roles(["ADMIN"]))):
    return await simulation_manager.resume(actor=admin_user)


@router.post("/stop")
async def stop_simulation(admin_user: User = Depends(require_roles(["ADMIN"]))):
    return await simulation_manager.stop(actor=admin_user)


@router.post("/speed")
async def update_simulation_speed(
    payload: SpeedUpdateRequest,
    admin_user: User = Depends(require_roles(["ADMIN"])),
):
    return await simulation_manager.set_speed(payload.interval_seconds, actor=admin_user)


@router.websocket("/ws")
async def simulation_websocket(websocket: WebSocket, token: str = Query(...)):
    """Authenticated WebSocket streaming simulated PaySim transactions in real time."""
    try:
        payload = decode_access_token(token)
        jti = payload.get("jti")
        sub = payload.get("sub")
        db = SessionLocal()
        try:
            revoked = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
            user = db.query(User).filter(User.id == int(sub)).first() if sub else None
            if revoked is not None or user is None or not user.is_active:
                await websocket.close(code=4001)
                return
        finally:
            db.close()
    except Exception:
        await websocket.close(code=4001)
        return

    await simulation_manager.connect_ws(websocket)
    try:
        while True:
            # Keep socket alive and allow ping/pong messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        simulation_manager.disconnect_ws(websocket)
