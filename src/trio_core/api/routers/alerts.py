"""Alerts API router for Trio Console."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/alerts", tags=["console-alerts"])


class AlertRuleIn(BaseModel):
    name: str
    camera_id: str | None = None
    condition: str
    channels: list[str] = Field(default_factory=list)
    cooldown_s: int = 60
    enabled: bool = True


class AlertRuleOut(BaseModel):
    id: str
    name: str
    camera_id: str | None = None
    condition: str
    channels: list[str] = Field(default_factory=list)
    cooldown_s: int = 60
    enabled: bool = True
    created_at: str = ""


class AlertHistoryOut(BaseModel):
    id: str
    rule_id: str
    event_id: str
    triggered_at: str
    channel: str
    status: str = "sent"


class AlertHistoryList(BaseModel):
    items: list[AlertHistoryOut]
    total: int
    limit: int
    offset: int


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


@router.get("/rules", response_model=list[AlertRuleOut])
async def list_rules(request: Request):
    """List all alert rules — filtered by tenant's cameras."""
    store = _get_store(request)
    rows = await store.list_alert_rules()

    # Tenant isolation
    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)
    if allowed_ids is not None:
        rows = [r for r in rows if r.get("camera_id") is None or r["camera_id"] in allowed_ids]

    return [AlertRuleOut(**r) for r in rows]


@router.post("/rules", response_model=AlertRuleOut, status_code=201)
async def create_rule(rule: AlertRuleIn, request: Request):
    """Create a new alert rule — tenant-isolated."""
    store = _get_store(request)
    # Tenant check: can only create rules for cameras you have access to
    if rule.camera_id:
        from trio_core.api.routers.auth import get_tenant_camera_ids
        allowed_ids = await get_tenant_camera_ids(request, store)
        if allowed_ids is not None and rule.camera_id not in allowed_ids:
            raise HTTPException(403, f"Access denied to camera {rule.camera_id}")
    rule_id = await store.create_alert_rule(rule.model_dump())
    # Fetch back for response
    rules = await store.list_alert_rules()
    for r in rules:
        if r["id"] == rule_id:
            return AlertRuleOut(**r)
    raise HTTPException(500, "Rule created but not found")


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(rule_id: str, request: Request):
    """Delete an alert rule — tenant-isolated."""
    store = _get_store(request)
    # Tenant check: only delete rules for cameras you own
    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)
    if allowed_ids is not None:
        rules = await store.list_alert_rules()
        rule = next((r for r in rules if r["id"] == rule_id), None)
        if rule and rule.get("camera_id") and rule["camera_id"] not in allowed_ids:
            raise HTTPException(403, "Access denied")
    deleted = await store.delete_alert_rule(rule_id)
    if not deleted:
        raise HTTPException(404, f"Rule {rule_id} not found")


@router.get("/history", response_model=AlertHistoryList)
async def list_history(
    request: Request,
    rule_id: str | None = Query(None),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List triggered alert history."""
    store = _get_store(request)
    result = await store.list_alert_history(
        rule_id=rule_id, start=start, end=end, limit=limit, offset=offset,
    )
    return AlertHistoryList(
        items=[AlertHistoryOut(**h) for h in result["items"]],
        total=result["total"],
        limit=limit,
        offset=offset,
    )
