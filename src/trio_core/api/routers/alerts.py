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
    """List all alert rules."""
    store = _get_store(request)
    rows = await store.list_alert_rules()
    return [AlertRuleOut(**r) for r in rows]


@router.post("/rules", response_model=AlertRuleOut, status_code=201)
async def create_rule(rule: AlertRuleIn, request: Request):
    """Create a new alert rule."""
    store = _get_store(request)
    rule_id = await store.create_alert_rule(rule.model_dump())
    # Fetch back for response
    rules = await store.list_alert_rules()
    for r in rules:
        if r["id"] == rule_id:
            return AlertRuleOut(**r)
    raise HTTPException(500, "Rule created but not found")


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(rule_id: str, request: Request):
    """Delete an alert rule."""
    store = _get_store(request)
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
