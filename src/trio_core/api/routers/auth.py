"""Authentication router for Trio Console (demo multi-tenant auth)."""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["console-auth"])

# ---------------------------------------------------------------------------
# Demo users (hardcoded — no database needed for demo)
# ---------------------------------------------------------------------------

DEMO_USERS = {
    "fund": {
        "password": "1234",
        "tenant": "fund",
        "display_name": "Renaissance Capital",
        "role": "investment_analyst",
    },
    "security": {
        "password": "abcd",
        "tenant": "security",
        "display_name": "Equinix Security",
        "role": "security_officer",
    },
    "mall": {
        "password": "abcd",
        "tenant": "benchmark",
        "display_name": "Accuracy Benchmark",
        "role": "benchmark_analyst",
    },
}

# ---------------------------------------------------------------------------
# Tenant-specific configurations
# ---------------------------------------------------------------------------

TENANT_CONFIG: dict[str, dict] = {
    "fund": {
        "dashboard_title": "Investment Intelligence",
        "report_type": "investment",
        "camera_filter": ["starbucks", "sbux", "cafe", "store", "shop", "retail", "coffee", "bottle", "westfield", "mall", "blue bottle"],
        "suggested_questions": [
            "What's the customer demographic profile and peak hours?",
            "Should I go long or short based on what you see?",
            "Calculate the estimated average ticket from drink sizes.",
        ],
    },
    "security": {
        "dashboard_title": "Security Operations Center",
        "report_type": "security",
        "camera_filter": ["data center", "entrance", "server", "warehouse", "parking", "dock", "gate"],
        "suggested_questions": [
            "Were there any unauthorized access attempts overnight?",
            "Who entered the server room in the last hour?",
            "Summarize today's security incidents.",
        ],
    },
    "benchmark": {
        "dashboard_title": "Accuracy Benchmark",
        "report_type": "investment",
        "camera_filter": ["mall", "benchmark", "test"],
        "suggested_questions": [
            "What is the counting accuracy vs ground truth?",
            "How many people were detected vs actual?",
            "Show me the error distribution over time.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    tenant: str
    display_name: str
    role: str


class UserInfo(BaseModel):
    username: str
    tenant: str
    display_name: str
    role: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_token(authorization: str | None) -> dict:
    """Decode a Bearer token and return the user record.

    Token format: base64(username:tenant) — simple demo auth.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization[len("Bearer "):]
    try:
        decoded = base64.b64decode(token).decode("utf-8")
        username, _tenant = decoded.split(":", 1)
    except Exception:
        raise HTTPException(401, "Invalid token")
    user = DEMO_USERS.get(username)
    if user is None:
        raise HTTPException(401, "Invalid token — user not found")
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Authenticate a demo user and return a token."""
    user = DEMO_USERS.get(body.username)
    if user is None or user["password"] != body.password:
        raise HTTPException(401, "Invalid username or password")

    token = base64.b64encode(f"{body.username}:{user['tenant']}".encode()).decode()
    return LoginResponse(
        token=token,
        tenant=user["tenant"],
        display_name=user["display_name"],
        role=user["role"],
    )


@router.get("/api/auth/me", response_model=UserInfo)
async def get_me(authorization: str | None = Header(None)):
    """Return current user info from token."""
    user = _decode_token(authorization)
    # Find the username from the user record
    username = next(k for k, v in DEMO_USERS.items() if v["tenant"] == user["tenant"])
    return UserInfo(
        username=username,
        tenant=user["tenant"],
        display_name=user["display_name"],
        role=user["role"],
    )


@router.post("/api/auth/logout")
async def logout():
    """Logout (no-op for demo — client clears token)."""
    return {"status": "ok"}


@router.get("/api/auth/config")
async def get_tenant_config(authorization: str | None = Header(None)):
    """Return tenant-specific configuration for the current user."""
    user = _decode_token(authorization)
    config = TENANT_CONFIG.get(user["tenant"], TENANT_CONFIG["fund"])
    return config
