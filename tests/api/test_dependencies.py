"""Integration tests for `src.api.dependencies.CurrentUserDep`.

Builds a throwaway route on a fresh `FastAPI()` instance rather than
reusing the production app: this proves `get_current_user` composes
correctly through real FastAPI dependency injection over HTTP — no
production endpoint currently requires authentication (see
`docs/authentication.md`), so there is no production route to test this
against yet.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from src.api.dependencies import CurrentUserDep


def _build_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(current_user: CurrentUserDep) -> dict[str, str]:
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role.value,
        }

    return app


async def test_valid_credentials_resolve_the_current_user() -> None:
    transport = httpx.ASGITransport(app=_build_test_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/whoami", auth=("admin01", "Admin01#Poc2026"))

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin01"
    assert body["role"] == "ADMINISTRATOR"
    assert body["user_id"]


async def test_missing_credentials_are_rejected() -> None:
    transport = httpx.ASGITransport(app=_build_test_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/whoami")

    assert response.status_code == 401


async def test_wrong_password_is_rejected() -> None:
    transport = httpx.ASGITransport(app=_build_test_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/whoami", auth=("admin01", "wrong-password"))

    assert response.status_code == 401
