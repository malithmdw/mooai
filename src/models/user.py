"""The authenticated caller and the RBAC roles bounding their access."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from src.models.common import EntityId, NonEmptyStr
from src.models.enums import Role


class User(BaseModel):
    """A user known to the system, and the roles that bound their access.

    Immutable: a `User` instance is a snapshot of identity/roles resolved
    for a given request, not a live-editable record — see
    `src.security.auth` for how it is currently (hardcoded) resolved.
    """

    model_config = ConfigDict(frozen=True)

    user_id: EntityId
    username: NonEmptyStr
    display_name: str | None = None
    roles: tuple[Role, ...]
    is_active: bool = True

    @field_validator("roles")
    @classmethod
    def _at_least_one_role(cls, roles: tuple[Role, ...]) -> tuple[Role, ...]:
        if not roles:
            raise ValueError("a user must have at least one role")
        return roles
