"""Security: authentication, authorization (RBAC), and content safety.

This package owns every authorization decision in the system. Agents and
tools must call into `security` to check permissions rather than
implementing their own checks — see CLAUDE.md "Agents must never bypass
application authorization".

POC NOTE: authentication is intentionally hardcoded for this proof of
concept (see `auth.py`). Keycloak/OAuth integration is explicitly out of
scope until a later milestone — do not implement it prematurely.
"""
