# Authentication (POC only)

> **This is not real security.** It exists so RBAC and audit-logging code
> have a stable `AuthenticatedUser` to build against while the real
> identity provider (Keycloak/OAuth) is deliberately deferred — see
> CLAUDE.md "Authentication is intentionally hardcoded for this POC
> phase." Do not point this at anything that matters, and do not treat
> its presence as evidence the application is secure.

## What this is

Three fixed example users, authenticated via **HTTP Basic Auth**,
implemented in `src/security/auth.py` (verification logic) and
`src/security/poc_users.py` (the hardcoded user directory). A successful
authentication returns an `AuthenticatedUser` with exactly three fields:
`user_id`, `username`, `role`.

| Username | Password | Role |
|---|---|---|
| `viewer01` | `Viewer01#Poc2026` | `VIEWER` |
| `analyst01` | `Analyst01#Poc2026` | `ANALYST` |
| `admin01` | `Admin01#Poc2026` | `ADMINISTRATOR` |

These plaintext passwords are documented **here only** — they never
appear in `src/`. `poc_users.py` stores only a salted PBKDF2-HMAC-SHA256
hash per user (260,000 iterations, matching current common guidance for
that algorithm); `auth.py` verifies a login attempt by re-hashing the
supplied password with the same salt and comparing the result in
constant time (`hmac.compare_digest`), never with `==` on a raw string.

## Trying it

```bash
curl -u viewer01:Viewer01#Poc2026 http://localhost:8000/some-protected-route
```

```python
import httpx

response = httpx.get(
    "http://localhost:8000/some-protected-route",
    auth=("admin01", "Admin01#Poc2026"),
)
```

## Using it in code

- `src.security.auth.get_current_user` — the FastAPI **authentication
  dependency**. Extracts HTTP Basic credentials, verifies them, and
  either returns an `AuthenticatedUser` or raises `401`.
- `src.api.dependencies.CurrentUserDep` — the **current-user dependency**
  route handlers actually declare (`Annotated[AuthenticatedUser,
  Depends(get_current_user)]`), matching the existing `SettingsDep`
  pattern so routes never call `get_current_user` directly.
- `src.security.auth.authenticate(username, password)` — the underlying
  credential check, usable outside the FastAPI dependency graph (e.g.
  directly in tests) for the same reason `Settings` can be constructed
  directly in tests without going through `get_settings()`.

No route requires `CurrentUserDep` yet — this is foundation
infrastructure (see CLAUDE.md "Foundation before features"), ready for
the RBAC/authorization work and real endpoints that will depend on it
next, without needing to change once a real identity provider replaces
`get_current_user`.

## Invalid credentials handling

Every failure path — an unknown username, a wrong password, or missing
credentials entirely — returns the same `401` with the same generic
message (`"Invalid username or password."` / `"Authentication required."`
for the missing-credentials case), via the app's existing global
exception handler (`src.api.exception_handlers.http_exception_handler`),
which formats it as the same `ErrorResponse` shape as every other error in
this API. Unknown username and wrong password are **not** distinguished
in the response — that would let a caller enumerate valid usernames by
comparing error messages.

## What is explicitly NOT implemented

- Session or token issuance — every request must carry Basic Auth
  credentials again; there is no "log in once" flow.
- Credential rotation, expiry, or lockout after repeated failures.
- Multi-factor authentication.
- Any real identity provider (LDAP, SSO, Keycloak, OAuth) — tracked as a
  future milestone in README.md "Roadmap". A real dependency should be
  able to replace `get_current_user` without changing its signature or
  the `AuthenticatedUser` shape callers rely on.
- Role/permission *enforcement* — `AuthenticatedUser.role` is resolved
  here, but centralized authorization (checking that role against what
  an action requires) is separate, ongoing work in `src.security`.
