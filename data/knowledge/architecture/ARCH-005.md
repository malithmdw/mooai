---
document_id: ARCH-005
title: Authentication and API Security Architecture
department: Security Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-10-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Authentication and API Security Architecture

## 1. Purpose

This document describes the authentication and API security architecture for all Novus Bank digital services.

## 2. Authentication Flow

```
client (mobile app / web / TPP)
        ↓
api-gateway (TLS termination, rate limiting)
        ↓
auth-service (JWT issuance and validation)
        ↓
resource API (validates JWT; enforces RBAC)
```

### JWT Structure

```json
{
  "sub": "customer:uuid-v4",
  "roles": ["CUSTOMER"],
  "scope": ["accounts:read", "payments:initiate"],
  "iss": "https://auth.novusbank.internal",
  "exp": 1700000000,
  "jti": "unique-token-id"
}
```

JWT signing uses RS256 (2048-bit RSA). Signing keys are rotated every 90 days.

## 3. Role-Based Access Control

| Role | Description | Typical Principals |
|------|-------------|-------------------|
| CUSTOMER | Personal account access | Retail customers |
| BUSINESS | Business account access | SME customers |
| TPP | Open Banking Third-Party Provider | Authorised fintechs |
| ENGINEER | Internal engineering access | Engineering staff |
| ANALYST | Data analysis and reporting | Analysts, compliance staff |
| ADMINISTRATOR | Full system access | System administrators |

RBAC is enforced at the resource API layer. The api-gateway does not enforce business-level RBAC (only rate limits and TLS).

## 4. Open Banking (PSD2) Authentication

TPPs authenticate via OAuth 2.0 with Dynamic Client Registration (DCR) per the Open Banking UK specification. Access tokens are short-lived (15 minutes). Refresh tokens expire after 90 days or upon customer revocation.

> See INC-2024-008: Open Banking API rate limiter misconfiguration locked out all TPPs for 76 minutes.

## 5. Security Controls

| Control | Implementation |
|---------|---------------|
| Transport security | TLS 1.3 minimum; TLS 1.2 deprecated Q2 2025 |
| API authentication | JWT (RS256) for internal; OAuth 2.0 for TPPs |
| Rate limiting | Per-client limits at api-gateway (nginx rate limit module) |
| DAST scanning | OWASP ZAP in CI/CD pipeline |
| Secret scanning | detect-secrets pre-commit hook |
| Response sanitisation | Sensitive field masking in error serialisers |

> **Security incident**: INC-2024-006 — JWT tokens leaked in HTTP 401 error responses. Drives requirement for automated response body secret scanning in CI.

## 6. Threat Model (Summary)

| Threat | Control |
|--------|---------|
| Token theft | Short-lived JWTs (15 min), RS256, key rotation |
| Replay attack | jti (JWT ID) nonce validation, 15-min window |
| Credential stuffing | Rate limiting, CAPTCHA on login |
| Data exfiltration | Response sanitisation, RBAC, DLP monitoring |
| Prompt injection (AI) | Input validation, output filtering in AI services |

## 7. Related Documents

- **INC-2024-006**: JWT token leak in error response
- **INC-2024-008**: Open Banking rate limiter misconfiguration
- **RUN-010**: Incident response for security events
- **POL-007**: Information security policy
