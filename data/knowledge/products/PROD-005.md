---
document_id: PROD-005
title: Open Banking API — Third-Party Provider Product Specification
department: Open Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-08-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Open Banking API — Third-Party Provider Product Specification

## 1. Product Overview

The Novus Open Banking API allows FCA-authorised Third-Party Providers (TPPs) to access customer account information and initiate payments on behalf of consenting customers, in compliance with PSD2 and the Open Banking UK Standard.

## 2. Available APIs

| API | Standard | Version | Description |
|-----|---------|---------|-------------|
| Account Information (AISP) | Open Banking UK | v3.1.11 | Account balances, transactions, statement |
| Payment Initiation (PISP) | Open Banking UK | v3.1.11 | Domestic and international payment initiation |
| Confirmation of Funds (CBPII) | Open Banking UK | v3.1.11 | Balance check without full account info |

## 3. Authentication

- OAuth 2.0 with PKCE
- Dynamic Client Registration (DCR) per Open Banking Directory
- Customer consent via bank's hosted authorisation page (redirect model)
- Access token lifetime: 15 minutes
- Refresh token lifetime: 90 days (customer can revoke at any time)

## 4. Rate Limits

| API | Requests per minute per TPP |
|-----|----|
| Account Information | 300/min |
| Payment Initiation | 60/min |
| Confirmation of Funds | 120/min |
| Token endpoint | 30/min |

> See INC-2024-008 for rate limit incident when units were misconfigured to req/min instead of req/second.

## 5. Supported Payment Types (PISP)

| Payment Type | Supported | Max Amount |
|---|---|---|
| Domestic Faster Payments | Yes | £1,000,000 |
| International SWIFT | No (roadmap Q1 2025) | N/A |
| CHAPS | No (roadmap Q2 2025) | N/A |

## 6. TPP Registration Process

1. TPP applies via Open Banking Limited Directory
2. TPP provides eIDAS certificate or OBIE Transport Certificate
3. Novus validates FCA registration and scope
4. Novus registers TPP in tpp-registry within 5 business days
5. TPP can begin Dynamic Client Registration

## 7. Related Documents

- **ARCH-007**: Open Banking API architecture
- **INC-2024-008**: Rate limiter misconfiguration
- **POL-006**: Open Banking compliance policy
- **RUN-008**: Open Banking API runbook
