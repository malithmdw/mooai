---
document_id: PROD-010
title: Novus Business Payments Hub — Product Specification
department: Business Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2024-03-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Novus Business Payments Hub — Product Specification

## 1. Product Overview

The Novus Business Payments Hub is a web and API-based payment management product for mid-market and enterprise business customers. It aggregates all payment types into a single dashboard with approval workflows, reporting, and ERP integration.

## 2. Key Features

| Feature | Description |
|---------|-------------|
| Payment initiation | FPS, Bacs, CHAPS, SWIFT in one interface |
| Bulk payment upload | CSV/XML upload for up to 50,000 payment records |
| Approval workflows | Configurable n-of-m authorisation |
| ERP integration | SAP, Oracle, Xero, QuickBooks API connectors |
| Real-time status | Live payment status from all schemes |
| Reporting | CAMT.053 bank statement, CSV export |

## 3. Approval Workflow Rules

| Payment Amount | Default Approval Requirement |
|---|---|
| < £10,000 | Single authoriser |
| £10,000 – £99,999 | Two authorisers |
| £100,000 – £999,999 | Two senior authorisers |
| ≥ £1,000,000 | Director-level authoriser + audit log |
| SWIFT international | Two authorisers + sanctions clearance confirmation |

Workflow rules are configurable per customer by the business banking team.

## 4. Integration Architecture

The Payments Hub integrates with the core payment systems via internal API:

```
Business Payments Hub (web + API)
        ↓
payment-gateway (orchestration)
        ├── fps-connector
        ├── bacs-submission-service
        ├── chaps-gateway
        └── swift-gateway
```

All payment instructions are subject to standard pre-payment checks (sanctions, fraud, idempotency) — see ARCH-003.

## 5. Audit and Compliance

- Full payment audit trail retained for 7 years
- All approval actions logged with user identity and timestamp
- Monthly reconciliation report generated automatically
- Compliant with PS21/3 (Contingent Reimbursement Model) for APP fraud

## 6. Pricing

| Tier | Monthly Fee | Included Payments | Additional |
|------|-----------|-------------------|-----------|
| Standard | £150 | 500 | £0.20 per payment |
| Professional | £400 | 2,000 | £0.15 per payment |
| Enterprise | Custom | Unlimited | Custom |

## 7. Related Documents

- **PROD-002**: Business Current Account (prerequisite)
- **ARCH-003**: Payment gateway architecture
- **POL-003**: Payment operations policy
- **POL-008**: Sanctions compliance policy
