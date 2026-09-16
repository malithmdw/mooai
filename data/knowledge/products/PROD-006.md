---
document_id: PROD-006
title: Multi-Currency Business Account — Product Specification
department: Business Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-09-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Multi-Currency Business Account — Product Specification

## 1. Product Overview

The Novus Multi-Currency Business Account (MCBA) allows businesses to hold balances and make/receive payments in multiple currencies without conversion costs. Available as an add-on to the Business Current Account (PROD-002).

## 2. Supported Currencies

| Currency | IBAN | SWIFT/BIC | Same-Day Settlement |
|----------|------|-----------|-------------------|
| GBP | Yes | Yes | Yes (CHAPS/FPS) |
| EUR | Yes | Yes | Yes (SEPA) |
| USD | No | Yes | No (typically T+1) |
| CHF | No | Yes | No (typically T+1) |
| AED | No | Yes | No (typically T+2) |

## 3. FX Conversion

- Competitive mid-market rate + 0.5% margin
- Rates updated every 15 minutes
- Rate lock available for 60 seconds during transaction completion
- No FX conversion fee for intra-account currency conversions
- FX limit: £5,000,000 equivalent per transaction (larger amounts via Treasury desk)

## 4. International Payments

| Payment Type | Currency | Delivery | Fee |
|---|---|---|---|
| SWIFT MT103 | Any | 1–5 days | £15 per payment |
| SEPA Credit Transfer | EUR | Next business day | £2 per payment |
| SEPA Instant | EUR | 10 seconds | £3.50 per payment |

All international payments are subject to sanctions screening (see ARCH-006, POL-008) and AML monitoring.

## 5. Technical Notes

- Multi-currency balances are held as separate ledger accounts per currency in core-banking-ledger
- FX conversion is processed by the fx-conversion-service which sources rates from Bloomberg FX
- SWIFT MT103 outbound payments are processed via swift-gateway (see ARCH-003)

## 6. Regulatory Considerations

- FCA regulated payment account
- FX business is conducted under Novus Bank's MiFID II investment firm authorisation
- AML screening for all international transactions > £10,000 equivalent

## 7. Related Documents

- **PROD-002**: Business Current Account
- **ARCH-003**: Payment gateway architecture
- **ARCH-006**: Sanctions screening architecture
- **POL-003**: Payment operations policy
- **POL-008**: Sanctions compliance policy
