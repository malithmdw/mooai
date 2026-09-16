---
document_id: PROD-008
title: Novus Fixed Rate Bond — Product Specification
department: Retail Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2024-01-15"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Novus Fixed Rate Bond — Product Specification

## 1. Product Overview

The Novus Fixed Rate Bond is a fixed-term, fixed-rate savings product. Customers deposit a lump sum for a set period and receive a guaranteed interest rate. No withdrawals are permitted during the term.

## 2. Current Available Terms

| Term | AER (as of 2024-09-01) | Min Deposit | Max Deposit |
|------|----------------------|------------|------------|
| 6 months | 4.80% | £5,000 | £500,000 |
| 12 months | 4.65% | £5,000 | £500,000 |
| 24 months | 4.40% | £5,000 | £500,000 |
| 36 months | 4.20% | £5,000 | £500,000 |

## 3. Key Conditions

- No withdrawals during the fixed term (funds locked)
- Interest paid monthly or at maturity (customer choice at opening)
- At maturity, funds transferred to a linked Novus current or savings account
- Early closure: not permitted (force majeure exceptions only, subject to bank discretion)

## 4. Account Opening

- Online application only (not available in branch)
- Requires an existing Novus current account for funding and maturity payment
- Minimum opening deposit must be received within 10 business days of application
- Funding accepted via Faster Payments or internal transfer only (no cheques)

## 5. FSCS Protection

Protected by FSCS up to £85,000 per depositor. For bonds > £85,000, the excess is unprotected. Customers are informed of this at application.

## 6. Technical Notes

- Account type in core-banking-ledger: `SAVINGS` with sub-type `FIXED_TERM`
- Rate is set at account opening and does not change for the term
- Maturity event is processed by the core-banking-ledger scheduled job `bond-maturity-processor`
- Rate data sourced from ALCO rate-setting service

## 7. Related Documents

- **PROD-001**: Everyday Current Account (linked account required)
- **PROD-003**: Instant Access Savings Account
- **POL-002**: Data retention policy
- **ARCH-001**: Core banking ledger architecture
