---
document_id: PROD-003
title: Novus Instant Access Savings Account — Product Specification
department: Retail Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-05-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Novus Instant Access Savings Account — Product Specification

## 1. Product Overview

The Novus Instant Access Savings Account (IASA) is a variable-rate savings product offering competitive interest with instant access to funds. Available to individual customers and jointly held.

## 2. Account Features

| Feature | Specification |
|---------|--------------|
| Account type | Instant access savings |
| Currency | GBP |
| Minimum balance | £1 |
| Maximum balance | £500,000 |
| Interest rate (variable) | 4.10% AER (as of 2024-09-01) |
| Interest payment | Monthly, credited to account |
| Notice period | None (instant access) |
| Withdrawals | Via linked current account only |

## 3. Withdrawal Rules

- Withdrawals must be made to a linked Novus current account or externally verified account
- Maximum 6 withdrawals per month (excess withdrawals incur £5 fee)
- Instant transfers to linked current account via Faster Payments (24/7)
- Transfers to external accounts via Faster Payments (up to £250,000 per transaction)

## 4. Interest Rate Management

- Rate is variable and set by the ALCO (Asset and Liability Committee)
- Rate changes require 14 days' notice to customers (PSR requirements)
- Rate change communication: in-app notification, email, and letter
- Historic rates published on public website for 5 years

## 5. FSCS Protection

Balances up to £85,000 (£170,000 for joint accounts) are protected by the Financial Services Compensation Scheme.

## 6. Technical Notes

- Account type in core-banking-ledger: `SAVINGS`
- Interest calculation: daily accrual, monthly credit
- Rate source: ALCO rate-setting service (internal API)
- Withdrawal validation: enforced by retail-banking-api withdrawal limits service

## 7. Related Documents

- **PROD-001**: Everyday Current Account (linked account requirement)
- **POL-002**: Data retention policy
- **ARCH-001**: Core banking ledger architecture
