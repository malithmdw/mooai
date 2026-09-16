---
document_id: PROD-001
title: Novus Everyday Current Account — Product Specification
department: Retail Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-04-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Novus Everyday Current Account — Product Specification

## 1. Product Overview

The Novus Everyday Current Account (ECA) is the primary retail deposit and payments product. It is designed for individual customers aged 18+ resident in the UK.

## 2. Account Features

| Feature | Specification |
|---------|--------------|
| Account type | Personal current account |
| Currency | GBP |
| Minimum opening balance | £0 |
| Monthly fee | £0 (standard tier) |
| Interest rate (credit) | 0.5% AER on balances up to £1,000 |
| Overdraft | Arranged overdraft available (see PROD-004) |
| Debit card | Visa debit, contactless |
| Mobile banking | iOS and Android app |
| Online banking | Web portal |

## 3. Payment Capabilities

| Payment Type | Limit | Cut-off |
|---|---|---|
| Faster Payments (outbound) | £100,000 per transaction; £250,000 per day | None (24/7) |
| Bacs Direct Debit | No limit | As per mandate |
| Bacs Direct Credit | No limit | As per mandate |
| CHAPS | Not available (available on Business Current Account) | N/A |
| International (SWIFT) | Not available (available on Premium account) | N/A |
| Contactless card | £100 per transaction; £300 per day | N/A |

## 4. Eligibility Criteria

- UK resident aged 18+
- Not subject to a bankruptcy order or individual voluntary arrangement
- Valid UK address and proof of identity (KYC completed)

## 5. Account Opening Process

1. Customer completes online application (10 minutes)
2. KYC verification: eIDV check within 60 seconds
3. If eIDV fails: manual document review (1–3 business days)
4. Account provisioned in core-banking-ledger
5. Debit card dispatched within 3 business days
6. Mobile app activation via card number + memorable date

## 6. Regulatory Classification

- **Regulated product**: Yes
- **FSCS protection**: Yes (up to £85,000)
- **FCA category**: Payment account (Payment Accounts Regulations 2015)

## 7. Related Documents

- **PROD-004**: Arranged overdraft specification
- **POL-002**: Data retention policy
- **ARCH-001**: Core banking ledger architecture
