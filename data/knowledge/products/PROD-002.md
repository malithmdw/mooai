---
document_id: PROD-002
title: Novus Business Current Account — Product Specification
department: Business Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-04-15"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Novus Business Current Account — Product Specification

## 1. Product Overview

The Novus Business Current Account (BCA) is designed for registered UK businesses (sole traders, partnerships, limited companies, LLPs). It provides full multi-scheme payment capabilities including CHAPS and international transfers.

## 2. Account Features

| Feature | Specification |
|---------|--------------|
| Account type | Business current account |
| Currency | GBP (multi-currency available via PROD-006) |
| Monthly fee | £8.50 (waived for first 12 months) |
| Minimum opening balance | £0 |
| Business debit card | Visa Business debit, contactless |
| Online banking | Business banking portal |
| API access | Open Banking AISP and PISP via authorised TPPs |

## 3. Payment Capabilities

| Payment Type | Limit | Cut-off |
|---|---|---|
| Faster Payments | £1,000,000 per transaction | 24/7 |
| Bacs Direct Credit (payroll) | Unlimited | Input by 22:30 (D-1) |
| Bacs Direct Debit (collections) | Unlimited | As per mandate |
| CHAPS | Unlimited | 17:00 same-day cut-off |
| SWIFT MT103 | Unlimited | 15:00 for same-day value |
| SEPA Credit Transfer | Not available (post-Brexit) | N/A |

## 4. Additional Business Features

- **Bulk payment file upload**: Supported (up to 10,000 records per Bacs file)
- **Multi-user access**: Up to 10 authorised signatories
- **Approval workflows**: Configurable dual-authorisation for payments > £10,000
- **Accounting integrations**: Xero, QuickBooks, Sage via Open Banking TPP
- **Dedicated relationship manager**: For accounts with average balance > £50,000

## 5. Eligibility Criteria

- UK-registered business entity
- Principal(s) must be UK resident and pass KYC
- Business must pass KYB (Know Your Business) including beneficial ownership check
- Not on PEP (Politically Exposed Person) list or subject to sanctions (see INC-2024-005, POL-008)

## 6. Sanctions and Compliance

All Business Current Account holders are subject to:
- Ongoing transaction monitoring by fraud-detection-service
- Sanctions screening on all outbound payments
- Enhanced Due Diligence (EDD) for high-risk business categories

## 7. Related Documents

- **PROD-001**: Everyday Current Account
- **PROD-006**: Multi-currency account
- **POL-003**: Payment operations policy
- **POL-008**: Sanctions compliance policy
- **ARCH-003**: Payment gateway architecture
