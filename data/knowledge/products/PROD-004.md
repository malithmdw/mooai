---
document_id: PROD-004
title: Arranged Overdraft — Product Specification
department: Retail Banking
document_type: product_specification
access_level: INTERNAL
created_date: "2023-06-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Arranged Overdraft — Product Specification

## 1. Product Overview

The Novus Arranged Overdraft is a revolving credit facility linked to the Everyday Current Account (PROD-001), allowing customers to borrow short-term up to an agreed limit.

## 2. Key Terms

| Feature | Specification |
|---------|--------------|
| Interest rate | 39.9% EAR (variable) |
| Arrangement fee | £0 |
| Minimum limit | £100 |
| Maximum limit | £3,000 |
| Repayment | Automatic when account returns to credit |
| Buffer | £10 interest-free buffer below zero |

## 3. Eligibility Criteria

- Must hold a Novus Everyday Current Account for at least 3 months
- Credit score assessment (minimum Experian score: 600)
- No CCJs, IVAs, or defaults in last 3 years
- Income verification: minimum £12,000 annual income

## 4. Credit Decision Process

1. Customer applies via mobile app or online banking
2. Soft credit search (no impact on credit file) to check basic eligibility
3. If eligible: hard credit search (Experian) and internal affordability assessment
4. Credit decision within 60 seconds for 95% of applications
5. Limit set by credit engine (model: `credit-scoring-service v3.2`)

## 5. Regulatory Requirements

- **FCA Consumer Credit regime**: Regulated credit agreement (CCA 1974)
- **Pre-contract information**: SECCI (Standard European Consumer Credit Information) provided before agreement
- **Cooling-off period**: 14 days from agreement date
- **CONC sourcebook**: Affordability assessment required; responsible lending obligations apply

## 6. Declined Application Handling

If declined:
- Customer receives written notification within 5 business days
- Decline reasons not provided (credit reference agency guidelines)
- Customer advised to check credit report via free statutory right

## 7. Related Documents

- **PROD-001**: Everyday Current Account
- **POL-002**: Data retention policy
