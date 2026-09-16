---
document_id: ARCH-008
title: Fraud Detection Platform Architecture
department: Financial Crime
document_type: architecture_document
access_level: INTERNAL
created_date: "2024-01-10"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Fraud Detection Platform Architecture

## 1. Purpose

This document describes the architecture of the Fraud Detection Platform (FDP), which provides real-time and batch fraud risk scoring for card transactions, payment initiations, and account access events.

## 2. Architecture

```
payment-gateway / card-processing
        ↓ gRPC (< 200 ms SLO)
fraud-detection-service
        ├── feature-store (Redis — real-time features)
        ├── model-server (TensorFlow Serving / XGBoost)
        ├── rules-engine (Drools — static rules)
        └── model-registry (MLflow — model versioning)
                ↓
        decision: ALLOW / HOLD / DECLINE
                ↓
        case-management-system (for HOLD decisions)
```

## 3. ML Models

| Model | Type | Trained On | Refresh Cadence |
|-------|------|-----------|----------------|
| card-not-present-fraud | XGBoost | 18 months transaction history | Quarterly |
| account-takeover | Neural Network | Login event features | Monthly |
| wire-fraud | Gradient Boosted Trees | Wire transfer patterns | Quarterly |

> **Model drift incident**: INC-2024-009 — bank holiday weekend payment patterns caused 18% false positive rate. Drives requirement for bank holiday calendar feature in training data.

## 4. Feature Store

Real-time features are pre-computed and stored in Redis:

| Feature | Update Frequency |
|---------|----------------|
| Customer 24h transaction velocity | Per transaction |
| Merchant category fraud rate (30d) | Hourly |
| Device fingerprint seen count | Per login |
| Beneficiary first-time flag | Per payment |

## 5. Threshold Configuration

Model score thresholds are configurable per payment type:

| Payment Type | ALLOW | HOLD | DECLINE |
|---|---|---|---|
| Card-not-present | < 0.7 | 0.7–0.9 | > 0.9 |
| Wire transfer | < 0.5 | 0.5–0.8 | > 0.8 |
| FPS payment | < 0.6 | 0.6–0.85 | > 0.85 |

Thresholds are overridable by the Fraud Operations team via the case-management-system. Emergency threshold overrides (e.g., bank holiday pattern) must be logged in the fraud-detection runbook (RUN-009).

## 6. Model Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| False positive rate | > 1 % for 5 consecutive minutes |
| Model score drift (KS statistic) | > 0.15 |
| Feature store staleness | > 5 minutes |
| Model serving latency p99 | > 150 ms |

## 7. Related Documents

- **INC-2024-009**: Fraud model false positive surge
- **RUN-009**: Fraud detection and sanctions runbook
- **POL-009**: Fraud management policy
