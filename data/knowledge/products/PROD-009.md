---
document_id: PROD-009
title: Novus AI Financial Assistant — Product Specification (Internal POC)
department: Digital Banking
document_type: product_specification
access_level: RESTRICTED
created_date: "2024-06-01"
allowed_roles: ["ANALYST", "ADMINISTRATOR"]
---

# Novus AI Financial Assistant — Product Specification (Internal POC)

## 1. Overview

The Novus AI Financial Assistant is an internal proof-of-concept demonstrating the use of large language models (LLMs) for enterprise knowledge retrieval and customer query resolution. This specification describes the intended features, constraints, and safety controls for the POC.

## 2. Intended Use Cases

| Use Case | Description | Access |
|----------|-------------|--------|
| Internal knowledge search | Engineers and analysts query internal documents | ENGINEER, ANALYST |
| Incident analysis | Root cause analysis across incident history | ANALYST, ADMINISTRATOR |
| Regulatory Q&A | Policy and compliance queries | ANALYST, ADMINISTRATOR |
| Product Q&A | Customer-facing product information (future) | CUSTOMER (future) |

## 3. Safety and Security Controls

| Control | Description |
|---------|-------------|
| RBAC enforcement | Documents served according to user role (see ARCH-005) |
| Prompt injection defense | Input sanitisation; system prompt separation |
| Data exfiltration prevention | Output filtering; no retrieval of restricted docs for non-authorised roles |
| Hallucination mitigation | All answers grounded in retrieved documents with citations |
| PII masking | Customer PII masked before inclusion in retrieval context |
| Audit trail | All queries and responses logged in LangSmith |

## 4. Retrieval Architecture (Target)

- **Semantic search**: Pinecone vector database (OpenAI `text-embedding-3-small`)
- **Keyword search**: BM25 index (Elasticsearch)
- **Hybrid ranking**: Reciprocal Rank Fusion (RRF)
- **Re-ranking**: Cross-encoder (sentence-transformers)
- **Context window**: 8 documents × 512 tokens per document

## 5. LLM Configuration

- **Model**: Anthropic Claude (claude-sonnet-4-6 for POC)
- **Temperature**: 0 (grounded, deterministic responses)
- **System prompt**: Role-aware; enforces citation requirement
- **Max output tokens**: 2,048

## 6. Out of Scope (POC)

The following are explicitly out of scope for the POC:
- Real customer-facing deployment
- Processing of real customer PII
- Transactional actions (payments, account changes)
- Integration with live core banking systems

## 7. Related Documents

- **ARCH-005**: Authentication and API security
- **ARCH-009**: Observability platform (LangSmith)
- **POL-007**: Information security policy
