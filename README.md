# Enterprise AI Assistant (POC)

An enterprise-grade AI Assistant proof of concept for a commercial bank.
The assistant searches organizational knowledge, answers questions grounded
in enterprise documents with citations, performs multi-step research using
external tools, enforces role-based access control, and is fully observable
via LangSmith.

> **Status: foundation only.** This repository currently contains the
> project scaffold — directory structure, configuration, and a health-check
> endpoint. No retrieval, agent, or LLM functionality is implemented yet.
> See [`CLAUDE.md`](./CLAUDE.md) for the standards all future work must
> follow, and "Roadmap" below for what's next.

## Business capabilities (target state)

- Search organizational knowledge
- Answer questions using enterprise documents
- Provide supporting evidence and citations
- Maintain conversation context
- Perform multi-step research
- Use external tools (via MCP)
- Demonstrate recursive language model (RLM) behavior
- Enforce RBAC
- Protect against prompt injection and data exfiltration
- Provide transparent agent execution
- Provide full LangSmith observability

## Technical stack

| Concern                | Technology                     |
|-------------------------|---------------------------------|
| Language / runtime      | Python 3.12                    |
| API                      | FastAPI                        |
| UI                       | Streamlit                      |
| Agent orchestration      | LangGraph                      |
| LLM                      | Anthropic Claude                |
| Embeddings               | OpenAI                          |
| Vector store             | Pinecone                        |
| Lexical search            | BM25                            |
| Relational store          | PostgreSQL                     |
| Cache / session store     | Redis                           |
| Tool protocol             | MCP                             |
| Observability              | LangSmith                       |
| Containerization           | Docker Compose                  |
| Testing                    | pytest, pytest-asyncio          |
| Linting                    | Ruff                             |
| Type checking               | mypy                            |

## Repository structure

```
.
├── README.md
├── CLAUDE.md              # standards & principles all contributors (incl. AI agents) must follow
├── pyproject.toml         # dependencies, ruff, mypy, pytest config
├── Makefile                # common dev commands
├── .env.example             # environment variable template
├── src/
│   ├── api/                # FastAPI app, routes — outermost HTTP layer
│   ├── agents/              # LangGraph agent graphs (RLM behavior)
│   ├── core/                # config, logging — no dependency on any other src package
│   ├── models/               # shared Pydantic schemas
│   ├── retrieval/            # hybrid search: Pinecone + BM25
│   ├── memory/                # conversation state (Postgres/Redis)
│   ├── tools/                  # external tool integrations (MCP)
│   ├── security/                # auth, RBAC, content safety
│   ├── observability/            # LangSmith tracing
│   └── services/                  # orchestration layer between api and everything else
├── tests/                    # pytest suite (mirrors src/ layout)
├── scripts/                   # dev-facing scripts (e.g. Streamlit entrypoint)
├── data/                       # local/POC data (gitignored except .gitkeep)
└── docs/                        # architecture & design notes
```

See [`CLAUDE.md`](./CLAUDE.md) "Dependency boundaries" for the allowed
import graph between these packages.

## Getting started

### Prerequisites

- Python 3.12
- Docker & Docker Compose (for Postgres/Redis, once those are wired up)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then fill in real secrets locally — never commit .env
```

### Run

```bash
make api          # FastAPI on http://localhost:8000 (docs at /docs)
make streamlit     # Streamlit placeholder UI
```

### Test & lint

```bash
make test          # pytest
make lint           # ruff check
make typecheck       # mypy
make check            # all of the above (CI gate)
```

## Authentication note (POC)

Authentication is **intentionally hardcoded** for this proof of concept
(`src/security/auth.py`, three example users — see
[`docs/authentication.md`](./docs/authentication.md) for credentials and
usage). Keycloak/OAuth integration is a deliberate, tracked future
milestone — do not implement it ahead of schedule, and do not treat the
hardcoded stub as representative of production security posture.

## Roadmap (not yet implemented)

- Retrieval pipeline (Pinecone + BM25 hybrid search, document ingestion)
- LangGraph agent graphs and RLM (recursive) reasoning
- Claude + OpenAI integration
- Conversation memory (Postgres/Redis)
- MCP tool integrations
- RBAC enforcement in `security`
- Prompt-injection / data-exfiltration defenses
- LangSmith tracing end-to-end
- Docker Compose stack (API, Streamlit, Postgres, Redis)
- Keycloak/OAuth (replacing hardcoded auth)

## License

Proprietary — internal POC, not for external distribution.
