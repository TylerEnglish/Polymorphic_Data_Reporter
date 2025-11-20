# Polymorphic Data Reporter

> A self-evolving, polymorphic data & reporting engine for building reusable, testable, and AI-assisted data workflows.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Build](https://img.shields.io/badge/status-WIP-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

**Polymorphic Data Reporter (PDR)** is a framework for building **composable, testable data workflows** that can:

- Ingest data from multiple sources (files, databases, APIs, SaaS tools)
- Normalize and unify schemas into a shared internal model
- Generate **consistent reports and features** for analytics, ML, and dashboards
- Use **RAG-style (Retrieval Augmented Generation)** and “agent” patterns to help reason about data, generate code, and self-improve over time
- Emphasize **Locality of Behavior (LOB)** – every module does one thing and lives close to related behavior (connectors, schemas, rules, tests, etc.)

The long-term goal is to have a **polymorphic reporting engine** that can point at a new domain (timesheets, Procore analytics, finance, housing, etc.) and reuse the same patterns: ingest → normalize → validate → snapshot → report → iterate.

---

## Core Ideas

- **Polymorphic Data Model**  
  Data from different systems is mapped into a consistent internal representation (tables/frames/objects) so downstream logic doesn’t care where it came from.

- **Connectors & Adapters**  
  Swappable connectors for CSV/Excel, SQL databases, REST APIs, and vendor-specific platforms. Each connector has:
  - A focused ingestion layer
  - A schema mapping layer
  - Tests & small example fixtures

- **RAG & Knowledge Layer**  
  Textual metadata, docs, and configuration are stored and indexed so an AI agent (Everabot) can:
  - Explain pipelines
  - Propose schema changes
  - Suggest tests or transformations
  - Help debug issues

- **Delta / Snapshot Engine**  
  Emphasis on **incremental updates** and **snapshots**:
  - Track what changed between runs
  - Persist “known good” states
  - Make it easy to reproduce reports

- **Locality of Behavior (LOB)**  
  Code is organized so that:
  - Each feature lives near its tests, config, and docs
  - Behavior and data definitions are close together
  - Cross-cutting concerns (logging, config, shared utils) are centralized but small

---

## Repository Layout

> This is a conceptual layout; adjust names to match your actual folders.

```text
Polymorphic_Data_Reporter/
├─ src/                         # Main source code
│  ├─ polymorphic_data_reporter/
│  │  ├─ connectors/            # Data source connectors (CSV, APIs, DBs, etc.)
│  │  ├─ schemas/               # Typed models, schema definitions, validators
│  │  ├─ pipelines/             # Orchestrated pipelines / DAG-like flows
│  │  ├─ ruleset/               # Rules, triage logic, DSLs, configs
│  │  ├─ rag/                   # Retrieval, embedding, knowledge store, prompts
│  │  ├─ snapshots/             # Snapshot & delta encoding utilities
│  │  ├─ bot/                   # “Everabot” / agent orchestration
│  │  ├─ app/                   # CLI / service entrypoints, API, UI hooks
│  │  ├─ utils/                 # Shared helpers (logging, config, etc.)
│  │  └─ __init__.py
├─ tests/                       # Unit + integration tests for each module
├─ examples/                    # Example configs, demo pipelines, sample data
├─ pyproject.toml               # Poetry project definition
├─ .env.example                 # Example environment configuration
└─ README.md                    # This file
```

---

## Tech Stack

* **Language**: Python 3.11+
* **Env / Packaging**: [Poetry](https://python-poetry.org/)
* **Data Layer**:

  * [Polars](https://www.pola.rs/) for fast in-memory dataframes
  * SQLite / DuckDB (or similar) for lightweight storage and caching
* **Testing**: `pytest`
* **Linting / Style**: `ruff`, `black` (or similar – adjust to match your tooling)
* **AI / RAG (planned or in progress)**: LLM integration via a connector layer; vector store for embeddings.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TylerEnglish/Polymorphic_Data_Reporter.git
cd Polymorphic_Data_Reporter
```

### 2. Install Dependencies (Poetry)

```bash
poetry install
```

If you don’t have Poetry yet:

```bash
pip install poetry
```

### 3. Environment Configuration

Copy the example environment file and fill in any required secrets (API keys, DB URIs, etc.):

```bash
cp .env.example .env
```

Typical settings might include:

* `PDR_ENV` – environment name (e.g. `dev`, `prod`)
* `PDR_DATA_DIR` – base path for local data, snapshots, and logs
* Connector-specific variables (e.g. `PROCORE_API_KEY`, `FINANCE_DB_URL`, etc.)

---

## Usage

> Adjust the commands below to match your actual entrypoint(s).

### Run a Demo Pipeline

```bash
poetry run python -m polymorphic_data_reporter.cli demo
```

What a typical pipeline might do:

1. Load raw data from a connector (e.g. CSV, Procore API, timesheet DB)
2. Normalize & validate against internal schemas
3. Compute derived columns / metrics
4. Write snapshots + deltas to disk or a DB
5. Emit a report (table, feature set, or JSON) to `outputs/` or your chosen sink

### Listing Available Pipelines

```bash
poetry run python -m polymorphic_data_reporter.cli list-pipelines
```

### Running a Specific Pipeline

```bash
poetry run python -m polymorphic_data_reporter.cli run --pipeline <pipeline_name> --config examples/<pipeline_name>.yml
```

---

## Design Principles

* **Locality of Behavior (LOB)**
  A pipeline’s code, schema, tests, and configs live close together. If you’re changing how something works, you shouldn’t need to hunt across the entire repo.

* **Composable Building Blocks**

  * Connectors are composable units of ingestion
  * Schemas and validators are reusable across domains
  * Pipelines are built from small, testable steps

* **Config-First, Code-Second**
  Much behavior is driven by configuration (YAML/JSON/py) where possible, so you can:

  * Spin up new pipelines faster
  * Parameterize sources, filters, and outputs without rewriting code

* **Testability from Day 1**

  * Each connector has mock/sample data in `tests/fixtures`
  * Pipelines can be run in “dry run” or “local only” modes
  * Schema validation catches issues early

---

## Development

### Run Tests

```bash
poetry run pytest
```

### Lint & Format

```bash
poetry run ruff check .
poetry run black .
```

(Adjust the tools/commands above to match what you actually use.)

### Adding a New Connector

1. Create a new module under `connectors/`
2. Define:

   * A small config object / dataclass
   * A `fetch()` or `run()` function that returns a Polars/DataFrame-like object
3. Add schema mappings in `schemas/`
4. Add tests under `tests/connectors/test_<name>.py`
5. Wire the connector into one or more pipelines in `pipelines/`

---

## Roadmap

High-level roadmap (subject to change):

1. **Task 0 – Core Scaffolding**

   * Base project structure, config, logging
   * Simple local connector (CSV / SQLite) + one demo pipeline

2. **Task 1 – Schema & Validation Layer**

   * Typed schemas and validators
   * Common core models shared across domains

3. **Task 2 – Connectors Library**

   * File-based sources (CSV, Excel, Parquet)
   * Relational DBs (e.g. Postgres, SQLite, DuckDB)
   * HTTP / SaaS connectors (Procore, timesheets, finance APIs, etc.)

4. **Task 3 – Snapshot & Delta Engine**

   * Incremental updates
   * Reproducible historical runs

5. **Task 4 – RAG / Knowledge Layer**

   * Ingest configs, docs, and code snippets into a knowledge store
   * LLM-powered helper to explain pipelines and suggest changes

6. **Task 5 – Everabot / Agent Layer**

   * Agent(s) that:

     * Propose schema updates
     * Suggest tests
     * Help design new pipelines

7. **Task 6 – UI / Reporting**

   * Basic CLI and/or web UI to:

     * Browse pipelines & runs
     * Inspect snapshots
     * Export data to downstream tools

