# WIN GO 30S DATA COLLECTION + HISTORICAL ANALYTICS PLATFORM

A production-quality system for continuously collecting historical data from the WinGo 30S JSON endpoint, storing the complete historical dataset in PostgreSQL, performing statistical/descriptive analysis and backtesting, and exposing verified latest historical results through a secure FastAPI API to a Chrome extension.

Designed for long-running operation on **Northflank**.

---

## Architecture Overview

```text
                    SOURCE API
                       |
                       v
              +------------------+
              | Python Collector | (24/7 Async Poller, Retry, Deduplication)
              +--------+---------+
                       |
                       v
              +------------------+
              | Data Validator   | (Field check, size verification 0-4=SMALL, 5-9=BIG)
              +--------+---------+
                       |
                       v
              +------------------+
              | Duplicate Guard  | (PostgreSQL UNIQUE issue_id ON CONFLICT)
              +--------+---------+
                       |
                       v
              +------------------+
              | PostgreSQL       | (Historical DB with audit trails)
              | Historical DB    |
              +--------+---------+
                       |
             +---------+---------+
             |                   |
             v                   v
      Analytics Engine      Data Quality / Monitoring
             | (Streaks, Frequencies, Transitions, Anomaly, Multi-Indicator Prediction)
             v
          FastAPI (v1 endpoints, Bearer API auth, CORS, TTL Cache)
             |
             v
      Chrome Extension (Manifest V3, Dark Glassmorphism Theme)
```

---

## Features

1. **Idempotent 24/7 Data Collection**:
   - Dynamic timestamp cache-busting parameter (`?ts=...`).
   - Network resilience with exponential backoff retries (`httpx`).
   - Transactional PostgreSQL writes using `INSERT ... ON CONFLICT (issue_id) DO UPDATE`.
   - Complete audit logging of raw HTTP payloads and source request metrics.
2. **Data Integrity & Auditability**:
   - `game_results` primary logical key is `issue_id`.
   - Never deletes or truncates historical data.
   - Separate `data_quality` table tracks gaps, anomalies, and parse errors.
3. **Statistical Analytics Engine**:
   - Small/Big frequency analysis across configurable windows (20, 50, 100, 500, 1000, all-time).
   - Streak analysis (current, longest, average lengths).
   - State transition matrix ($S \rightarrow S$, $S \rightarrow B$, $B \rightarrow S$, $B \rightarrow B$).
   - Statistical anomaly indicator (`NORMAL`, `WATCH`, `ANOMALY`).
   - Historical backtesting framework.
   - Multi-indicator weighted prediction engine (combines streak reversal, transition probability, frequency rebalance, momentum, and pattern matching).
4. **FastAPI Server**:
   - `/health` & `/health/detailed` monitoring endpoints.
   - Versioned REST API (`/api/v1/`).
   - Bearer API key authentication.
   - In-memory TTL response cache.
5. **Chrome Extension (Manifest V3)**:
   - Premium dark glassmorphism theme.
   - Clearly labels **ACTUAL OBSERVED RESULT** vs **STATISTICAL ANALYSIS**.
   - 10-second auto-refresh with staleness indicator (`● LIVE`, `● STALE`, `● OFFLINE`).

---

## Project Structure

```text
wingo-data-platform/
├── app/
│   ├── api/                 # FastAPI server, routes, dependencies
│   ├── collector/           # 24/7 async collector, parser, validator, deduplicator
│   ├── analytics/           # Frequency, streaks, transitions, rolling, anomaly, prediction
│   ├── models/              # SQLAlchemy 2.x declarative models (6 tables)
│   ├── services/            # Result, analytics, health, recovery, cache services
│   ├── core/                # Settings, database connection pool, structlog
│   └── database/            # Retention & maintenance scripts
├── migrations/              # Alembic migrations
├── extension/               # Chrome Extension Manifest V3
├── scripts/                 # Source inspector, backfill, health check, export, backup, icons
├── tests/                   # Pytest async test suite
├── Dockerfile               # Production Dockerfile
├── docker-compose.yml       # Full stack compose setup (PostgreSQL, Migrations, Collector, API)
├── northflank.json          # Northflank deployment blueprint
└── README.md
```

---

## Quick Start (Local Setup)

### 1. Requirements
- Python 3.12+ (or Docker)
- PostgreSQL 16+

### 2. Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment variables:

```env
APP_ENV=development
SOURCE_URL=https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json
POLL_INTERVAL_SECONDS=5
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/wingo_db
DATABASE_SYNC_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/wingo_db
API_KEY=your-secret-api-key
```

### 3. Run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will automatically:
1. Start PostgreSQL 16 container.
2. Run Alembic migrations.
3. Start the 24/7 Python Collector.
4. Start the FastAPI API server on `http://localhost:8000`.

---

## Inspecting the Source API

Run the API inspector script:

```bash
python scripts/inspect_source.py
```

---

## Database Migrations

Apply database schema changes:

```bash
alembic upgrade head
```

---

## Running Tests

Run the async test suite:

```bash
pytest tests/ -v
```

---

## Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`.
2. Turn on **Developer Mode**.
3. Click **Load Unpacked** and select the `extension/` directory.
4. Open the popup, enter API URL (`http://localhost:8000`) and your `API_KEY`, then click **Save**.

---

## Northflank Deployment Instructions

1. Push this repository to GitHub.
2. Log into Northflank dashboard.
3. Create a new PostgreSQL Addon.
4. Create two services:
   - **Collector**: Command `python -m app.collector`
   - **API**: Command `uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}`
5. Set environment variables from Northflank secrets (`DATABASE_URL`, `API_KEY`).

---

## Disclaimer & Principles

- **DATA INTEGRITY FIRST**: This platform stores complete historical data and performs statistical calculations.
- **STATISTICAL ANALYSIS**: Lottery draw outcomes are independent events. Historical statistical analysis does NOT guarantee future game results.
