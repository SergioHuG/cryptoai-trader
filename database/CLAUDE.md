# Database Module — Claude Code Context

## Purpose
PostgreSQL data layer. All persistent storage for trades, signals, candles, and performance.

## Files
| File | Responsibility |
|---|---|
| `models.py` | SQLAlchemy ORM models |
| `schemas.py` | Pydantic schemas for API serialization |
| `migrations/` | Alembic migration files — never edit schema directly |

## Core Tables
| Table | Purpose |
|---|---|
| `candles` | Historical OHLCV data per symbol/timeframe |
| `signals` | Every generated signal with full agent reasoning JSON |
| `approvals` | Human approval/rejection record with timestamp |
| `trades` | Executed trades (paper and live) with full lifecycle |
| `performance` | Daily P&L, drawdown tracking, kill switch events |

## Critical Rules
- NEVER modify the database schema directly — always use Alembic migrations
- Every schema change: `alembic revision --autogenerate -m "description"`
- All DB access goes through SQLAlchemy models — no raw SQL except in migrations
- Decimals for all price and quantity fields — never floats (floating point = money bugs)
- Every trade record is immutable once written — append-only for audit trail

## Environment Variables Required
```
DATABASE_URL=postgresql://user:password@localhost:5432/cryptoai_trader
```

## Testing Requirements
- Use a separate test database — never run tests against the development database
- Fixtures provide known data states for each test
- Migration tests verify schema forward and rollback correctly
