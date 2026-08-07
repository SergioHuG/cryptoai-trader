# CryptoAI Trader — Claude Code Context

## Project Identity
AI-driven cryptocurrency trading system with human-in-the-loop approval.
The human (final approver) reviews every trade signal before execution.
Claude Code is a development partner — not an autonomous decision maker.

## Current Phase
**Phase 2 — Backtest build.**
The AFML statistical strategy-evaluation layer (`backtest/`) is under active construction.
("Phase 2" here = "Branch 2" in PROTOCOL.md = git branch `feature/phase2-backtest` — one stream, three names.)
Module design and contracts live in `backtest/CLAUDE.md`; the governing build protocol is
`docs/agent/PROTOCOL.md`. Live task and branch status lives in Notion, not in this file.
Ladder ahead: Backtest → Paper → Live.

---

## Architecture Overview
```
Kraken Data Feed (CCXT)
        ↓
  Async Event Bus (asyncio)
        ↓
 Signal Agents (parallel)
 - Technical (EMA/RSI/Volume)
 - Sentiment
 - Fundamental
        ↓
 LLM Synthesizer Agent
 (builds human-readable case)
        ↓
 Risk Management Gate ← HARDCODED LIMITS, NEVER BYPASSED
        ↓
 Telegram Push Notification
        ↓
 React Dashboard (charts + reasoning)
        ↓
 Human: Approve / Reject (60-90 second window)
        ↓
 Execution Agent (Kraken API)
        ↓
 Trade Logger (PostgreSQL)
```

---

## Hardcoded Risk Constants
These values live in `agents/risk.py` as constants.
**Never modify these without explicit human instruction. Never make them agent-configurable.**

```python
MAX_RISK_PER_TRADE_PCT = 0.01       # 1% of total capital per trade
MAX_CONCURRENT_POSITIONS = 3         # Maximum open positions at once
DAILY_DRAWDOWN_KILL_SWITCH_PCT = 0.05  # 5% daily loss → full system halt
SIGNAL_TIMEOUT_SECONDS = 90         # Auto-cancel if no human response
```

---

## Trading Parameters
- **Exchange:** Kraken Pro (via CCXT library)
- **Assets:** BTC/USD, ETH/USD
- **Timeframe:** 15-minute candles for live signals (intraday); 1-hour for historical backtest (ADR-003)
- **Backtest layer:** scores per-event bet returns, not candles — it does **not** replay this
  strategy candle-by-candle (EMA-replay design retired, ADR-029). See `backtest/CLAUDE.md`.
- **Strategy (live signals):** EMA 9/21 crossover + RSI confirmation + volume filter, 1:2 R/R
- **Default mode:** Paper trading — live requires explicit `--live` flag

---

## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Dependency management | Poetry (never use pip directly) |
| API framework | FastAPI |
| Exchange integration | CCXT |
| Database | PostgreSQL (SQLAlchemy + Alembic) |
| Cache | Redis |
| Event bus | asyncio + queues |
| Frontend | React |
| Notifications | Telegram Bot API |
| Secrets (local) | `.env` via python-dotenv |
| Secrets (production) | AWS Secrets Manager |
| Containers | Docker + Docker Compose |
| Testing | pytest + pytest-cov |

---

## Hard Rules — Follow Every Session Without Exception

### Code Rules
- Always use Poetry commands: `poetry add`, `poetry run`, `poetry install`
- Never use `pip install` directly
- Never hardcode credentials — always use `os.getenv()`
- Every new function gets a corresponding test in `/tests`
- Risk constants in `agents/risk.py` are never changed without explicit instruction
- All database changes go through Alembic migrations, never raw schema edits

### Architecture Rules
- `execution/live.py` is OFF LIMITS — file stays empty until the Live stage
- Paper trading is always the default execution mode
- All trade execution MUST pass through `agents/risk.py` gate first — no exceptions
- Docker Compose is the only way to run the full stack locally

### Git Rules
- Never commit directly to `main`
- Always work on a feature branch
- Commit message format: `type(scope): description`
  - Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
  - Examples: `feat(agents): add EMA signal calculator`
  - Examples: `test(risk): add drawdown kill switch tests`
- Never commit `.env` files — only `.env.example`

### Safety Rules
- Paper trading mode is always default — live requires explicit `--live` flag
- No execution code runs without passing through risk gate first
- Daily drawdown kill switch must have passing tests before any execution code is written
- Signal timeout auto-cancel is enforced — never assume human approval

---

## Validation Ladder
The strategy advances through three stages, in order:
- **Backtest** (current stage): statistical evaluation on 6–12 months of Kraken historical data
- **Paper:** paper-trade 4 weeks minimum with the full stack running
- **Live:** live trading only after Paper shows consistent positive expectancy

---

## Folder Structure
```
cryptoai-trader/
├── CLAUDE.md                    ← You are here (loaded every session)
├── agents/CLAUDE.md             ← Agent-specific context
├── data/CLAUDE.md               ← Data pipeline context
├── execution/CLAUDE.md          ← Execution safety rules
├── backtest/CLAUDE.md           ← Backtesting context
├── notifications/CLAUDE.md      ← HITL notification context
├── dashboard/CLAUDE.md          ← Frontend context
├── database/CLAUDE.md           ← Database context
├── api/CLAUDE.md                ← API context
├── docs/                        ← Architecture decisions (ADRs)
└── tests/                       ← Mirrors src structure
```

---

## Communication Style
- Be explicit about which phase/module you are working on
- Flag any deviation from the architecture before implementing it
- When in doubt about risk-related code, ask before writing
- Always confirm before touching `execution/` folder contents
