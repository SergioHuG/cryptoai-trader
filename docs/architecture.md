# Architecture Overview

## System Purpose
AI-driven cryptocurrency trading system with human-in-the-loop approval.
Every trade signal is reviewed and approved by a human before execution.

---

## Key Decisions Made (ADRs)

### ADR-001 — Exchange: Kraken Pro
- **Decision:** Single exchange to start, Kraken Pro
- **Reason:** Already have account + API access, strong regulatory standing, excellent API
- **CCXT abstracts the integration** — adding future exchanges is a config change, not a rewrite
- **Future:** Stocks, futures, options via Kraken when system is validated

### ADR-002 — HITL Model: Final Approver
- **Decision:** Human approves every trade before execution
- **Reason:** Building trust in the system before any automation
- **Interface:** Telegram push → React dashboard → Approve/Reject
- **Timeout:** 90 seconds → auto-cancel (never assume approval)
- **Future:** Autonomous mode possible after system edge is validated

### ADR-003 — Timeframe: 15-minute Intraday
- **Decision:** 15-minute candles, intraday trading
- **Reason:** Scalping is incompatible with HITL (signals expire before human can respond)
- **Future:** Scalping mode possible as autonomous mode in Phase 3+

### ADR-004 — Stack: Python + Poetry + Docker
- **Decision:** Python 3.11+, Poetry for deps, Docker from day one
- **Reason:** Portable across AWS EC2 → Orange Pi/Ubuntu home server migration
- **Docker Compose** is the only way to run the full stack locally

### ADR-005 — Secrets: .env locally, AWS Secrets Manager on EC2
- **Decision:** python-dotenv for local, AWS Secrets Manager for production
- **Reason:** Code reads from `os.getenv()` — doesn't care about the source
- **Never:** Hardcoded credentials, committed `.env` files

### ADR-006 — Risk: Hardcoded Constants, Never Agent-Configurable
- **Decision:** Risk limits are constants in `agents/risk.py`, not settings
- **Reason:** Agents should never be able to negotiate around risk limits
- **Constants:** 1% per trade, 3 max positions, 5% daily drawdown kill switch

### ADR-007 — Validation: Backtest → Paper → Live
- **Decision:** Three mandatory phases before real money
- **Phase 1:** 6-12 months historical backtest
- **Phase 2:** 4 weeks paper trading with full stack
- **Phase 3:** Live only after Phase 2 shows positive expectancy

---

## Assets (Phase 1)
- BTC/USD
- ETH/USD

## Strategy (Phase 1 Baseline)
- EMA 9 crosses EMA 21
- Confirmed by RSI (not overbought/oversold)
- Volume above 20-period average
- Stop loss: below recent swing low
- Take profit: 1:2 risk/reward ratio
