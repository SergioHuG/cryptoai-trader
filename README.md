# CryptoAI Trader

AI-driven cryptocurrency trading system with human-in-the-loop approval.

## How It Works
1. AI agents analyze market conditions in parallel (Technical, Sentiment, Fundamental)
2. LLM Synthesizer builds a human-readable trade recommendation
3. Risk gate validates against hardcoded limits
4. You receive a Telegram notification with Approve / Reject buttons
5. You review the full case in the dashboard (90 second window)
6. On approval, order executes via Kraken Pro API

## Current Phase
**Phase 0 — Foundation**

## Quick Start
```bash
bash scripts/setup.sh
```

## Development
```bash
# Start infrastructure
docker compose up -d postgres redis

# Run API
poetry run uvicorn api.main:app --reload

# Run tests
poetry run pytest

# Add a dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name
```

## Git Workflow
```bash
# Start a new feature
git checkout -b feature/component-name

# Commit
git commit -m "feat(scope): description"

# Never commit directly to main
```

## Architecture
See `docs/architecture.md` for full system design and ADRs.

## Safety
- Default mode is always **paper trading**
- Live trading requires explicit `--live` flag and Phase 2 validation
- Risk limits are hardcoded — not configurable at runtime
