# Agents Module — Claude Code Context

## Purpose
This module contains all AI agents. Each agent has a single responsibility.
The coordinator synthesizes their outputs into a human-readable trade case.

## Agent Responsibilities
| Agent | File | Responsibility |
|---|---|---|
| Coordinator | `coordinator.py` | LLM synthesizer — combines all signals into trade recommendation |
| Technical | `technical.py` | EMA 9/21 crossover, RSI, volume filter calculations |
| Sentiment | `sentiment.py` | Sentiment analysis from news/social sources |
| Fundamental | `fundamental.py` | On-chain data, market cap trends, ecosystem health |
| Risk Gate | `risk.py` | HARDCODED limits enforcement — final gate before execution |

## Critical Rules for This Module
- `risk.py` constants are NEVER modified without explicit human instruction
- The risk gate is NEVER bypassed — every signal passes through it
- Agents run in PARALLEL via asyncio — coordinator waits for all before synthesizing
- Agents return structured Pydantic models, never raw dicts
- Each agent must handle its own exceptions — one failing agent does not kill the pipeline

## Risk Constants (defined here, referenced everywhere)
```python
MAX_RISK_PER_TRADE_PCT = 0.01
MAX_CONCURRENT_POSITIONS = 3
DAILY_DRAWDOWN_KILL_SWITCH_PCT = 0.05
SIGNAL_TIMEOUT_SECONDS = 90
```

## Signal Flow
```
technical.py ──┐
sentiment.py ──┼──→ coordinator.py → risk.py → notifications/
fundamental.py ┘
```

## Testing Requirements
- `risk.py` requires 100% test coverage before any other agent is built
- Every signal calculation in `technical.py` must have unit tests with known inputs/outputs
- Coordinator must have integration tests covering approve and reject paths
