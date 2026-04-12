# Execution Module — Claude Code Context

## Purpose
Order management and trade execution. This is the most safety-critical module.

## Files
| File | Responsibility | Status |
|---|---|---|
| `orders.py` | Order creation, cancellation, status tracking | Active |
| `paper.py` | Paper trading engine — simulates execution with real market data | Active |
| `live.py` | Live execution against real Kraken account | 🔒 OFF LIMITS UNTIL PHASE 3 |

## CRITICAL SAFETY RULES — READ BEFORE TOUCHING ANY FILE HERE
1. `live.py` is EMPTY and stays EMPTY until Phase 2 paper trading is validated
2. Default mode is ALWAYS paper trading — live requires explicit `--live` CLI flag
3. Every order MUST be pre-validated by `agents/risk.py` before reaching this module
4. No order is placed without a valid human approval token from the HITL interface
5. The daily drawdown kill switch in `agents/risk.py` is checked before every order

## Execution Flow
```
Human Approval (with token)
        ↓
Risk Gate validation (agents/risk.py)
        ↓
Mode check: paper or live?
        ↓
paper.py OR live.py (Phase 3 only)
        ↓
database/ trade log
```

## Testing Requirements
- `paper.py` must be fully tested before ANY live code is written
- Kill switch behavior must have dedicated tests: system halts, alert fires, no orders placed
- Order rejection paths must be tested as thoroughly as happy paths
