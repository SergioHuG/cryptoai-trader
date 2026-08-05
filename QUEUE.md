# QUEUE.md — Task Queue

Owned by Brooke. The agent never writes to this file (PROTOCOL.md §12).

One task in flight globally at any time (PROTOCOL.md §9).

| Task | Module | Spec | Status |
|---|---|---|---|
| task-07 | `backtest/engine.py` | _not yet written_ | BLOCKED — spec pending |
| task-08 | `backtest/__init__.py` | _not yet written_ | QUEUED |

## Status values

- `QUEUED` — not started, no spec required yet
- `SPEC READY` — spec committed, agent may be dispatched
- `IN FLIGHT` — agent run active
- `BLOCKED` — open `*.BLOCKED.md` on the task branch; branch frozen
- `DONE` — merged to the integration branch
