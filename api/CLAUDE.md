# API Module — Claude Code Context

## Purpose
FastAPI backend. The communication layer between dashboard, agents, and database.

## Files
| File | Responsibility |
|---|---|
| `main.py` | App entrypoint, middleware, startup/shutdown events |
| `routes/` | Endpoint definitions by domain |

## Route Groups
| Route prefix | Responsibility |
|---|---|
| `/signals` | Signal history, active signals, signal status |
| `/approvals` | Human approval/rejection endpoint (validates token) |
| `/trades` | Trade history, open positions, paper vs live |
| `/performance` | P&L, drawdown, daily stats |
| `/system` | Health check, kill switch status, mode (paper/live) |

## Critical Rules
- Approval endpoint validates token expiry before any execution — reject expired tokens
- All endpoints return consistent error shapes: `{error: str, code: str}`
- Paper/live mode is determined by server startup flag — not a runtime API call
- WebSocket endpoint streams real-time signal updates to dashboard
- No authentication bypass — dashboard must present valid session token

## Environment Variables Required
```
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=  # For session tokens
```
