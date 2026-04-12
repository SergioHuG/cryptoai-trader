# Notifications Module — Claude Code Context

## Purpose
The HITL (Human in the Loop) interface. This module is the bridge between
the AI system and the human approver. It must be reliable, fast, and clear.

## Files
| File | Responsibility |
|---|---|
| `telegram.py` | Telegram Bot — push notifications and approval button handling |
| `formatter.py` | Formats signal data into a clear, scannable human-readable message |

## HITL Flow
```
Signal approved by risk gate
        ↓
formatter.py builds the message (symbol, direction, entry, stop, target, agent reasoning summary)
        ↓
telegram.py sends push notification with Approve / Reject buttons
        ↓
60-90 second countdown starts (SIGNAL_TIMEOUT_SECONDS)
        ↓
Human taps Approve → approval token generated → execution/
Human taps Reject → signal discarded → logged
Timeout → signal auto-cancelled → logged → human notified of cancellation
```

## Message Format Requirements
Every Telegram notification must include:
- 📊 Asset and direction (BTC/USD LONG)
- 💰 Entry price, stop loss, take profit
- ⚖️ Risk amount in $ and %
- 🤖 One-line summary of agent reasoning
- ⏱️ Time remaining to approve
- ✅ Approve button | ❌ Reject button

## Critical Rules
- Auto-cancel on timeout is NON-NEGOTIABLE — never assume approval
- Approval tokens are single-use and time-limited
- Every notification sends regardless of dashboard state — Telegram is the primary alert
- Emergency alerts (kill switch triggered) bypass any rate limiting
- Never send trade details to any chat except the authorized user chat ID

## Environment Variables Required
```
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```
