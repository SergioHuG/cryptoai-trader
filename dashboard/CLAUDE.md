# Dashboard Module — Claude Code Context

## Purpose
React-based approval interface. Opened from Telegram notification link.
Gives the human full context (charts, agent reasoning, risk metrics) in the 60-90 second window.

## Tech Stack
- React (Vite)
- TailwindCSS for styling
- Lightweight charting library (Lightweight Charts by TradingView)
- Communicates with `api/` via REST + WebSocket

## Dashboard Panels (approval screen)
1. **Price chart** — 15min candles with EMA 9/21 overlaid, signal entry marked
2. **Signal summary** — direction, entry, stop loss, take profit, R/R ratio
3. **Risk panel** — $ at risk, % of capital, current open positions, daily P&L
4. **Agent reasoning** — collapsible cards per agent (Technical / Sentiment / Fundamental)
5. **Countdown timer** — visible, urgent when under 30 seconds
6. **Approve / Reject buttons** — large, clearly differentiated, require single tap

## Critical Rules
- Dashboard is READ-ONLY for market data — no trading actions except Approve/Reject
- Approve/Reject calls the API which validates the approval token — UI cannot bypass risk gate
- Mobile-first design — you will approve trades on your phone
- Dashboard must load and be interactive within 5 seconds of opening
- Countdown timer is always visible and accurate
