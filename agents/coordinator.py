"""
LLM Synthesizer Agent — agents/coordinator.py

Runs all signal agents in parallel via asyncio.
Waits for all agents to complete, then synthesizes their outputs
into a human-readable trade recommendation using the Anthropic API.

Output is passed to the risk gate before reaching the HITL interface.

Built in Phase 1 — feature/coordinator-agent branch.
"""
