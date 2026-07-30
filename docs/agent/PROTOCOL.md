# Agent Protocol — CryptoAI Trader Delegation

**Status:** Locked. Source of truth for decisions: [OpenClaw Protocol — Decisions](https://app.notion.com/p/95d88ccfe54044e4a5d4f2843ba017e5) (Notion). This file is a transcription for in-repo, offline reference — if the two ever disagree, Notion wins and this file is stale.

**Scope:** governs how the OpenClaw builder agent executes Branch-2 (`cryptoai-trader`) module-build tasks. It does not restate domain knowledge (`CLAUDE.md`) or task specifics (the task spec) — see Three-File Split below.

---

## 1. Identity & Autonomy

**Execution-only agent (DEC-1).** No committed, design-locked spec → no work. The agent never designs, never picks between valid implementations, never adds a dependency, never reinterprets an ADR. If a spec is ambiguous or missing something it needs, that is a STOP, not a judgment call.

**Binary allow/stop autonomy (DEC-5).** The agent assumes Brooke is absent for the duration of the run. There is no waiting/ask state — the agent does not pause mid-task expecting a reply. A stop ends the run cleanly with a written blocker (see §5).

**No subagents (DEC-13).** The builder never spawns subagents. This preserves the red-first ordering guarantee and the one-task-in-flight invariant; parallel execution would make the two/three-commit structure a costume rather than a real constraint.

---

## 2. Three-File Split (DEC-4)

| File | Owns | Mutability |
|---|---|---|
| `AGENTS.md` (workspace bootstrap stub) | Protocol pointer only | Immutable by the agent |
| `docs/agent/PROTOCOL.md` (this file) | The protocol itself | Edited only by Brooke, out-of-band |
| `CLAUDE.md` (repo root, per-module as needed) | Domain knowledge | Read, never edited by the agent |
| Task spec (`docs/agent/specs/task-NN.md`) | Task-specific contract | Read, never edited by the agent |

No restatement across files — each fact lives in exactly one place.

**Resolved 2026-07-27 (DEC-19):** OpenClaw's bootstrap injection reads a **fixed six-file allowlist** (`AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`) into context automatically. `CLAUDE.md` is **not** in that list — confirmed structurally (absent from `injectedWorkspaceFiles` in the run report) and behaviorally (a canary token placed only in `CLAUDE.md` was not reproducible without an explicit file read). **Action:** `AGENTS.md` must instruct the agent to read `./CLAUDE.md` explicitly before starting any task. This is Option 1 from the canary findings — it preserves the three-file separation of concerns at the cost of one explicit line, rather than folding domain content into an auto-injected file like `TOOLS.md`.

---

## 3. Task Scope

**One module per task; design-time split points (DEC-2).** A task ends at Docker-green, full coverage, one commit set (see §4). Oversized modules (e.g. `engine.py`) get spec-declared split points decided at design time — the agent never decides how to split a module mid-task.

**Spec-per-task with design-time numeric oracles (DEC-3) — ★ KEYSTONE.** Every task spec is self-contained and includes:
- identity (what module, what file)
- contract (verbatim function/class signatures)
- behavior (what it must do, in prose)
- numeric oracles, fixed at design time (exact expected values/thresholds — not "reasonable," not agent-derived)
- test manifest (machine-readable; see §6)
- Definition of Done
- explicit out-of-scope list
- escalation triggers (conditions that are an automatic STOP)

Correctness rides entirely on the oracles being right. A thin spec is the single point of failure in this whole protocol — treat spec-writing as the highest-leverage, highest-care step in the pipeline.

---

## 4. Commit Shape — Three-Commit TDD

**Two-commit frozen-test TDD (DEC-7), amended to three commits (DEC-9 amendment, 2026-07-26):**

1. **Commit 1 — tests only.** Red for the right reason (missing implementation, not a broken test). Verify red before committing.
2. **Commit 2 — implementation only.** May not touch a test file. If a test *must* change during commit 2, that is a STOP, not an edit — the spec or the test was wrong, and that's a systemic-correction case (§8), not a quick fix.
3. **Commit 3 — `RUN-REPORT.md`.** Sha-pinned run report emitted by the self-verification gate (§7). Added this session (DEC-16) as part of the loop-engineering supplement; decisions 2 and 7's tests↔implementation separation is otherwise **unchanged**.

---

## 5. Stalls & Blockers (DEC-8)

A stall (spec ambiguity, escalation trigger hit, unresolvable red) produces:
- a WIP commit of whatever work exists, and
- a committed `*.BLOCKED.md` file describing the blocker,

both pushed to the task branch. An open blocker **freezes the branch** — no further agent work on it — until a desktop session resolves the blocker in the same commit that fixes the spec.

---

## 6. Mechanical Enforcement (DEC-6)

Two layers, deliberately minimal — everything not listed here is instruction-only (i.e., relies on the agent following this document, not on a machine check):
- **Branch protection on the integration branch** — no direct pushes, no agent merge/rebase (see §9). "The integration branch" is not always `main` — see the integration-branch note in §9.
- **Two pre-commit hooks** rejecting (a) coverage-gaming (`pragma`/`skip`/`xfail`) and (b) dependency drift (`pyproject.toml` / `poetry.lock` mismatches).

**Install note — per clone, not automatic.** `core.hooksPath` is local git config; it does not travel with the repo via any committed file, so it must be set explicitly in **every** clone that should run the pre-push gate (§7) — including the builder agent's separate clone under `~/agents/cryptoai/cryptoai-trader/` (DEC-12) once it exists, not just Brooke's primary dev clone:

```bash
chmod +x scripts/hooks/pre-push
git config core.hooksPath scripts/hooks
```

Verify with `git config core.hooksPath` (expect `scripts/hooks`) and `ls -la scripts/hooks/pre-push` (expect `-rwxr-xr-x`). A clone missing this step silently skips the gate entirely — it fails open, not closed.

---

## 7. Self-Verification Gate (DEC-16) — Loop-Engineering Supplement, Part 1

Added 2026-07-26 as a bounded, additive supplement (DEC-15 — see §10). Two tiers:

- **Local pre-push hook** (`scripts/hooks/pre-push`, installed via `core.hooksPath`). Self-verifies commit shape, test-manifest match, and the Decision-6 checks. Emits the sha-pinned `RUN-REPORT.md` as commit 3. **Deter-not-prevent** — the agent could bypass with `--no-verify`; the hook is a tripwire, not a wall.
- **Phone / per-task review** reads the `RUN-REPORT.md`'s git facts and judges only whether red was red for the right reason. It does not re-derive coverage or counts.
- **Desktop / per-phase review — the actual wall.** A committed, agent-immutable required GitHub Actions check at PR-to-integration-branch (agent-immutable because the fine-grained PAT is missing the `workflow` scope — the agent literally cannot edit the workflow file) owns the hard numbers (coverage %, test count) via a full Docker re-run. This is the one un-forgeable checkpoint in the whole pipeline.

**Status (2026-07-27): partially live — see integration-branch note in §9 before trusting this section at face value.** `.github/workflows/phase-gate.yml` is committed and currently triggers on PRs to `feature/phase2-backtest` (not `main` — `main` lags Branch 2 intentionally; see §9). `main`'s branch protection was configured first, before this mismatch was caught mid-session, with `"Phase Gate"` as a required status check (`strict: true`), force-pushes and deletions blocked, `enforce_admins: false`, `required_approving_review_count: 0`. That configuration is harmless but currently guards a branch nothing merges into yet — the equivalent protection has **not** been applied to `feature/phase2-backtest`, deliberately, because doing so would block Brooke's own ongoing direct pushes there before the Task-7 pilot begins. The coverage/count comparison itself runs against a fresh Docker re-run of both branches on every check — never a stored baseline file, so nothing the agent's PAT could touch (even indirectly) affects the ratchet, regardless of which branch is "base."

Commit shape itself is deterred (hook) + human-eyeballed at phase review, not mechanically re-validated per-commit — re-segmenting history per task was judged not worth the added complexity versus a human catching a misplaced test file in a diff.

**Resolved 2026-07-27 (DEC-20):** `/goal` is a real, session-scoped slash command (`/goal start|edit|pause|resume|complete|block|clear`) backed by model-facing tools (`get_goal`, `create_goal`, `update_goal`, `update_plan`) confirmed present in the tool schema. It is **not** a CLI-level dispatch mechanism — `openclaw goal` as a standalone command does not exist; `/goal` only operates inside a running session. **This does not reopen automation.** The agent may use its own session's goal state to track progress on the assigned task, but each session remains a single bounded dispatch — no auto-resumption, no cross-session self-triggering. This is a third bucket, distinct from both "literal automation primitive" and "conceptual-only": a real mechanism, scoped to bookkeeping within one already-bounded run.

---

## 8. Systemic-Correction Rule (DEC-17) — Loop-Engineering Supplement, Part 2

A committed STOP **or** a phase-gate semantic finding (i.e., something the desktop reviewer catches, not the local hook) triggers a "fix the defect class, not the delta" pass:
- resolved **spec-first** (fix the spec/task definition, then re-derive the fix),
- logged in the unblocking commit.

**Explicitly excluded:** honest-mistake hook catches (the hook working as intended — that's not systemic, that's the safety net doing its job) and ordinary red→green TDD churn (that's just TDD, not a defect class).

---

## 9. Repo & Origin Discipline (DEC-10, DEC-12)

**Origin-authoritative; one task in flight; no agent merge/rebase (DEC-10).** Every agent run: fetch → clean-check → fast-forward-or-STOP → verify the spec at HEAD and confirm no open blocker. Exactly one task in flight globally, across the whole delegation system — never two branches being worked in parallel. The agent never merges or rebases.

**Dedicated builder agent; repo nested in workspace (DEC-12).** Builder agent workspace at `~/agents/cryptoai/`, with `cryptoai-trader/` cloned inside it. This structurally protects the clean-tree invariant (the agent's home isn't the repo root) and unlocks a per-agent skill allowlist scoped narrowly to this workspace.

**Integration-branch note (2026-07-27) — read this before trusting any "main" reference elsewhere in this file at face value.** `main` lags `feature/phase2-backtest` intentionally: Branch 2 development (Tasks 1–5, plus the Step-0 infra commit) is active on `feature/phase2-backtest` and has not merged to `main` yet. Every reference in this document to "PR-to-`main`" or "`main` branch protection" describes the protocol's steady-state design, not today's literal target — **the actual current integration branch is `feature/phase2-backtest`.** `.github/workflows/phase-gate.yml` was retargeted accordingly. Two deferred actions follow from this:

1. **Enable branch protection on `feature/phase2-backtest`** (mirroring what's already configured on `main` — required `"Phase Gate"` check, no force-push, no deletion) — but only once ready to hand that branch to the agent, i.e. right before the Task-7 pilot begins, **not now**. Doing it now would block Brooke's own ongoing direct pushes to that branch, since `required_pull_request_reviews` (even at 0 approvals) inherently requires changes go through a PR.
2. **When Branch 2 completes and merges to `main`:** update `phase-gate.yml`'s trigger and base-checkout `ref:` from `feature/phase2-backtest` to `main` (both are marked with `# UPDATE TO main` comments in the file), and move branch protection enforcement from `feature/phase2-backtest` to `main`. At that point every "main" reference in this document becomes literal again.

**Deferred action — bump branch protection once the agent's PAT exists.** Whichever branch is the live integration target (see note above) should have `required_approving_review_count: 0` set initially, since Brooke is the only collaborator with write access and a review requirement would only gate her own merges. Once the builder agent's fine-grained PAT (this section, DEC-12) is issued and the agent has push access, **bump `required_approving_review_count` to `1`** on that branch — at that point a required review becomes the actual mechanism preventing agent self-merge, not just a formality. Until this bump happens, DEC-10's "no agent merge/rebase" is enforced only by the agent's PAT lacking merge rights (if configured that way) and by instruction, not by branch protection itself.

---

## 10. Loop-Engineering Adoption Scope (DEC-15)

Adopted as **Option B — bounded, additive supplement.** All 14 originally-locked decisions are untouched (DEC-9's three-commit amendment is the one explicit, narrow exception — see §4). Two new mechanisms were added additively: the self-verification gate (§7) and the systemic-correction rule (§8).

**Explicitly rejected:** the automation pillars of the source loop-engineering guide — `/loop`, `/schedule`, proactive/time-based orchestration, and sub-agent fan-out. These contradict the product's deliberate human-in-the-loop identity and decisions 5 and 13 above. Nothing in this protocol authorizes the agent to trigger its own future runs.

---

## 11. Review Tiers (DEC-9)

- **Phone, per task (structural):** commit shape (tests / impl / report) is correct; counts and coverage are present; red evidence in `RUN-REPORT.md` looks legitimate; test manifest matches the spec. Reads facts, does not re-derive them.
- **Desktop, per phase (semantic), at PR-to-integration-branch:** the actual correctness review, plus the required CI re-run (§7) that owns the hard numbers. See §9's integration-branch note for which branch that currently is.

---

## 12. Kickoff & Status (DEC-11)

Kickoff for a task is one line: `Run task-NN per AGENTS.md.` Brooke owns `QUEUE.md` status updates — the agent reports what happened, but never writes its own status into `QUEUE.md`.

---

## 13. Transport (DEC-14)

PinchChat is the driving/reviewing surface (phone). A dedicated Telegram bot pushes run-end notifications only. The messaging skill is an explicit, reasoned allowlist entry for the builder agent's workspace; the **Notion skill stays OFF** for the builder — a push notification is not the same permission as a workspace write, and this agent doesn't get the latter.

---

## 14. Pilot Plan (DEC-18)

**Resolved 2026-07-27: push-then-pilot-Task-7.** The stranded Task 6 (`storage.py`) commit, already built and Docker-verified on the Windows machine, gets pushed to origin directly (770 → 783) rather than rebuilt from spec by the agent as a rehearsal. The agent's first real run is **Task 7** (`engine.py`), genuine unbuilt work — not a known-good rehearsal target.

Rationale: push-then-pilot unblocks Branch 2 immediately regardless of pilot outcome, and avoids a two-implementations-of-`storage.py` reconciliation problem that rebuild-from-spec would create. The phase-gate CI (§7) — built this session — is the mechanical backstop that de-risks skipping a rehearsal-first approach; that backstop didn't exist when DEC-18 was first raised as open.

---

## Change Log

| Date | Change |
|---|---|
| 2026-07-26 | Base 14 decisions locked; loop-engineering supplement (DEC-15–17) adopted as Option B; DEC-9 amended to three-commit shape; DEC-18/19/20 opened. |
| 2026-07-27 | DEC-19 resolved (Option 1 — explicit `CLAUDE.md` read instruction). DEC-20 resolved (`/goal` is real but session-bounded; no automation reopened). DEC-18 resolved (push-then-pilot-Task-7). This file authored. Spec-format template built (`docs/agent/specs/_TEMPLATE.md`, strict manifest match). Pre-push hook built and smoke-tested (`scripts/hooks/pre-push` + `scripts/protocol_lib.py` + `scripts/hooks/lib/verify_manifest.py`). `scripts/test.sh` updated with `--cov`/`--junit-xml` output; duplicate pytest run and duplicate pip package removed. Phase-gate CI built and smoke-tested (`.github/workflows/phase-gate.yml` + `.github/scripts/phase_gate_check.py`), `main` branch protection configured live. **Mid-session correction:** discovered `main` lags `feature/phase2-backtest` intentionally (Branch 2 unmerged); retargeted `phase-gate.yml`'s trigger + base-checkout to `feature/phase2-backtest`, generalized `phase_gate_check.py`'s `--main-dir` to `--base-dir`, added the integration-branch note and its two deferred actions to §9. Branch protection on `feature/phase2-backtest` itself deliberately deferred until the Task-7 pilot begins. |