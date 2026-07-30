<!--
  TASK SPEC TEMPLATE — CryptoAI Trader delegation protocol
  Governed by: docs/agent/PROTOCOL.md §3 (DEC-3, keystone) and §7 (self-verification gate manifest match)

  File naming convention: docs/agent/specs/task-NN.md
  Branch naming convention: feature/task-NN-<slug>
  The pre-push hook derives the expected spec path from the current branch name, then
  cross-checks the frontmatter's task_id/branch fields against reality — a mismatch is a STOP.

  Do not delete this comment block when instantiating a real spec; it documents the contract
  for humans reading the raw file. Everything below --- start-of-frontmatter is real content.
-->

---
task_id: task-NN
branch: feature/task-NN-slug
module: src/path/to/module.py
depends_on: []              # list of task_ids that must be merged to main first, e.g. [task-06]

test_manifest:
  # One entry per test file this task is allowed to create or modify.
  # `functions` is the exhaustive list of test function names commit 1 must add.
  # The pre-push hook diffs commit 1's actual test file(s) against this list:
  #   - any function present in the diff but absent here => STOP (undeclared test)
  #   - any function declared here but absent in the diff => STOP (incomplete commit 1)
  - file: tests/path/to/test_module.py
    functions:
      - test_example_behavior_one
      - test_example_behavior_two

dod:
  coverage_min_pct: 100       # zero-pragma bar; phase-gate CI is the authoritative check (§7)
  docker_command: "scripts/test.sh path/to/test_module.py"

out_of_scope:
  - "One bullet per explicitly excluded concern. Be specific — this is what stops scope creep,
     not a formality."

escalation_triggers:
  # Conditions that are an automatic STOP (§5), not a judgment call for the agent.
  - "Any ambiguity in a shared contract (e.g. SignalPacket) this module touches"
  - "A new third-party dependency would be required"
  - "A numeric oracle below cannot be satisfied without changing an already-locked ADR"
---

# Task NN: [Module Name]

## Identity

- **Module:** `src/path/to/module.py`
- **Task branch:** `feature/task-NN-slug`
- **Depends on:** [none | task-06, ...]

## Contract

Verbatim signatures the implementation must match exactly — no renaming, no signature drift,
no "improved" parameter order. If a signature here turns out to be wrong, that's a spec bug:
STOP, don't silently fix it in commit 2.

```python
def function_name(arg: Type) -> ReturnType:
    """One-line summary of contract, not implementation."""
    ...


class ClassName:
    def method_name(self, arg: Type) -> ReturnType:
        ...
```

## Behavior

Prose description of what the module must do. Written for a competent engineer with zero
context on this specific module but full context on the codebase's conventions (AFML
grounding, Decimal↔pandas seam via SignalPacket, fail-loud over silent fallback, etc. — see
`CLAUDE.md`, not restated here).

- Behavior point 1.
- Behavior point 2.
- Edge cases and how they must be handled (explicitly — "handle appropriately" is not
  acceptable here per the no-placeholders rule).

## Numeric Oracles

Design-time, fixed values — never agent-derived, never "reasonable." These become the
literal assertions in the commit-1 tests. If you can't produce a specific expected number
for a case, that's a design gap in the spec, not something to leave to the agent's judgment.

| Case | Input | Expected Output | Rationale / Source |
|---|---|---|---|
| Nominal case | `example_input_1` | `0.1234` | Hand-computed / reference implementation |
| Edge case: empty input | `[]` | raises `ValueError` | Fail-loud convention |
| Edge case: boundary | `example_input_2` | `1.0` | AFML §X.Y |

## Definition of Done

- [ ] Commit 1: test file(s) listed in `test_manifest` above, red for the right reason
      (`ImportError` / `AttributeError` on the not-yet-implemented target, not a logic bug
      in the test itself).
- [ ] Commit 2: implementation only, no test file touched. Docker green.
- [ ] Coverage ≥ `dod.coverage_min_pct` above (zero pragmas — genuinely unreachable branches
      are deleted, not pragma'd).
- [ ] `dod.docker_command` above passes clean.
- [ ] Commit 3: `RUN-REPORT.md` emitted by the pre-push hook.

## Out of Scope

See `out_of_scope` in frontmatter above — restated here in prose if useful for a human skim,
but the frontmatter list is authoritative for tooling.

## Escalation Triggers

See `escalation_triggers` in frontmatter above. Hitting any of these is an automatic STOP
per PROTOCOL.md §5 — commit the WIP state and a `*.BLOCKED.md`, do not attempt to reason
past it.