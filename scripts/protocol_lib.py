"""
Shared helpers for the CryptoAI Trader delegation protocol tooling.

Used by both:
  - scripts/hooks/lib/verify_manifest.py (the pre-push self-verification gate, §7)
  - .github/scripts/phase_gate_check.py (the phase-gate CI check, §7)

so the task-id-extraction / spec-loading / branch-verification logic exists in
exactly one place rather than being duplicated and risking drift between the
two enforcement layers.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

TASK_ID_RE = re.compile(r"feature/(task-\d+)-")


class Violation(Exception):
    """Any hard-fail condition. Message is shown to the agent/CI verbatim."""


def extract_task_id(branch: str) -> str | None:
    match = TASK_ID_RE.search(branch)
    return match.group(1) if match else None


def load_spec(repo_root: Path, task_id: str) -> dict:
    spec_path = repo_root / "docs" / "agent" / "specs" / f"{task_id}.md"
    if not spec_path.exists():
        raise Violation(
            f"No spec found at docs/agent/specs/{task_id}.md. Per PROTOCOL.md §1 "
            f"(DEC-1): no committed, design-locked spec => no work."
        )
    text = spec_path.read_text()
    text_wo_comment = re.sub(r"^<!--.*?-->\s*", "", text, flags=re.DOTALL)
    match = re.match(r"^---\n(.*?)\n---\n", text_wo_comment, re.DOTALL)
    if not match:
        raise Violation(f"Spec docs/agent/specs/{task_id}.md has no parseable YAML frontmatter.")
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise Violation(f"Spec docs/agent/specs/{task_id}.md frontmatter is not valid YAML: {exc}")
    if not isinstance(data, dict):
        raise Violation(f"Spec docs/agent/specs/{task_id}.md frontmatter did not parse to a mapping.")
    return data


def verify_branch_matches_spec(branch_name: str, spec: dict) -> None:
    spec_branch = spec.get("branch")
    spec_task_id = spec.get("task_id")
    if spec_branch and spec_branch != branch_name:
        raise Violation(
            f"Branch mismatch: current branch is '{branch_name}' but the spec declares "
            f"branch '{spec_branch}' (task_id '{spec_task_id}'). Wrong-branch STOP."
        )
