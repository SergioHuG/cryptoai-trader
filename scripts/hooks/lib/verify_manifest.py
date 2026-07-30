#!/usr/bin/env python3
"""
Self-verification gate helper for the CryptoAI Trader delegation protocol.
Implements PROTOCOL.md §7 (DEC-16): commit-shape check, strict manifest
match (locked 2026-07-27, both directions), DEC-6 redundant checks
(coverage-gaming patterns, dependency drift), and RUN-REPORT.md (commit 3)
emission.

Invoked by scripts/hooks/pre-push — not meant to be run standalone, though
it can be for debugging:
    python3 scripts/hooks/lib/verify_manifest.py \\
        --repo-root . --branch feature/task-07-engine \\
        --range-base origin/main --range-head HEAD

Exit codes:
    0 = push allowed
    1 = push blocked — either a real violation (fix required), or
        RUN-REPORT.md was just committed locally (retry `git push`)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# scripts/hooks/lib/verify_manifest.py -> parents[2] is scripts/, where the
# shared protocol_lib module lives (also used by .github/scripts/phase_gate_check.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from protocol_lib import (  # noqa: E402
    TASK_ID_RE,
    Violation,
    load_spec,
    verify_branch_matches_spec,
)

DEC6_PRAGMA_PATTERNS = [
    re.compile(r"pragma:\s*no\s*cover", re.IGNORECASE),
    re.compile(r"@pytest\.mark\.skip"),
    re.compile(r"@pytest\.mark\.xfail"),
]
TEST_FUNC_ADD_RE = re.compile(r"^\+def (test_\w+)\(", re.MULTILINE)
TEST_FUNC_DEL_RE = re.compile(r"^-def (test_\w+)\(", re.MULTILINE)


@dataclass
class CommitInfo:
    sha: str
    subject: str
    files_changed: list[str] = field(default_factory=list)
    diff_text: str = ""


def sh(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Violation(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def get_commit_range(repo_root: Path, base: str, head: str) -> list[str]:
    out = sh(repo_root, "rev-list", "--reverse", f"{base}..{head}")
    return [line for line in out.splitlines() if line]


def get_commit_info(repo_root: Path, sha: str) -> CommitInfo:
    subject = sh(repo_root, "log", "-1", "--format=%s", sha).strip()
    files = [
        f for f in sh(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).splitlines() if f
    ]
    diff_text = sh(repo_root, "show", "--format=", sha)
    return CommitInfo(sha=sha, subject=subject, files_changed=files, diff_text=diff_text)


def is_run_report_commit(commit: CommitInfo, task_id: str) -> bool:
    expected = f"docs/agent/reports/{task_id}-RUN-REPORT.md"
    return commit.files_changed == [expected]


def is_spec_only_commit(commit: CommitInfo, task_id: str) -> bool:
    expected = f"docs/agent/specs/{task_id}.md"
    return commit.files_changed == [expected]


def classify_commits(
    commits: list[CommitInfo], task_id: str
) -> tuple[CommitInfo, CommitInfo, CommitInfo | None]:
    # DEC-3 requires the spec to already be committed before the agent's
    # tests/impl work begins. On a brand-new task branch's first-ever push,
    # everything new since the merge-base is in range — including that spec
    # commit — so it legitimately sits ahead of the tests/impl/report triad
    # rather than being a violation. Strip at most one such leading commit
    # before applying the shape check to what remains.
    if commits and is_spec_only_commit(commits[0], task_id):
        commits = commits[1:]

    if len(commits) == 3 and is_run_report_commit(commits[2], task_id):
        return commits[0], commits[1], commits[2]
    if len(commits) == 2:
        return commits[0], commits[1], None
    raise Violation(
        f"Expected exactly 2 unreported commits (tests, impl) or 3 already-reported "
        f"(tests, impl, report) in the pushed range — optionally preceded by exactly "
        f"one spec-only commit (docs/agent/specs/{task_id}.md); found {len(commits)}. Per "
        f"PROTOCOL.md §4: one task = tests-commit + impl-commit + report-commit, "
        f"nothing else in the pushed range."
    )


def verify_test_commit(commit: CommitInfo, manifest: list[dict]) -> dict:
    manifest_by_file = {entry["file"]: set(entry["functions"]) for entry in manifest}
    manifest_files = set(manifest_by_file)

    touched = set(commit.files_changed)
    unexpected_files = touched - manifest_files
    if unexpected_files:
        raise Violation(
            f"Commit {commit.sha[:8]} (tests) touches file(s) not in test_manifest: "
            f"{sorted(unexpected_files)}. Manifest match is strict (locked 2026-07-27)."
        )

    added = set(TEST_FUNC_ADD_RE.findall(commit.diff_text))
    removed = set(TEST_FUNC_DEL_RE.findall(commit.diff_text))
    net_added = added - removed

    declared: set[str] = set()
    for funcs in manifest_by_file.values():
        declared |= funcs

    extra = net_added - declared
    missing = declared - net_added
    if extra or missing:
        parts = []
        if extra:
            parts.append(f"undeclared test function(s) added: {sorted(extra)}")
        if missing:
            parts.append(f"declared test function(s) missing from commit: {sorted(missing)}")
        raise Violation(
            f"Commit {commit.sha[:8]} (tests) fails strict manifest match — " + "; ".join(parts)
        )

    return {"files": sorted(touched), "functions": sorted(net_added)}


def verify_impl_commit(commit: CommitInfo, manifest: list[dict]) -> dict:
    manifest_files = {entry["file"] for entry in manifest}
    violating = set(commit.files_changed) & manifest_files
    if violating:
        raise Violation(
            f"Commit {commit.sha[:8]} (impl) touches test file(s) declared in the "
            f"manifest: {sorted(violating)}. Per PROTOCOL.md §4 (DEC-7): a test "
            f"needing to change during commit 2 is a STOP, not an edit."
        )
    return {"files": sorted(commit.files_changed)}


def verify_dec6(commits: list[CommitInfo]) -> None:
    """Redundant push-time check for the same concerns the pre-commit hooks enforce (DEC-6)."""
    for commit in commits:
        for pattern in DEC6_PRAGMA_PATTERNS:
            if pattern.search(commit.diff_text):
                raise Violation(
                    f"Commit {commit.sha[:8]} contains a coverage-gaming pattern "
                    f"matching '{pattern.pattern}'. Zero-pragma bar (PROTOCOL.md §6, "
                    f"DEC-6): genuinely unreachable branches are deleted, not pragma'd."
                )

    pyproject_changed = any("pyproject.toml" in c.files_changed for c in commits)
    lockfile_changed = any("poetry.lock" in c.files_changed for c in commits)
    if pyproject_changed != lockfile_changed:
        raise Violation(
            "pyproject.toml and poetry.lock changed independently in this push — "
            "dependency drift (PROTOCOL.md §6, DEC-6). An execution-only agent "
            "(DEC-1) should never add a dependency unilaterally; if this is "
            "legitimate, it needs a spec update and Brooke's sign-off first."
        )


def build_run_report(
    task_id: str,
    branch: str,
    test_commit: CommitInfo,
    impl_commit: CommitInfo,
    test_result: dict,
    impl_result: dict,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""# Run Report — {task_id}

**Branch:** `{branch}`
**Generated:** {now}
**Generated by:** `scripts/hooks/pre-push` (self-verification gate, PROTOCOL.md §7 / DEC-16)

This report is committed as commit 3 (PROTOCOL.md §4, DEC-9 amendment). It is
**deter-not-prevent evidence**, not a correctness judgment — commit shape,
manifest match, and DEC-6 checks all passed locally. Coverage and test-count
numbers are **not** verified here; they are owned exclusively by the phase-gate
CI at PR-to-`main` (§7), which is the actual un-forgeable wall.

## Commit 1 — Tests (sha-pinned)

- **sha:** `{test_commit.sha}`
- **subject:** {test_commit.subject}
- **files:** {', '.join(test_result['files'])}
- **test functions added:** {', '.join(test_result['functions'])}

> Phone-tier reviewer: judge here whether red was red for the right reason
> (missing implementation, not a broken test). This hook does not make that
> judgment — it only captures the diff for you to read.

## Commit 2 — Implementation (sha-pinned)

- **sha:** `{impl_commit.sha}`
- **subject:** {impl_commit.subject}
- **files:** {', '.join(impl_result['files'])}
- **test files touched:** none (verified)

## Checks Passed

- [x] Commit shape: tests-only then impl-only, in order
- [x] Manifest match: strict, both directions (test_manifest vs. commit 1 diff)
- [x] No test file touched in commit 2
- [x] DEC-6: no coverage-gaming pattern (`pragma: no cover` / `@pytest.mark.skip` / `@pytest.mark.xfail`) in either commit
- [x] DEC-6: no independent `pyproject.toml` / `poetry.lock` drift

## Not Verified Here (owned by phase-gate CI, §7)

- [ ] Coverage percentage
- [ ] Full Docker test-suite pass/fail
- [ ] Numeric oracle correctness
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--range-base", required=True)
    parser.add_argument("--range-head", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    branch = args.branch

    task_match = TASK_ID_RE.search(branch)
    if not task_match:
        # Not a task branch (e.g. main, or an ad-hoc branch) — nothing to verify.
        return 0
    task_id = task_match.group(1)

    try:
        spec = load_spec(repo_root, task_id)
        verify_branch_matches_spec(branch, spec)

        manifest = spec.get("test_manifest") or []
        if not manifest:
            raise Violation(
                f"Spec for {task_id} has an empty test_manifest — nothing to verify against."
            )

        shas = get_commit_range(repo_root, args.range_base, args.range_head)
        if not shas:
            return 0  # nothing new to push

        commits = [get_commit_info(repo_root, sha) for sha in shas]
        test_commit, impl_commit, report_commit = classify_commits(commits, task_id)

        if report_commit is not None:
            print(
                f"[pre-push] {task_id}: RUN-REPORT.md already present "
                f"(commit {report_commit.sha[:8]}). Push allowed."
            )
            return 0

        test_result = verify_test_commit(test_commit, manifest)
        impl_result = verify_impl_commit(impl_commit, manifest)
        verify_dec6(commits)

        report_text = build_run_report(
            task_id, branch, test_commit, impl_commit, test_result, impl_result
        )
        report_path = repo_root / "docs" / "agent" / "reports" / f"{task_id}-RUN-REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text)

        sh(repo_root, "add", str(report_path.relative_to(repo_root)))
        sh(repo_root, "commit", "-m", f"report: {task_id} run report")

        print(
            f"[pre-push] {task_id}: verification passed. RUN-REPORT.md committed as commit 3.\n"
            f"[pre-push] This push is now stale (missing commit 3) — run `git push` again "
            f"to include it."
        )
        return 1  # block this push; the retry hits the report_commit branch above and succeeds

    except Violation as exc:
        print(f"[pre-push] BLOCKED — {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())