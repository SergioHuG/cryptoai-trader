#!/usr/bin/env python3
"""
Phase-gate CI check (PROTOCOL.md §7 / DEC-16) — the un-forgeable wall.

Invoked by .github/workflows/phase-gate.yml after `scripts/test.sh` has
already been run fresh in both --pr-dir and --base-dir (two independent
Docker re-runs on the CI runner, not a stored/committed baseline — nothing
this check relies on is a file the agent's PAT could ever touch, even
indirectly).

Two layered checks (locked 2026-07-27):
  1. Per-task floor: PR coverage >= the task spec's dod.coverage_min_pct.
  2. Global ratchet: PR coverage and test count are both non-decreasing
     versus the integration branch (this PR must not make the whole suite
     worse, independent of whether the per-task floor was met). The
     integration branch is passed in via --base-dir/--branch context from
     the workflow — see phase-gate.yml for which branch that currently is.

Exit codes: 0 = gate passed, 1 = gate failed (see stderr for reasons).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_coverage_percent(dir_path: Path, *, required: bool = True) -> float | None:
    cov_file = dir_path / "coverage.json"
    if not cov_file.exists():
        if required:
            print(
                f"ERROR: {cov_file} not found — did scripts/test.sh run successfully in {dir_path}?",
                file=sys.stderr,
            )
            sys.exit(1)
        return None
    data = json.loads(cov_file.read_text())
    return float(data["totals"]["percent_covered"])


def load_test_count(dir_path: Path, *, required: bool = True) -> int | None:
    xml_file = dir_path / "report.xml"
    if not xml_file.exists():
        if required:
            print(
                f"ERROR: {xml_file} not found — did scripts/test.sh run successfully in {dir_path}?",
                file=sys.stderr,
            )
            sys.exit(1)
        return None
    root = ET.parse(xml_file).getroot()
    # pytest's --junit-xml emits either a bare <testsuite> or a <testsuites>
    # wrapper containing one or more <testsuite> children — handle both.
    if root.tag == "testsuites":
        return sum(int(ts.get("tests", 0)) for ts in root.findall("testsuite"))
    if root.tag == "testsuite":
        return int(root.get("tests", 0))
    print(f"ERROR: unrecognized JUnit XML root tag '{root.tag}' in {xml_file}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-dir", required=True, help="Path to the PR-branch checkout (post scripts/test.sh)")
    parser.add_argument("--base-dir", required=True, help="Path to the integration-branch checkout (post scripts/test.sh)")
    parser.add_argument("--branch", required=True, help="PR source branch name")
    parser.add_argument(
        "--allow-base-missing",
        action="store_true",
        help=(
            "If the base branch's coverage.json/report.xml are missing (e.g. this PR is "
            "itself the one introducing --cov/--junit-xml to scripts/test.sh, so the base "
            "branch predates coverage instrumentation), skip the ratchet check instead of "
            "failing. One-time bootstrapping flag — remove once the base branch has run "
            "the instrumented test.sh at least once."
        ),
    )
    args = parser.parse_args()

    pr_dir = Path(args.pr_dir).resolve()
    base_dir = Path(args.base_dir).resolve()

    # protocol_lib.py lives in the PR checkout's scripts/ dir — use the PR's
    # copy since that's the up-to-date version; spec files only ever live on
    # task branches, never on the integration branch.
    sys.path.insert(0, str(pr_dir / "scripts"))
    from protocol_lib import Violation, extract_task_id, load_spec, verify_branch_matches_spec  # noqa: E402

    # PR side is always required — a missing coverage.json/report.xml there is a real
    # problem (test.sh failed or was never run), never a legitimate bootstrapping case.
    pr_coverage = load_coverage_percent(pr_dir, required=True)
    pr_tests = load_test_count(pr_dir, required=True)

    base_coverage = load_coverage_percent(base_dir, required=not args.allow_base_missing)
    base_tests = load_test_count(base_dir, required=not args.allow_base_missing)

    print(f"PR branch ({args.branch}):")
    print(f"  coverage = {pr_coverage:.2f}%")
    print(f"  tests    = {pr_tests}")
    if base_coverage is None or base_tests is None:
        print(
            "base branch: coverage.json/report.xml not found — base branch predates "
            "coverage instrumentation (--allow-base-missing set). Ratchet check skipped "
            "for this run only."
        )
    else:
        print(f"base branch:")
        print(f"  coverage = {base_coverage:.2f}%")
        print(f"  tests    = {base_tests}")
    print()

    failures: list[str] = []

    task_id = extract_task_id(args.branch)
    if task_id:
        try:
            spec = load_spec(pr_dir, task_id)
            verify_branch_matches_spec(args.branch, spec)
            floor = float(spec.get("dod", {}).get("coverage_min_pct", 0))
            print(f"Task {task_id} declared coverage floor: {floor:.2f}%")
            if pr_coverage < floor:
                failures.append(
                    f"PR coverage {pr_coverage:.2f}% is below task {task_id}'s declared "
                    f"floor of {floor:.2f}% (spec dod.coverage_min_pct)."
                )
        except Violation as exc:
            failures.append(str(exc))
    else:
        print(
            f"Branch '{args.branch}' does not match the task-branch pattern "
            f"(feature/task-NN-*) — skipping per-task floor check."
        )

    if base_coverage is not None and base_tests is not None:
        if pr_coverage < base_coverage:
            failures.append(
                f"Coverage regression: PR is {pr_coverage:.2f}%, base branch is {base_coverage:.2f}% "
                f"(global non-regression ratchet, PROTOCOL.md §7)."
            )
        if pr_tests < base_tests:
            failures.append(
                f"Test-count regression: PR has {pr_tests} tests, base branch has {base_tests} "
                f"(global non-regression ratchet, PROTOCOL.md §7)."
            )

    print()
    if failures:
        print("PHASE GATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("Phase gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())