#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
REPORT_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_[a-zA-Z0-9_-]+\.md$")
SECTION_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)

REQUIRED_SECTIONS = [
    "## 1. Summary",
    "## 2. Goal",
    "## 3. Scope",
    "## 4. Files Changed",
    "## 5. Detailed Changes",
    "## 6. Public Contracts / Interfaces",
    "## 7. Database Changes",
    "## 8. API Behavior",
    "## 9. Business Logic Changes",
    "## 10. Tests",
    "## 11. Manual Verification",
    "## 12. Risks / Known Issues",
    "## 13. Deviations from Spec",
    "## 14. Next Recommended Step",
    "## 15. Commit / PR",
    "## 16. Critical Review",
    "## 17. Что не сделано из запланированного",
]

HIGH_IMPACT_PREFIXES = (
    "core/",
    "toto/",
    "database/",
    "core/",
    "toto/",
    "database/",
)


@dataclass
class ChangedFile:
    status: str
    path: str


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def parse_name_status(output: str) -> list[ChangedFile]:
    changed: list[ChangedFile] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0].strip()
        path = parts[-1].strip()
        changed.append(ChangedFile(status=status, path=path))
    return changed


def get_changed_files(mode: str, base: str | None, head: str | None) -> list[ChangedFile]:
    if mode == "staged":
        output = run_git(["diff", "--name-status", "--cached"])
        return parse_name_status(output)

    if mode == "range":
        if not base or not head:
            raise ValueError("--base and --head are required for range mode")
        output = run_git(["diff", "--name-status", f"{base}...{head}"])
        return parse_name_status(output)

    raise ValueError(f"Unsupported mode: {mode}")


def extract_sections(text: str) -> list[tuple[str, int, int]]:
    """Return list of (heading, start_index, end_index)."""
    matches = list(re.finditer(r"^##\s+\d+\.\s+.+$", text, flags=re.MULTILINE))
    sections: list[tuple[str, int, int]] = []
    for idx, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((heading, start, end))
    return sections


def validate_report_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Report file does not exist: {path.relative_to(REPO_ROOT)}")
        return errors

    if not REPORT_NAME_RE.match(path.name):
        errors.append(
            f"Invalid report filename '{path.name}'. Required format: YYYY-MM-DD_HH-MM_name.md"
        )

    text = path.read_text(encoding="utf-8")

    for heading in REQUIRED_SECTIONS:
        if heading not in text:
            errors.append(f"Missing required section heading: {heading}")

    sections = extract_sections(text)
    heading_to_body: dict[str, str] = {}
    for heading, start, end in sections:
        heading_to_body[heading] = text[start:end].strip()

    for heading in REQUIRED_SECTIONS:
        body = heading_to_body.get(heading)
        if body is None:
            continue
        if not body.strip():
            errors.append(f"Empty section: {heading}")

    if re.search(r"\bнет\s+проблем\b", text, flags=re.IGNORECASE):
        errors.append("Forbidden phrase detected: 'нет проблем'. Be explicit about risks.")

    # Optional structural sanity: ensure sections are numbered 1..17 in order.
    section_numbers = [int(num) for num, _ in SECTION_HEADING_RE.findall(text)]
    if section_numbers and section_numbers != sorted(section_numbers):
        errors.append("Section numbering order is invalid.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate mandatory PR reports policy")
    parser.add_argument("--mode", choices=["staged", "range"], default="staged")
    parser.add_argument("--base", help="Base commit for range mode")
    parser.add_argument("--head", help="Head commit for range mode")
    args = parser.parse_args()

    try:
        changed = get_changed_files(mode=args.mode, base=args.base, head=args.head)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: unable to determine changed files: {exc}")
        return 2

    changed_paths = [c.path for c in changed]
    changed_non_reports = [p for p in changed_paths if not p.startswith("reports/")]
    added_reports = [c.path for c in changed if c.path.startswith("reports/") and c.status.startswith("A")]

    errors: list[str] = []

    if changed_paths and not added_reports:
        errors.append("PR must include at least one NEW file in reports/.")

    if changed_non_reports and not added_reports:
        errors.append("Changed files outside reports/ require a new report in reports/.")

    changed_high_impact = [p for p in changed_paths if p.startswith(HIGH_IMPACT_PREFIXES)]
    if changed_high_impact and not added_reports:
        errors.append(
            "Changes in core/toto/database detected; a report file in reports/ is mandatory."
        )

    for report_rel in added_reports:
        report_errors = validate_report_file(REPO_ROOT / report_rel)
        for err in report_errors:
            errors.append(f"{report_rel}: {err}")

    if errors:
        print("REPORT POLICY CHECK: FAILED")
        for err in errors:
            print(f" - {err}")
        return 1

    print("REPORT POLICY CHECK: PASSED")
    if added_reports:
        print("Validated report files:")
        for path in added_reports:
            print(f" - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
