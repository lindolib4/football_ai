#!/usr/bin/env python3
"""Load and display TOTO audit report without warnings."""

import json
import subprocess
import sys

# Run audit, capture stdout only
result = subprocess.run(
    [
        sys.executable,
        "scripts/audit_toto_quality_real.py",
    ],
    cwd="/FootAI",
    capture_output=True,
    text=True,
)

# Try to find JSON in output
output_lines = result.stdout.split("\n")
json_start = None
for i, line in enumerate(output_lines):
    if line.strip().startswith("{"):
        json_start = i
        break

if json_start is not None:
    json_str = "\n".join(output_lines[json_start:])
    try:
        report = json.loads(json_str)
        
        # Display key metrics
        print("TOTO Audit Summary (Post-Runtime Fixes):")
        print("=" * 90)
        print(f"Draws audited: {report.get('draws_total', 0)}")
        print(f"Consistency failures: {report.get('consistency_failures', 'unknown')}")
        print()
        
        summary = report.get("summary_aggregates", {})
        for key in ["off", "0.7", "0.9"]:
            if key in summary:
                s = summary[key]
                print(f"{key:>5}: matched_count={s.get('matched_count', 0):4d}, "
                      f"total_coupon_lines={s.get('total_coupon_lines', 0):4d}, "
                      f"coverage={s.get('coverage_pct', 0):.1f}%")
        
        print()
        print("Insurance coupon tracking:")
        insurance_counts = report.get("insurance_coupons_counts", {})
        for key, value in insurance_counts.items():
            print(f"  {key}: {value}")
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"First 500 chars of output:\n{output_lines[json_start][:500]}")
else:
    print("No JSON found in output")
    print("Last 20 lines of output:")
    for line in output_lines[-20:]:
        if line.strip():
            print(line)
