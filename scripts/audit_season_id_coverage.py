#!/usr/bin/env python3
"""
Audit season_id coverage after fixes to normalizers.py and db.py.
Validate that season_id is properly persisted and joins work.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from database.db import Database

def main():
    db = Database()
    
    # Get audit results
    audit_result = db.audit_season_id_coverage()
    
    print("\n" + "="*80)
    print("SEASON_ID COVERAGE AUDIT RESULTS")
    print("="*80)
    
    print("\n[SEASON_ID BASELINE]")
    for k, v in audit_result["season_id_audit"].items():
        print(f"  {k:.<40} {v}")
    
    print("\n[SEASONS AVAILABLE]")
    for k, v in audit_result["seasons_available"].items():
        print(f"  {k:.<40} {v}")
    
    print("\n[STATS ROWS]")
    for k, v in audit_result["stats_rows"].items():
        print(f"  {k:.<40} {v}")
    
    print("\n[JOIN VIABILITY]")
    for k, v in audit_result["join_viability"].items():
        if k == "interpretation":
            print(f"  {k:.<40}")
            print(f"    → {v}")
        else:
            print(f"  {k:.<40} {v}")
    
    print("\n[CRITICAL ISSUES]")
    issues = [i for i in audit_result["critical_issues"] if i is not None]
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"  ✅ No critical issues detected")
    
    print("\n" + "="*80)
    print("FULL JSON OUTPUT:")
    print("="*80)
    print(json.dumps(audit_result, indent=2))
    
    # Also show implications for feature coverage
    completed = audit_result["season_id_audit"]["completed_with_season_id"]
    joinable = audit_result["join_viability"]["matches_joinable_to_team_stats"]
    
    print("\n" + "="*80)
    print("FEATURE COVERAGE IMPLICATIONS")
    print("="*80)
    if completed == 0:
        print("🔴 CRITICAL: Zero season_id coverage in completed matches")
        print("   → All team/league stats features will fallback to 0.0")
        print("   → This explains goals_diff, xg_diff, shots_diff, possession_diff, draw_pct all=0.0")
    elif joinable == 0:
        print("🔴 CRITICAL: season_id exists but no team stats loaded")
        print("   → Joins fail silently, features fallback to 0.0")
        print("   → Fix: Run load_league_teams() and load_league_season() for each season")
    elif joinable < completed // 2:
        print(f"🟡 WARNING: Only {audit_result['join_viability']['joinable_rate']} of matches can join to stats")
        print("   → Extended features partially degraded")
    else:
        print(f"✅ GOOD: {audit_result['join_viability']['joinable_rate']} of matches can join to stats")
        print("   → Feature pipeline should be mostly healthy")
    
    db.close()

if __name__ == "__main__":
    main()
