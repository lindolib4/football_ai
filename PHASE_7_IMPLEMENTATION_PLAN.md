# PHASE 7 IMPLEMENTATION PLAN
## Complete API Field Audit & Feature Pipeline Correction

**Project:** FootAI  
**Phase:** 7 — FootyStats/TotoBrief API Key Audit, Sentinel Value Handling, Feature Pipeline Correction  
**Date:** March 2026  
**Status:** PLANNING → IMPLEMENTATION  

---

## EXECUTIVE SUMMARY

**Audit Findings:**

The codebase has a **foundational data quality issue**: while sentinel values (-1, -2) are correctly normalized to `None` at the database layer, the **feature builder silently converts all `None` values to 0.0**, losing the distinction between:

- ✓ Real data (team actually has 10 shots)
- ✗ Missing/sentinel data (API returned -1, now None, treated as real 0)
- ✗ Default fallback (no source data, forced to use default)

This explains why `shots_diff`, `possession_diff`, `xg_diff` show **100% fallback_rate and zero_rate** — they're being filled with 0.0 values that represent missing data, not real statistics.

**Phase 7 Solution:** Implement **source tracking** throughout the pipeline to distinguish real vs. missing vs. fallback data, enabling honest feature quality reporting and allowing the model to make informed decisions about feature usage.

---

## AUDIT RESULTS SUMMARY

### Current API Integration (CORRECT)
✅ **Endpoints Used:**
- `/league-teams?include=stats` — Team season stats (ppg, shots, xg, possession, etc.)
- `/league-season` — League-level priors (draw%, home_advantage, avg_goals)
- `/league-matches` — Historical completed matches (for training)
- `/todays-matches` — Daily/live matches (for UI, not training)

✅ **Sentinel Value Normalization (Database Layer):**
- All -1, -2, "", None → normalized to None in database
- Validation done in `ingestion/validators.py` and `ingestion/normalizers.py`

### Critical Problem (INCORRECT)
❌ **Feature Builder Loss of Distinction:**
```python
# In core/features/builder.py _to_float():
def _to_float(val, key: str):
    if val is None:
        return 0.0  # ← PROBLEM: Loses distinction between missing and real zero
    if val in {-1, -2, -1.0, -2.0}:
        return 0.0  # ← PROBLEM: Treats sentinel as real zero
    # ... more processing
    return float(val)
```

Result: No way to tell if a feature value of 0.0 came from:
- Real data (team has 0 shots) — statistically rare
- Missing data (API returned -1, normalized to None) — common
- Default fallback (no team stats available) — increasingly common as historical data grows

### Diagram: Current Data Loss
```
API Response: team_shots = -1 (missing marker)
    ↓
normalize_missing(): -1 → None
    ↓
SQLite: NULL (correct)
    ↓
Feature Builder _to_float(None): None → 0.0  ← PROBLEM
    ↓
Feature Vector: shots_diff = 0.0 (indistinguishable from real zero)
    ↓
Model: Learns that 0.0 shots_diff is valid feature
    ↓
Quality Report: "100% fallback_rate" (but no source tracking)
```

---

## PRIMARY OBJECTIVES (PRIORITY 1)

### 1.1 Add Source Metadata to Feature Pipeline

Create a unified tracking system where each feature value carries metadata:

```python
@dataclass
class FeatureValue:
    """A feature value with source/quality metadata."""
    value: float
    source: str  # "real" | "missing_sentinel" | "fallback_default" | "computed"
    quality: str  # "strong" | "moderate" | "weak"
    confidence: float  # 0.0-1.0
    
    @property
    def is_real(self) -> bool:
        return self.source == "real"
    
    @property
    def is_fallback(self) -> bool:
        return self.source in ("missing_sentinel", "fallback_default")
```

Where to implement:
- `core/features/builder_v2.py` (new version, preserves backward compatibility)
- Modify `_to_float_with_source()` instead of `_to_float()`

### 1.2 Preserve Sentinel Information Through Pipeline

**In Database Layer:**
Add optional columns to `team_season_stats` to track data quality:
```sql
ALTER TABLE team_season_stats ADD COLUMN shots_avg_overall_source TEXT;  -- "real" | "missing" | "fallback"
ALTER TABLE team_season_stats ADD COLUMN shots_avg_overall_confidence REAL;  -- 0.0-1.0
-- (Repeat for all stats fields)
```

OR use JSON metadata in existing row:
```python
# Instead of:
{"shots_avg_overall": 10.5}

# Store:
{"shots_avg_overall": {
    "value": 10.5,
    "source": "real",
    "confidence": 1.0,
    "api_key": "shotsAVG_overall"
}}
```

### 1.3 Distinguish in Feature Builder

Modify feature building to NOT silently treat None as 0.0:

```python
# OLD (loses information):
def build_features(match_data):
    shots_diff = _to_float(home_stats.get("shots")) - _to_float(away_stats.get("shots"))
    return {"shots_diff": shots_diff}

# NEW (preserves source):
def build_features(match_data):
    home_shots_val, home_shots_src = _extract_with_source(home_stats, "shots")
    away_shots_val, away_shots_src = _extract_with_source(away_stats, "shots")
    
    shots_diff = home_shots_val - away_shots_val
    shots_diff_source = "real" if (home_shots_src == "real" and away_shots_src == "real") else "fallback"
    
    return {
        "shots_diff": shots_diff,
        "__source_shots_diff": shots_diff_source
    }
```

---

## SECONDARY OBJECTIVES (PRIORITY 2)

### 2.1 Create API Field Audit Report

Generate automated report showing:

```python
# api/field_auditor.py (new module)
class ApiFieldAuditor:
    """Audits API field coverage and sentinel usage."""
    
    def audit_footystats(self, limit: int = 5000) -> dict:
        """
        Returns:
        {
            "endpoints": {
                "league-teams": {
                    "fields": {
                        "shotsAVG_overall": {
                            "total_records": 5000,
                            "present": 4950,
                            "missing_sentinel": 50,
                            "coverage": 0.99,
                            "sentinel_markers": [-1, -2]
                        },
                        "xg_for_avg_overall": {
                            "total_records": 5000,
                            "present": 0,
                            "missing_sentinel": 5000,
                            "coverage": 0.0,
                            "note": "Field not provided by API for this season"
                        }
                    }
                },
                "league-season": {...},
                "league-matches": {...}
            },
            "summary": {
                "high_coverage_fields": [...],
                "weak_coverage_fields": [...],
                "missing_fields": [...],
                "recommendations": [...]
            }
        }
        """
        pass
```

Output saved to: `reports/api_field_audit_YYYYMMDD_HHMMSS.json`

### 2.2 Create Feature Quality Report v2

Enhanced version that shows real vs. missing vs. fallback:

```python
# scheduler/auto_train.py (enhanced)
def _build_feature_profile_v2(rows: list) -> dict:
    """
    Enhanced feature profiling with source tracking.
    
    Returns:
    {
        "feature_name": {
            "fill_rate": 0.95,
            "real_rate": 0.95,           # NEW: How many values from "real" source
            "missing_sentinel_rate": 0.04,  # NEW: How many from sentinel (-1/-2)
            "fallback_rate": 0.01,       # NEW: How many from fallback default
            "zero_rate": 0.02,
            "distinct_values": 47,
            "quality": "strong" | "moderate" | "weak",
            "recommendation": "use" | "caution" | "exclude"
        }
    }
    """
    pass
```

### 2.3 Document Expected vs. Actual Field Mapping

Create reference document: `API_FIELD_REFERENCE.md`

```markdown
## FootyStats Field Mapping Reference

### league-teams?include=stats

**Expected Fields (from docs):**
| API Field | Internal Name | Type | Sentinel | Coverage | Use Case |
|---|---|---|---|---|---|
| shotsAVG_overall | shots_avg_overall | float | -1, -2 | ~99% | shots_diff |
| shotsAVG_home | shots_avg_home | float | -1, -2 | ~99% | shots_diff (home bias) |
| possessionAVG_overall | possession_avg_overall | float | -1 | ~98% | possession_diff |
| xg_for_avg_overall | xg_for_avg_overall | float | MISSING | ~0% | xg_diff ← NOT PROVIDED BY API |
| | | | | | |

**Findings:**
- xG fields are NOT reliably provided by FootyStats API for team stats
- Consider alternative source or exclude from model
```

### 2.4 Create Decision Matrix for Weak Features

For each weak feature, document:

```markdown
## shots_diff Analysis

**Current Status:**
- Fill rate: 95%
- Real rate: 5%
- Missing sentinel rate: 94%
- Fallback rate: 1%

**Problem:**
- 94% of values come from -1/-2 sentinel (field not provided in response)
- Feature is being trained on 99% missing data markers, not real stats

**Root Cause:**
- FootyStats `/league-teams?include=stats` does NOT reliably provide shots stats
  for all teams/seasons
- API docs don't explicitly guarantee this field

**Solution:**
- Option A: Exclude shots_diff from model (prevent phantom feature)
- Option B: Find alternative source (league stats, fixtures data)
- Option C: Accept weak feature, mark as "low-confidence" in model

**Recommendation:** EXCLUDE shots_diff until reliable source found
```

---

## IMPLEMENTATION TASKS (PRIORITY 1)

### Task 1.1: Modify Feature Builder for Source Tracking
**File:** `core/features/builder.py` → `core/features/builder_with_source.py`

Changes:
```python
def _extract_with_source(stats_dict: dict, field_names: list[str]) -> tuple[float, str]:
    """
    Extract field value and track source.
    
    Returns (value, source) where source in:
    - "real": Value from actual API data
    - "missing_sentinel": API returned -1/-2 (normalized to None in DB)
    - "fallback_default": No source found, using hardcoded default
    """
    for field_name in field_names:
        raw_value = stats_dict.get(field_name)
        
        if raw_value is None:
            # Check if this was originally a sentinel (-1/-2)
            # by checking database flag
            source_flag = stats_dict.get(f"__{field_name}_source")
            if source_flag == "missing_sentinel":
                return (0.0, "missing_sentinel")
            elif source_flag == "fallback_default":
                return (0.0, "fallback_default")
        else:
            # Real value found
            return (float(raw_value), "real")
    
    # No field found anywhere -> fallback
    return (0.0, "fallback_default")

def build_features(match_data: dict) -> dict:
    """
    Build features with source tracking.
    
    Returns feature dict where:
    - "feature_name": float value
    - "__source_feature_name": "real" | "missing_sentinel" | "fallback_default"
    """
    features = {}
    
    home_stats = match_data["home_team_stats"]
    away_stats = match_data["away_team_stats"]
    
    # shots_diff
    home_shots_val, home_shots_src = _extract_with_source(home_stats, ["shots_avg_overall", "shots_avg_home"])
    away_shots_val, away_shots_src = _extract_with_source(away_stats, ["shots_avg_overall", "shots_avg_away"])
    
    shots_diff = home_shots_val - away_shots_val
    shots_diff_source = "real" if (home_shots_src == "real" and away_shots_src == "real") else "missing_sentinel"
    
    features["shots_diff"] = shots_diff
    features["__source_shots_diff"] = shots_diff_source
    
    # ... repeat for all features
    
    return features
```

### Task 1.2: Update Feature Quality Profiling
**File:** `scheduler/auto_train.py` → `_build_feature_profile_v2()`

Changes:
```python
def _build_feature_profile_v2(rows: list[dict]) -> dict:
    """Enhanced profiling with source metadata."""
    
    profile = {}
    for feature_name in self.required_features:
        real_count = 0
        missing_count = 0
        fallback_count = 0
        total_count = 0
        
        for row in rows:
            value = row.get(feature_name)
            source = row.get(f"__source_{feature_name}", "unknown")
            
            if value is not None:
                total_count += 1
                if source == "real":
                    real_count += 1
                elif source == "missing_sentinel":
                    missing_count += 1
                elif source == "fallback_default":
                    fallback_count += 1
        
        total = total_count
        profile[feature_name] = {
            "total": total,
            "real_count": real_count,
            "real_rate": real_count / total if total else 0.0,
            "missing_count": missing_count,
            "missing_rate": missing_count / total if total else 0.0,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / total if total else 0.0,
            "quality": self._classify_feature_quality(real_count, missing_count, fallback_count, total)
        }
    
    return profile
```

### Task 1.3: Create API Field Auditor
**File:** `api/field_auditor.py` (new)

```python
class FootyStatsFieldAuditor:
    """Audits FootyStats API field coverage."""
    
    def __init__(self, api: FootyStatsClient | None = None):
        self.api = api or FootyStatsClient()
    
    def audit_team_stats_fields(self, season_id: int, limit: int = 5000) -> dict:
        """Audit team_season_stats coverage."""
        # Load teams with stats
        # Check which fields are present vs sentinel vs missing
        # Return coverage report
        pass
    
    def audit_match_fields(self, season_id: int, limit: int = 5000) -> dict:
        """Audit match-level fields."""
        pass
    
    def generate_report(self) -> str:
        """Generate human-readable audit report."""
        pass
```

### Task 1.4: Create Database Migration (Optional but Recommended)
**File:** `database/migrations/add_source_tracking.sql` (new)

```sql
-- Add source tracking columns to team_season_stats
ALTER TABLE team_season_stats ADD COLUMN shots_avg_overall_source TEXT DEFAULT 'unknown';
ALTER TABLE team_season_stats ADD COLUMN possession_avg_overall_source TEXT DEFAULT 'unknown';
-- ... etc for all stats fields

-- Or use JSON if schema modification is risky:
ALTER TABLE team_season_stats ADD COLUMN field_metadata JSON DEFAULT '{}';
```

---

## IMPLEMENTATION TASKS (PRIORITY 2)

### Task 2.1: Create Reference Documentation
**File:** `docs/API_FIELD_REFERENCE.md` (new)

Document for each endpoint/field:
- Expected from docs
- Actual observed
- Coverage %
- Sentinel markers
- Use case
- Reliability rating

### Task 2.2: Create Feature Quality Report UI
**File:** `ui/main.py` → Enhanced `_analysis_tab()`

Display:
```
┌─ Feature Quality Analysis ──────────────────────┐
│                                                  │
│ shots_diff                                       │
│  Real: 5%  Missing: 94%  Fallback: 1%           │
│  Status: ⚠️ WEAK (94% missing sentinel data)     │
│  Recommendation: Consider excluding             │
│                                                  │
│ possession_diff                                  │
│  Real: 95%  Missing: 4%  Fallback: 1%           │
│  Status: ✓ USABLE (95% real data)               │
│                                                  │
│ xg_diff                                          │
│  Real: 0%  Missing: 100%  Fallback: 0%          │
│  Status: ✗ UNUSABLE (API doesn't provide xG)    │
│  Recommendation: EXCLUDE                        │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Task 2.3: Create Audit Script
**File:** `scripts/audit_api_fields.py` (new)

```python
#!/usr/bin/env python
"""Audit API field coverage across all endpoints."""

from api.field_auditor import FootyStatsFieldAuditor
import json
from pathlib import Path
from datetime import datetime

def main():
    auditor = FootyStatsFieldAuditor()
    
    # Audit all endpoints
    team_stats_audit = auditor.audit_team_stats_fields(limit=5000)
    match_audit = auditor.audit_match_fields(limit=5000)
    league_audit = auditor.audit_league_fields(limit=1000)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "team_stats": team_stats_audit,
        "matches": match_audit,
        "league": league_audit,
        "summary": auditor.generate_summary()
    }
    
    # Save report
    reports_dir = Path("reports")
    filename = f"api_field_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(reports_dir / filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Audit report saved: {filename}")
    print(auditor.generate_human_readable_report())

if __name__ == "__main__":
    main()
```

---

## IMPLEMENTATION TASKS (PRIORITY 3)

### Task 3.1: Document Weak Features Decision Matrix

Create for each weak feature a decision document:
- `docs/WEAK_FEATURES_ANALYSIS.md`
- Includes: problem analysis, root cause, decision, rationale

### Task 3.2: Implement Automated Feature Quality Checker

Add to training pipeline:
- Flag which features have >60% missing_sentinel_rate
- Recommend action: use / caution / exclude
- Log decisions in training report

### Task 3.3: Update Model Trainer to Use Quality Metadata

Modify `core.model.trainer`: 
- Option to exclude low-quality features
- Option to use source metadata as confidence weights
- Option to retrain with only "real" data

---

## TOTOBRIEF AUDIT (PARALLEL WORK)

### Verify TotoBrief Integration

Audit goals:
- ✓ Confirm TotoBrief used ONLY for TOTO history, not ML feature source
- ✓ Verify TotoBrief field keys match documentation
- ✓ Check that all references to Totobrief are in `toto/` module only

**Files to check:**
- `api/toto_api.py` — Should only extract: drawing_id, events, quotes (pool/bk odds)
- `toto/pipeline.py` — Should use Totobrief for decision layer, not features
- No imports of TotoBrief in `core/features/builder.py` (should be zero)

**Report:** Create `docs/TOTOBRIEF_INTEGRATION_VERIFICATION.md`

---

## TESTING STRATEGY

### Unit Tests
- `tests/test_field_auditor.py` — Verify audit logic
- `tests/test_builder_with_source.py` — Verify source tracking in features
- `tests/test_feature_profile_v2.py` — Verify enhanced profiling

### Integration Tests
- `tests/test_api_field_audit_end_to_end.py` — Run full audit, verify report
- `tests/test_source_tracking_pipeline.py` — End-to-end: API → DB → features → profile

### Manual Validation
- [ ] Run audit script on actual database
- [ ] Compare audit output with docs
- [ ] Verify weak features show correct missing_rate
- [ ] Confirm TotoBrief not imported in core/features

---

## DELIVERABLES

### By End of Phase 7

**Code Changes:**
- ✓ `core/features/builder_with_source.py` — New version with source tracking
- ✓ `api/field_auditor.py` — Audit module
- ✓ `scripts/audit_api_fields.py` — Standalone audit script
- ✓ Enhanced `scheduler/auto_train.py` — Source-aware profiling
- ✓ Enhanced `ui/main.py` — Feature quality display

**Documentation:**
- ✓ `docs/API_FIELD_REFERENCE.md` — All field mappings
- ✓ `docs/WEAK_FEATURES_ANALYSIS.md` — Decision matrix for each weak feature
- ✓ `docs/TOTOBRIEF_VERIFICATION.md` — TotoBrief integration audit
- ✓ `reports/api_field_audit_*.json` — Automated reports

**Tests:**
- ✓ Unit tests (>80% coverage of new modules)
- ✓ Integration tests
- ✓ Manual validation checklist completed

---

## PRIORITY & TIMELINE

```
PRIORITY 1 (4 days) — Core Source Tracking
├─ Task 1.1: Feature builder with source tracking
├─ Task 1.2: Enhanced feature profiling
├─ Task 1.3: API field auditor module
└─ Task 1.4: Tests + validation

PRIORITY 2 (3 days) — Documentation & Reports
├─ Task 2.1: API field reference doc
├─ Task 2.2: Feature UI enhancements
├─ Task 2.3: Audit script + automation
└─ Tests + manual validation

PRIORITY 3 (2 days) — Decision & Knowledge Base
├─ Task 3.1: Weak features decision matrix
├─ Task 3.2: Automated quality checker
├─ Task 3.3: Model trainer integration
└─ TotoBrief verification

TOTAL: 9 working days
```

---

## SUCCESS CRITERIA

✅ **Functional:**
1. Feature values carry source metadata (real/missing/fallback)
2. Feature builder no longer silently converts None → 0.0
3. Audit reports show actual API field coverage
4. Feature quality dashboard shows real vs. missing rates
5. Weak features properly classified with root causes
6. No breaking changes to existing training/prediction

✅ **Documentation:**
7. API field mapping fully documented
8. Weak features analysis with decisions recorded
9. TotoBrief integration verified and documented

✅ **Data Quality:**
10. Can distinguish between real zero, missing sentinel, and fallback
11. Training reports honest about data quality
12. Model can make informed decisions about feature usage

---

## RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| Breaking existing models | New source tracking is additive; old code still works |
| Performance impact | Source metadata minimal (string field per feature) |
| Incomplete audit | Automated audit script ensures consistency |
| TotoBrief confusion | Clear documentation + grep verification |

---

## NEXT STEPS

1. **Immediate:** Review this plan
2. **Day 1-2:** Implement Priority 1 (source tracking in features)
3. **Day 3:** Unit & integration tests
4. **Day 4-5:** Priority 2 (documentation, reports, UI)
5. **Day 6-7:** Priority 3 (decision matrix, auto-quality checker)
6. **Day 8-9:** Full integration testing, validation, documentation

---

**Questions? Ready to start with Priority 1?**

