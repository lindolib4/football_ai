# PHASE 5 IMPLEMENTATION PLAN
## TOTO Decision Layer Integration

**Project:** FootAI  
**Phase:** 5 — TOTO Model Integration, TotoBrief History, Coupon Generation, Insurance  
**Date:** March 2026  
**Status:** PLANNING → IMPLEMENTATION  

---

## EXECUTIVE SUMMARY

This plan outlines the systematic implementation of a **profit-oriented TOTO decision layer** that wraps the existing FootAI prediction model. The layer will:

1. Unify input data across TOTO operations
2. Score matches by model vs. market value
3. Integrate historical TotoBrief data for bias correction
4. Generate coupons with 4 strategy modes (Conservative/Balanced/Aggressive/Profit-max)
5. Add intelligent insurance coverage for uncovered outcomes
6. Maintain full backward compatibility with the existing prediction pipeline

**Key Constraint:** The TOTO layer is **additive**, not **replacement**. The core FootAI model (1/X/2 prediction) remains unchanged.

---

## CURRENT STATE ASSESSMENT

### Existing TOTO Architecture
```
toto/
├── generator.py          # TotoGenerator: match analysis → coupon combinations
├── strategies.py         # Minimal: entropy-based double picking
├── pipeline.py           # TotoPipeline: API orchestration
├── backtest.py           # Historical testing
├── batch_backtest.py     # Bulk validation
├── optimizer.py          # Coupon optimization
├── coverage.py           # Coverage analysis
└── __init__.py
```

### Key Dependencies
- **core.decision.FinalDecisionEngine**: Combines prediction + value betting logic
- **core.value.ValueEngine**: EV calculation against market odds
- **api.TotoAPI**: Fetches draw/match data
- **core.decision.DecisionEngine**: Raw prediction logic

### Current Limitations
- ❌ No TotoBrief history integration
- ❌ No profit-oriented strategy selection
- ❌ No insurance layer
- ❌ No structured scoring framework (model vs. market)
- ❌ No match classification for intelligent coverage
- ❌ No fallback mode documentation

### Strengths to Build On
- ✅ TotoGenerator architecture is sound
- ✅ FinalDecisionEngine already combines prediction + value
- ✅ TotoPipeline provides orchestration pattern
- ✅ Decision + Value engines provide foundation

---

## IMPLEMENTATION ROADMAP

### PHASE 5 TIMELINE & PRIORITIES

```
Priority 1: TOTO Decision Layer Core (Week 1)
├─ Data Models (unified TOTO input structure)
├─ Scoring layer (model vs. market framework)
├─ Match classification
└─ Single/double decision logic

Priority 2: TotoBrief & Profit Logic (Week 2)
├─ TotoBrief history provider
├─ Strategy modes (Conservative/Balanced/Aggressive/Profit-max)
├─ Profit-oriented coupon generation
└─ Coverage priority ranking

Priority 3: Insurance & Fallback (Week 3)
├─ Insurance layer architecture
├─ Uncovered outcome detection
├─ Insurance coupon generation
└─ Budget-aware insurance

Priority 4: UI Integration & Backtest (Week 4)
├─ TOTO tab settings expansion
├─ Summary generation
├─ TOTO backtest module
└─ Strategy comparison reports
```

---

## PRIORITY 1: TOTO DECISION LAYER CORE

### 1.1 Data Models (New Module: `toto/data_models.py`)

**Purpose:** Single, versioned structure for all TOTO-input data. All modules consume/produce this.

```python
# Structure (pseudo-code for clarity)

@dataclass
class TotoMatchInput:
    """Unified TOTO match input."""
    
    # Identification
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: datetime
    order_in_coupon: int | None
    
    # Market data
    odds_ft_1: float | None
    odds_ft_x: float | None
    odds_ft_2: float | None
    pool_prob_1: float | None
    pool_prob_x: float | None
    pool_prob_2: float | None
    
    # Model predictions
    model_prob_1: float
    model_prob_x: float
    model_prob_2: float
    model_top_pick: str  # "1" | "X" | "2"
    model_confidence: float
    entropy: float
    
    # Quality flags
    feature_quality_flag: str  # "strong" | "moderate" | "weak"
    predictable_flag: bool
    
    # TotoBrief data (optional)
    toto_brief_history: TotoBriefSignal | None
    historical_outcome_freq: dict[str, float] | None
    public_bias: str | None  # "favors_1" | "favors_x" | "favors_2" | None

@dataclass
class TotoBriefSignal:
    """Optional historical signal from TotoBrief."""
    draw_id: str | None
    similar_matches_count: int
    historical_1_freq: float
    historical_x_freq: float
    historical_2_freq: float
    popular_pick: str | None
    surprise_frequency: float  # How often "unexpected" outcomes happen

@dataclass
class TotoMatchScore:
    """Scoring result for a match."""
    match_id: str
    
    # Model vs. market comparison
    delta_1: float  # model_prob - market_prob for 1
    delta_x: float
    delta_2: float
    
    # Value identification
    best_value_pick: str | None  # Which outcome has best delta
    value_score: float  # 0.0-1.0
    
    # Confidence
    model_confidence: float
    uncertainty_penalty: float
    
    # Classification
    match_type: str  # "strong_single" | "value_single" | "borderline_double" | ...
    danger_level: str  # "low" | "moderate" | "high"
    
    # Recommendations
    recommended_pick: str  # Single outcome or double "1X" | "X2" | "12"
    confidence_for_single: float
    must_insure: bool

@dataclass
class TotoStrategy:
    """Strategy parameters for coupon generation."""
    name: str  # "conservative" | "balanced" | "aggressive" | "profit_max"
    use_model_probs: bool
    use_market_probs: bool
    use_toto_brief: bool
    use_pool_probs: bool
    
    # Strategy-specific thresholds
    single_confidence_threshold: float
    double_on_uncertainty: bool
    max_doubles_per_coupon: int
    max_risky_picks_per_coupon: int
    
    # Insurance settings
    enable_insurance: bool
    insurance_strength: str  # "light" | "moderate" | "heavy"
    max_insurance_coupons: int
    
    # Constraints
    max_total_coupons: int
    prefer_singles_when_confident: bool
    allow_contrarian_picks: bool

@dataclass
class TotoGenerationResult:
    """Output of coupon generation."""
    coupons: list[list[str]]  # List of coupons, each is list of picks
    main_coupons: list[list[str]]
    insurance_coupons: list[list[str]]
    
    # Summary metrics
    total_coupons_count: int
    main_coverage: dict[str, int]  # Count of each outcome covered
    uncovered_outcomes: dict[str, list[int]]  # match_id -> [outcomes not in main]
    
    # Explanation
    match_decisions: dict[str, TotoMatchScore]
    strategy_notes: str
    insurance_reason: str | None
```

**Files to Create:**
- `toto/data_models.py` — All dataclasses

**Integration Points:**
- **Generator → DataModel:** Convert input dict to `TotoMatchInput`
- **Scorer → DataModel:** Produce `TotoMatchScore`
- **StrategyEngine → DataModel:** Consume `TotoStrategy`

---

### 1.2 Scoring Module (New Module: `toto/scoring.py`)

**Purpose:** Model vs. market comparison framework. Zero dependency on TotoBrief.

```python
class TotoScorer:
    """Scores matches by model confidence + market value."""
    
    def score_match(self, match_input: TotoMatchInput) -> TotoMatchScore:
        """
        Score a single match.
        
        Calculates:
        - Delta between model prob and market prob
        - Value identification
        - Uncertainty penalty
        - Match type classification
        - Danger level
        - Recommended pick
        """
        
    def _calculate_value_score(self, deltas: dict) -> float:
        """Value score: 0.0 (no value) to 1.0 (strong value)."""
        
    def _classify_match_type(self, 
                            confidence: float, 
                            entropy: float,
                            value_delta: float) -> str:
        """
        Returns one of:
        - "strong_single" (high confidence, high value, low entropy)
        - "value_single" (good delta, moderate confidence)
        - "borderline_double" (close probabilities)
        - "high_uncertainty" (high entropy)
        - "market_trap" (crowd bias detected later)
        - "low_information" (weak features)
        """
        
    def _calculate_danger_level(self, match_type: str, entropy: float) -> str:
        """Risk level: low | moderate | high."""
        
    def _recommend_pick(self, match_type: str, probs: dict) -> str:
        """Single outcome or double "1X" | "X2" | "12"."""
```

**Algorithm:**

```
For each match:
  1. Extract model probs (P1, PX, P2)
  2. Derive market probs from odds (implied_prob)
  3. Calculate deltas for each outcome
  4. Identify best_value_pick (outcome with highest positive delta)
  5. Calculate value_score from delta magnitude
  6. Apply uncertainty penalty from entropy
  7. Classify match type from (confidence, entropy, value_delta)
  8. Determine danger_level from match_type + entropy
  9. Recommend pick: single (if confidence high) or double (if borderline)
  10. Set must_insure flag based on danger + entropy
```

**Files to Create:**
- `toto/scoring.py` — TotoScorer class

**Testing:**
- Unit tests: `tests/test_toto_scorer.py`
  - Verify delta calculation
  - Verify match_type classification
  - Verify single vs. double recommendation

---

### 1.3 Strategy Module (New Module: `toto/strategy.py`)

**Purpose:** Implement 4 strategy modes + single/double decision logic.

```python
class StrategyEngine:
    """Applies strategy-specific rules to decisions."""
    
    def apply_strategy(self, 
                      scores: list[TotoMatchScore],
                      strategy: TotoStrategy) -> list[TotoMatchScore]:
        """
        Modify/filter scores based on strategy mode.
        
        Strategy modes:
        1. Conservative
           - Prioritize hit rate over upside
           - More doubles
           - More insurance
           
        2. Balanced
           - Mix singles and doubles
           - Moderate insurance
           
        3. Aggressive
           - Prioritize value edges
           - Fewer doubles, more singles
           - Light insurance
           
        4. Profit-max
           - Maximize expected value
           - Accept higher risk for better upside
           - Minimal but strategic insurance
        """
        
    def _apply_conservative_rules(self, scores: list[TotoMatchScore]) -> list[TotoMatchScore]:
        """Increase doubles, reduce singles, boost insurance."""
        
    def _apply_aggressive_rules(self, scores: list[TotoMatchScore]) -> list[TotoMatchScore]:
        """Favor singles, reduce doubles, minimize insurance."""
        
    def _select_picks_for_coupon(self, 
                                scores: list[TotoMatchScore],
                                strategy: TotoStrategy) -> list[str]:
        """Generate final picks list from scores + strategy."""
```

**Decision Logic:**

```
For each match (given strategy):
  1. Get base recommendation (from Scorer)
  2. Check strategy thresholds
  3. If Conservative:
     - Promote doubles if entropy > threshold
     - Enforce max_doubles
     - Mark for insurance if danger_level == "high"
  4. If Aggressive:
     - Prefer single if confidence high
     - Skip doubles unless borderline
     - Only insure if value present
  5. Output final pick (single or double)
```

**Files to Create:**
- `toto/strategy.py` — StrategyEngine class

**Testing:**
- Unit tests: `tests/test_toto_strategy.py`
  - Verify conservative promotes doubles
  - Verify aggressive prefers singles
  - Verify strategy-specific recommendations

---

### 1.4 Integration into Existing Pipeline

**File to Modify:** `toto/generator.py`

**Changes:**
```python
from toto.data_models import TotoMatchInput, TotoMatchScore, TotoStrategy
from toto.scoring import TotoScorer
from toto.strategy import StrategyEngine

class TotoGenerator:
    def __init__(self):
        self.final_engine = FinalDecisionEngine()
        self.scorer = TotoScorer()  # NEW
        self.strategy_engine = StrategyEngine()  # NEW
    
    def generate(self, matches: list, mode: str, strategy: TotoStrategy = None) -> TotoGenerationResult:
        """Enhanced generate with strategy + scoring."""
        
        strategy = strategy or self._get_default_strategy(mode)
        
        # Convert matches to unified input
        match_inputs = self._prepare_match_inputs(matches)
        
        # Score each match
        scores = [self.scorer.score_match(m) for m in match_inputs]
        
        # Apply strategy
        scores = self.strategy_engine.apply_strategy(scores, strategy)
        
        # Generate coupons (existing logic, with strategy-aware picks)
        coupons = self._generate_coupons(scores, strategy)
        
        # Build result
        return TotoGenerationResult(...)
    
    def _prepare_match_inputs(self, matches: list) -> list[TotoMatchInput]:
        """Convert raw match dicts to TotoMatchInput."""
        pass
```

---

## PRIORITY 2: TOTOBRIEF & PROFIT LOGIC

### 2.1 TotoBrief History Provider (New Module: `toto/history_provider.py`)

**Purpose:** Unified interface for TotoBrief data, works with API and fallback.

```python
class TotoBriefProvider:
    """Loads and normalizes TotoBrief historical data."""
    
    def __init__(self, api: TotoAPI | None = None):
        self.api = api  # Optional external API
        self.cache: dict = {}  # Local cache
        
    def get_similar_matches(self, 
                           home_team: str, 
                           away_team: str, 
                           league: str,
                           limit: int = 50) -> list[dict]:
        """Find similar past matches."""
        
    def get_outcome_frequencies(self, 
                               home_team: str, 
                               away_team: str,
                               league: str) -> dict[str, float]:
        """Historical freq of 1/X/2."""
        
    def get_public_bias(self, 
                       draw_id: str = None) -> dict[str, str]:
        """Current/historical public bias for outcomes."""
        
    def is_available(self) -> bool:
        """Check if TotoBrief data accessible."""
```

**Fallback Behavior:**
- If API unavailable → return empty/neutral signals
- If data incomplete → log warning, use available fields
- Never halt on missing TotoBrief

**Files to Create:**
- `toto/history_provider.py` — TotoBriefProvider class

---

### 2.2 Profit-Oriented Coupon Builder (New Module: `toto/coupon_builder.py`)

**Purpose:** Build coupons with profit optimization, not just coverage.

```python
class ProfitOrientedCouponBuilder:
    """Generates coupons optimizing expected profitability."""
    
    def build(self, 
             scores: list[TotoMatchScore],
             strategy: TotoStrategy,
             odds_data: dict[str, float]) -> TotoGenerationResult:
        """
        Build coupons with profit orientation.
        
        Algorithm:
        1. Rank matches by match_type + value_score
        2. For each strategy:
           - Conservative: maximize hit probability
           - Balanced: balance hit rate + upside
           - Aggressive: maximize expected value
           - Profit-max: pure EV optimization
        3. Select coverage priority
        4. Build coupon combinations
        5. Respect constraints (max coupons, max doubles, etc.)
        """
        
    def _rank_matches_by_importance(self, 
                                   scores: list[TotoMatchScore]) -> list[TotoMatchScore]:
        """Sort by harm if missed (danger_level) + profit if hit (value_score)."""
        
    def _estimate_coupon_value(self, coupon: list[str], scores: dict) -> float:
        """Estimate expected profitability of a coupon."""
        
    def _select_coverage_strategy(self, 
                                 scores: list[TotoMatchScore],
                                 strategy: TotoStrategy) -> dict[str, str]:
        """Return dict: match_id -> pick (single or double or skip)."""
```

**Files to Create:**
- `toto/coupon_builder.py` — ProfitOrientedCouponBuilder class

---

### 2.3 Enhanced Pipeline (New Module: `toto/engine.py`)

**Purpose:** Orchestrate Priority 1+2 workflow.

```python
class ToToDecisionEngine:
    """Main orchestration layer: data → scoring → strategy → coupons."""
    
    def __init__(self, 
                 api: TotoAPI | None = None,
                 history_provider: TotoBriefProvider | None = None):
        self.api = api
        self.history = history_provider or TotoBriefProvider(api)
        self.scorer = TotoScorer()
        self.strategy_engine = StrategyEngine()
        self.coupon_builder = ProfitOrientedCouponBuilder()
        
    def generate_coupons(self,
                        matches: list,
                        strategy: TotoStrategy) -> TotoGenerationResult:
        """
        End-to-end: raw matches → TotoGenerationResult.
        
        Workflow:
        1. Normalize input (→ TotoMatchInput)
        2. Fold in TotoBrief history (if available)
        3. Score matches (→ TotoMatchScore)
        4. Apply strategy rules (→ modified TotoMatchScore)
        5. Build coupons (→ TotoGenerationResult)
        """
        pass
```

**Files to Create:**
- `toto/engine.py` — ToToDecisionEngine class (renamed from current pipeline usage)

---

## PRIORITY 3: INSURANCE & FALLBACK

### 3.1 Insurance Layer (New Module: `toto/insurance.py`)

**Purpose:** Intelligent coverage of uncovered outcomes.

```python
class InsuranceLayer:
    """Builds insurance coupons for uncovered outcomes."""
    
    def generate_insurance(self,
                          main_coupons: list[list[str]],
                          match_scores: dict[str, TotoMatchScore],
                          strategy: TotoStrategy) -> list[list[str]]:
        """
        Build insurance coupons covering high-risk uncovered outcomes.
        
        Algorithm:
        1. Find uncovered outcomes in main coupons
        2. Rank by danger (high-danger matches first)
        3. Filter by strategy:
           - Conservative: insure most dangerous
           - Aggressive: insure only high-value alternatives
        4. Build coupon combinations for insurance
        5. Respect max_insurance_coupons budget
        """
        
    def _find_uncovered_outcomes(self,
                                main_coupons: list[list[str]],
                                all_matches: int) -> dict[str, list[str]]:
        """Return dict: match_idx -> uncovered outcomes."""
        
    def _rank_insurance_priorities(self,
                                  uncovered: dict,
                                  scores: dict[str, TotoMatchScore],
                                  strategy: TotoStrategy) -> list[tuple[str, float]]:
        """Return sorted list of (match_id, insurance_priority)."""
        
    def _build_insurance_coupons(self,
                                priorities: list[tuple],
                                limit: int) -> list[list[str]]:
        """Generate combinations covering high-priority uncovered outcomes."""
```

**Files to Create:**
- `toto/insurance.py` — InsuranceLayer class

---

### 3.2 Summary & Explanation (New Module: `toto/summary.py`)

**Purpose:** Generate human-readable explanations.

```python
class TotoSummaryBuilder:
    """Builds summary/explanation of generation results."""
    
    def build(self, result: TotoGenerationResult) -> str:
        """
        Generate multi-section summary:
        
        Section 1: Match Classification
        - X strong singles
        - Y borderline doubles
        - Z high-uncertainty (need insurance)
        
        Section 2: Coverage by Outcome
        - Main coupons cover 1XXX times, X XXX times, 2 XXX times
        - Uncovered: match_id outcomes
        
        Section 3: Strategy Notes
        - Why this strategy selected
        - Critical risks
        - Value opportunities
        
        Section 4: Insurance Justification
        - Why insurance added
        - Which outcomes covered by insurance
        - Insurance coupons count
        """
        pass
```

**Files to Create:**
- `toto/summary.py` — TotoSummaryBuilder class

---

## PRIORITY 4: UI INTEGRATION & BACKTEST

### 4.1 UI Settings Expansion (Modify: `ui/main.py` TOTO tab)

**New Settings Section:**

```
[Источники] (Sources)
☑ Model predictions
☑ Market odds (implied probability)
☐ TotoBrief history (if available)
☐ Pool/crowd probabilities (if available)

[Стратегия] (Strategy)
◯ Conservative (more doubles, more insurance)
◯ Balanced (mix of singles/doubles)
◯ Aggressive (profit edges, fewer doubles)
◯ Profit-max (pure EV optimization, high risk)

[Страховка] (Insurance)
☑ Enable insurance
  Strength: ⊢Light──Moderate──Heavy─⊣
  Max insurance coupons: [int]

[Ограничения] (Constraints)
Max total coupons: [int]
Max doubles per coupon: [int]
Prefer singles when confident: ☑
```

**Files to Modify:**
- `ui/main.py` — _toto_tab() method

---

### 4.2 TOTO Backtest Module (New Module: `toto/backtest_strategy.py`)

**Purpose:** Validate strategy performance on historical data.

```python
class TotoStrategyBacktest:
    """Test strategy modes against historical toto draws."""
    
    def backtest(self, 
                historical_draws: list[dict],
                strategies: list[TotoStrategy]) -> dict:
        """
        Run backtest. For each historical draw:
        1. Get actual match results
        2. Generate coupons using each strategy
        3. Check if coupons hit
        4. Calculate metrics
        
        Returns:
        {
            "strategy_name": {
                "hit_rate": 0.XX,
                "avg_coupons_count": X,
                "full_hits": X,
                "near_misses": X,
                "profitability_estimate": X
            }
        }
        """
        pass
```

**Files to Create:**
- `toto/backtest_strategy.py` — TotoStrategyBacktest class

---

## MODULE DEPENDENCY GRAPH

```
ui/main.py (TOTO tab)
    ↓
toto/engine.py (ToToDecisionEngine)
    ├─ toto/data_models.py (TotoMatchInput, TotoMatchScore, etc.)
    ├─ toto/scoring.py (TotoScorer)
    ├─ toto/strategy.py (StrategyEngine)
    ├─ toto/coupon_builder.py (ProfitOrientedCouponBuilder)
    ├─ toto/insurance.py (InsuranceLayer)
    ├─ toto/summary.py (TotoSummaryBuilder)
    ├─ toto/history_provider.py (TotoBriefProvider)
    │   └─ api/toto_api.py (TotoAPI, optional)
    └─ core/decision/final_decision_engine.py (FinalDecisionEngine, unchanged)

toto/generator.py (refactored, still backward compatible)
    └─ Same dependencies as engine.py

toto/backtest_strategy.py (TotoStrategyBacktest)
    └─ toto/engine.py + historical data

toto/backtest.py (existing, unchanged)
    └─ Remains for compatibility
```

---

## IMPLEMENTATION SEQUENCE

### Week 1: Priority 1
- [ ] Create data_models.py with all dataclasses
- [ ] Create scoring.py with TotoScorer
- [ ] Create strategy.py with StrategyEngine
- [ ] Integrate into generator.py
- [ ] Write tests for all three modules
- [ ] Manual validation with existing UI

### Week 2: Priority 2
- [ ] Create history_provider.py with TotoBriefProvider
- [ ] Create coupon_builder.py with ProfitOrientedCouponBuilder
- [ ] Create engine.py with ToToDecisionEngine orchestration
- [ ] Implement strategy.py methods (apply_conservative_rules, etc.)
- [ ] Write tests for history + builder
- [ ] Integration testing

### Week 3: Priority 3
- [ ] Create insurance.py with InsuranceLayer
- [ ] Create summary.py with TotoSummaryBuilder
- [ ] Integrate insurance into engine.py
- [ ] Build summary generation UI
- [ ] Write tests for insurance + summary

### Week 4: Priority 4
- [ ] Expand TOTO tab in ui/main.py
- [ ] Create backtest_strategy.py with TotoStrategyBacktest
- [ ] Build summary UI panels
- [ ] Full integration testing
- [ ] Documentation + examples

---

## TESTING STRATEGY

### Unit Tests (per module)
- `test_data_models.py` — dataclass validation
- `test_toto_scorer.py` — scoring logic
- `test_toto_strategy.py` — strategy application
- `test_coupon_builder.py` — coupon generation
- `test_history_provider.py` — history retrieval
- `test_insurance.py` — insurance logic
- `test_summary.py` — summary generation

### Integration Tests
- `test_toto_engine.py` — full workflow
- `test_toto_with_history.py` — with/without TotoBrief
- `test_strategy_comparison.py` — all 4 strategy modes

### Manual Testing Checklist
- [ ] Generate coupons with Conservative strategy
- [ ] Generate coupons with Aggressive strategy
- [ ] Verify insurance coupons added when enabled
- [ ] Verify summary accuracy
- [ ] Verify TotoBrief integration (if API available)
- [ ] Verify fallback behavior (TotoBrief unavailable)
- [ ] Backtest on historical draws

---

## SUCCESS CRITERIA

✅ **Functional:**
1. TOTO uses model probabilities (not manual-only)
2. Single unified input structure (TotoMatchInput)
3. TotoBrief history integrated when available
4. Fallback mode works without TotoBrief
5. 4 strategy modes implemented (Conservative/Balanced/Aggressive/Profit-max)
6. Insurance layer covers uncovered outcomes
7. Summary explains decisions for each match

✅ **Code Quality:**
8. All logic in backend modules (not ui/main.py)
9. No breaking changes to existing modules
10. < 100 lines per function
11. Full test coverage (>80% lines)
12. Type hints on all function signatures

✅ **Backward Compatibility:**
13. Existing TotoGenerator still works
14. FinalDecisionEngine unchanged
15. TotoPipeline unchanged

---

## EFFORT ESTIMATE

| Phase | Task | Effort | Notes |
|-------|------|--------|-------|
| 1 | data_models.py | 4h | Dataclass design |
| 1 | scoring.py | 8h | Delta calculation, classification |
| 1 | strategy.py | 6h | Strategy application rules |
| 1 | Integration + tests | 6h |  |
| **Total P1** | **24h** | | ~3 days |
| 2 | history_provider.py | 4h | API integration |
| 2 | coupon_builder.py | 8h | Profit optimization |
| 2 | engine.py | 6h | Orchestration |
| 2 | Tests + integration | 6h | |
| **Total P2** | **24h** | | ~3 days |
| 3 | insurance.py | 6h | Coverage logic |
| 3 | summary.py | 4h | Explanation generation |
| 3 | Tests + integration | 4h | |
| **Total P3** | **14h** | | ~2 days |
| 4 | UI expansion | 6h | Settings + controls |
| 4 | backtest_strategy.py | 8h | Historical validation |
| 4 | Tests + documentation | 6h | |
| **Total P4** | **20h** | | ~3 days |
| **TOTAL PHASE 5** | **82h** | | ~10 working days |

---

## RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| TotoBrief API unavailable | Fallback mode planned, tested without API |
| Performance with large datasets | Lazy loading for history, caching strategy |
| Strategy logic conflicts | Explicit priority rules defined per strategy |
| Breaking existing TOTO | Maintain backward compatibility, new engine separate |
| Complex logic → unmaintainable | Max 100 lines per function, comprehensive tests |

---

## NEXT STEPS

1. **Immediate:** Review this plan with team
2. **Day 1:** Create toto/data_models.py (dataclass definitions)
3. **Day 2-3:** Implement Priority 1 modules (scoring + strategy)
4. **Day 4-5:** Implement Priority 2 modules (history + builder)
5. **Day 6:** Implement Priority 3 modules (insurance + summary)
6. **Day 7:** Expand UI, add backtest
7. **Day 8-10:** Full integration testing, documentation, refinement

---

**Questions?** Let's validate the architecture before starting implementation.

