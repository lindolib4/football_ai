"""Quick smoke test for the new optimizer."""
import sys
sys.path.insert(0, ".")
from toto.optimizer_new import TotoOptimizer

opt = TotoOptimizer()

# Test 1: pure model (no pool probs) - should stay model-driven
m1 = [{"name": "A vs B", "probs": {"P1": 0.55, "PX": 0.28, "P2": 0.17}}]
r1 = opt.optimize(m1, "16")
s1 = opt.last_run_summary
print("TEST1 model-only:")
print("  coupons:", len(r1))
hs1 = s1.get("history_strategy", {})
print("  pool_signals_used_count:", hs1.get("pool_signals_used_count"))
print("  history_strategy_used:", hs1.get("history_strategy_used"))

# Test 2: pool_probs differ from model - should trigger pool signal
m2 = [
    {
        "name": "A vs B",
        "probs": {"P1": 0.55, "PX": 0.28, "P2": 0.17},
        "pool_probs": {"P1": 0.30, "PX": 0.42, "P2": 0.28},
    },
    {
        "name": "C vs D",
        "probs": {"P1": 0.48, "PX": 0.30, "P2": 0.22},
        "pool_probs": {"P1": 0.25, "PX": 0.33, "P2": 0.42},
    },
    {
        "name": "E vs F",
        "probs": {"P1": 0.40, "PX": 0.35, "P2": 0.25},
        "pool_probs": {"P1": 0.18, "PX": 0.28, "P2": 0.54},
    },
]

r2 = opt.optimize_insurance(m2, "16", 0.7)
s2 = opt.last_run_summary
print("\nTEST2 pool divergence (strength=0.7):")
print("  coupons:", len(r2))
hs2 = s2.get("history_strategy", {})
print("  pool_signals_used_count:", hs2.get("pool_signals_used_count"))
print("  history_changed_decisions_count:", hs2.get("history_changed_decisions_count"))
print("  strategy_signals_used:", hs2.get("strategy_signals_used", [])[:6])
diag2 = s2.get("match_level_decision_diagnostics", [])
for d in diag2:
    mi = d["match_index"]
    md = d["model_decision"]
    fd = d["strategy_adjusted_decision"]
    pu = d.get("pool_signals_used")
    reason = d.get("reason", "")
    print(f"  match {mi}: model={md} -> final={fd} | pool_used={pu} | reason={reason}")

# Test 3: strength 0.9 vs 0.7 difference
r3 = opt.optimize_insurance(m2, "16", 0.9)
s3 = opt.last_run_summary
ins3 = s3.get("insurance_diagnostics", {})

r7 = opt.optimize_insurance(m2, "16", 0.7)
s7 = opt.last_run_summary
ins7 = s7.get("insurance_diagnostics", {})

print("\nTEST3 strength=0.9 vs 0.7:")
print("  0.9 -> insured_coupons:", ins3.get("insured_coupons_count"), "cells_changed:", ins3.get("insurance_cells_changed_count"))
print("  0.7 -> insured_coupons:", ins7.get("insured_coupons_count"), "cells_changed:", ins7.get("insurance_cells_changed_count"))

# Test 4: base vs insured ordering
base_idxs = s3.get("insurance_diagnostics", {}).get("base_coupons_count", 0)
insured_idxs = s3.get("insured_coupon_indices", [])
print("\nTEST4 coupon ordering (0.9):")
print("  base_coupons_count:", base_idxs)
print("  insured_coupon_indices (first 5):", insured_idxs[:5])

# Test 5: toto_brief_history.stats fallback
m3 = [
    {
        "name": "G vs H",
        "probs": {"P1": 0.45, "PX": 0.30, "P2": 0.25},
        "toto_brief_history": {
            "drawings": [{"id": 1}, {"id": 2}, {"id": 3}],
            "history_draws_loaded_count": 3,
            "history_events_loaded_count": 45,
            "history_stats_ready": True,
            "history_state": "full_history_ready",
            "stats": {
                "events_count": 45.0,
                "draw_results_rate": 0.31,
                "upset_rate": 0.42,
                "favorite_fail_rate": 0.38,
                "pool_vs_bookmaker_gap": 0.08,
            },
        },
    }
]
r5 = opt.optimize(m3, "16")
s5 = opt.last_run_summary
diag5 = s5.get("match_level_decision_diagnostics", [])
print("\nTEST5 toto_brief_history.stats fallback:")
for d in diag5:
    print("  history_state:", d.get("history_state"))
    print("  history_ready:", d.get("history_ready"))
    print("  reason:", d.get("reason"))
hs5 = s5.get("history_strategy", {})
print("  draws_loaded:", hs5.get("history_draws_loaded_count"))
print("  events_loaded:", hs5.get("history_events_loaded_count"))

print("\nALL TESTS COMPLETE")
