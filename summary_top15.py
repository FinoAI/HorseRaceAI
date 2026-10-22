import pandas as pd
import json

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
top15 = df.sort_values(by="full_test_roi", ascending=False).head(15)

print("\n=== TOP 15 FULL TEST ROI STEPS (out of 276 models trained across 100 chains) ===")
for _, r in top15.iterrows():
    c_id = int(r["chain_id"])
    st = int(r["step"])
    m_id = int(r["model_idx"])
    roi = r["full_test_roi"] * 100
    hit = r["full_test_hit_rate"] * 100
    hits = int(r["full_test_hits"])
    odds = r["full_test_mean_odds"]
    payout = r["full_test_payout"]
    print(f"Rank {_+1:2d} | Chain {c_id:3d} Step {st} (M{m_id:2d}) | Full ROI: {roi:6.2f}% | Hit: {hit:5.2f}% ({hits:3d}/1502R) | Odds: {odds:5.2f}x | Payout: ¥{payout:,.0f}")

print("\n=== ROI >= 100% PATTERNS DETAIL ===")
with open("artifacts_cascade_full1502/roi_over_100_full1502_patterns.json") as f:
    patterns = json.load(f)

for p in patterns:
    print(f"Pattern {p['pattern_id']}: Chain {p['chain_id']} Step {p['step']}")
    print(f"  Chain: {' -> '.join([f'M{m}' for m in p['chain_models']])}")
    print(f"  Full Test ROI: {p['full_test_roi']*100:.2f}% | Hit: {p['full_test_hit_rate']*100:.2f}% ({p['full_test_hits']}/1502R) | Odds: {p['full_test_mean_odds']:.2f}x | Payout: ¥{p['full_test_payout']:,.0f}")
    print(f"  Val ROI: {p['val_roi']*100:.2f}% | Val Hit: {p['val_hit_rate']*100:.2f}%\n")
