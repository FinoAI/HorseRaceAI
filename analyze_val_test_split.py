import pandas as pd
import numpy as np

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")

# ValとTestの分解
val_inv = 751 * 100
test_inv = 751 * 100
full_inv = 1502 * 100

val_payout = df["val_roi"] * val_inv
test_payout = df["full_test_payout"] - val_payout
test_roi = test_payout / test_inv

df["val_payout"] = val_payout
df["test_payout"] = test_payout
df["test_roi"] = test_roi

print("=== 3 Winning Patterns: Val vs Test Breakdown ===")
winning_chains = [(33, 3, 42), (84, 2, 43), (99, 2, 45)]

for cid, st, mid in winning_chains:
    row = df[(df["chain_id"] == cid) & (df["step"] == st)].iloc[0]
    print(f"\n--- Chain {cid} Step {st} (Model {mid}) ---")
    print(f"  [前半 Val  (751R)] ROI: {row['val_roi']*100:6.2f}% | Payout: ¥{row['val_payout']:,.0f}")
    print(f"  [後半 Test (751R)] ROI: {row['test_roi']*100:6.2f}% | Payout: ¥{row['test_payout']:,.0f}")
    print(f"  [全量 1502R 合計 ] ROI: {row['full_test_roi']*100:6.2f}% | Payout: ¥{row['full_test_payout']:,.0f}")

print("\n=== All Steps where TEST (後半751R) ROI >= 100% ===")
test_over_100 = df[df["test_roi"] >= 1.00].sort_values(by="test_roi", ascending=False)
print(f"Found {len(test_over_100)} steps where TEST alone >= 100%:")
for _, r in test_over_100.head(15).iterrows():
    c_id = int(r["chain_id"])
    st = int(r["step"])
    m_id = int(r["model_idx"])
    print(f"Chain {c_id:3d} Step {st} (M{m_id:2d}) | Test ROI: {r['test_roi']*100:6.2f}% | Val ROI: {r['val_roi']*100:6.2f}% | Full ROI: {r['full_test_roi']*100:6.2f}%")

print("\n=== Steps where BOTH Val >= 100% AND Test >= 100% (Double 100%+) ===")
both_over_100 = df[(df["val_roi"] >= 1.00) & (df["test_roi"] >= 1.00)].sort_values(by="full_test_roi", ascending=False)
print(f"Found {len(both_over_100)} steps where BOTH Val >= 100% and Test >= 100%:")
for _, r in both_over_100.iterrows():
    c_id = int(r["chain_id"])
    st = int(r["step"])
    m_id = int(r["model_idx"])
    print(f"Chain {c_id:3d} Step {st} (M{m_id:2d}) | Val: {r['val_roi']*100:6.2f}% | Test: {r['test_roi']*100:6.2f}% | Full: {r['full_test_roi']*100:6.2f}%")
