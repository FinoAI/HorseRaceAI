import pandas as pd
import json

df = pd.read_csv("artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv")
print(f"Total pairs in summary: {len(df)}")

print("\n=== TOP 10 BY TEST ROI ===")
top_test = df.sort_values(by="test_roi", ascending=False).head(10)
for _, r in top_test.iterrows():
    m1 = int(r["origin_model"])
    m2 = int(r["second_model"])
    t_roi = r["test_roi"] * 100
    v_roi = r["val_roi"] * 100
    t_hit = r["test_hit_rate"] * 100
    t_hits = int(r["test_hits"])
    t_odds = r["test_mean_odds"]
    t_pay = r["test_payout"]
    print(f"M{m1:2d} -> M{m2:2d} | Test ROI: {t_roi:6.2f}% (Hit={t_hit:5.2f}%, {t_hits:3d}R, {t_odds:4.2f}x, ¥{t_pay:,.0f}) | Val ROI: {v_roi:6.2f}%")

print("\n=== ALL 14 PAIRS WITH VAL ROI >= 100% ===")
top_val = df[df["val_roi"] >= 1.00].sort_values(by="val_roi", ascending=False)
for _, r in top_val.iterrows():
    m1 = int(r["origin_model"])
    m2 = int(r["second_model"])
    t_roi = r["test_roi"] * 100
    v_roi = r["val_roi"] * 100
    v_hit = r["val_hit_rate"] * 100
    v_hits = int(r["val_hits"])
    v_odds = r["val_mean_odds"]
    v_pay = r["val_payout"]
    t_hit = r["test_hit_rate"] * 100
    t_hits = int(r["test_hits"])
    t_odds = r["test_mean_odds"]
    t_pay = r["test_payout"]
    print(f"M{m1:2d} -> M{m2:2d} | Val ROI: {v_roi:6.2f}% (Val Hit={v_hit:5.2f}%, {v_hits:3d}R, {v_odds:4.2f}x) | Test ROI: {t_roi:6.2f}% (Test Hit={t_hit:5.2f}%, {t_hits:3d}R, {t_odds:4.2f}x, ¥{t_pay:,.0f})")

print("\n=== 4 ORIGINS BREAKDOWN ===")
for m1 in [45, 2, 8, 47]:
    sub = df[df["origin_model"] == m1]
    t100 = len(sub[sub["test_roi"] >= 1.00])
    v100 = len(sub[sub["val_roi"] >= 1.00])
    best_t = sub.sort_values(by="test_roi", ascending=False).iloc[0]
    best_v = sub.sort_values(by="val_roi", ascending=False).iloc[0]
    print(f"\n--- Origin Model {m1:2d} (全49ペア完了) ---")
    print(f"  Test ROI >= 100% ペア数: {t100}件 | Val ROI >= 100% ペア数: {v100}件")
    print(f"  最高 Test ROI ペア: M{m1} -> M{int(best_t['second_model'])} (Test ROI: {best_t['test_roi']*100:.2f}%, Val: {best_t['val_roi']*100:.2f}%)")
    print(f"  最高 Val  ROI ペア: M{m1} -> M{int(best_v['second_model'])} (Val  ROI: {best_v['val_roi']*100:.2f}%, Test: {best_v['test_roi']*100:.2f}%)")
