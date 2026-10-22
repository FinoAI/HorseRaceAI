import pandas as pd
import json

df = pd.read_csv("artifacts_cascade_unstarted/unstarted_origins_cascade_all_steps.csv")

# Val 100% 超えのパターンを含むチェーン
# Pattern 1: Chain 6
# Pattern 2 & 3: Chain 13
# Pattern 4: Chain 15
# Pattern 5: Chain 17
# さらに各起点（45, 2, 8, 47）の代表チェーン

chains_to_detail = [6, 13, 15, 17, 2, 7, 16]

for cid in chains_to_detail:
    cdf = df[df["chain_id"] == cid]
    start_m = int(cdf.iloc[0]["model_idx"])
    print(f"\n==========================================================================================")
    print(f"  Chain {cid:2d} (起点: Model {start_m:2d}) 成績推移")
    print(f"==========================================================================================")
    print(f"{'Step':<6} | {'接続モデル':<14} | {'特徴量数':<6} | {'的中率 (的中数)':<16} | {'的中平均オッズ':<10} | {'払戻金額(Test)':<12} | {'テストROI':<10} | {'検証ROI(Val)':<10}")
    print("-" * 105)
    for _, r in cdf.iterrows():
        st = int(r["step"])
        m = int(r["model_idx"])
        n_feat = int(r["num_features"])
        t_hit = r["test_hit_rate"] * 100
        t_hits = int(r["test_hits"])
        t_odds = r["test_mean_odds"]
        t_pay = r["test_payout"]
        t_roi = r["test_roi"] * 100
        v_roi = r["val_roi"] * 100
        m_str = f"M{m} (起点)" if st == 1 else f"... -> M{m}"
        print(f"Step {st:<2} | {m_str:<14} | {n_feat:<6} | {t_hit:5.2f}% ({t_hits:3d}R)   | {t_odds:4.2f}倍     | ¥{t_pay:>9,.0f} | {t_roi:>8.2f}% | {v_roi:>8.2f}%")
