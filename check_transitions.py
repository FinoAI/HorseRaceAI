import pandas as pd

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
for cid in [33, 84, 99]:
    cdf = df[df["chain_id"] == cid]
    print(f"\n--- Chain {cid} ---")
    for _, r in cdf.iterrows():
        st = int(r["step"])
        m = int(r["model_idx"])
        roi = r["full_test_roi"] * 100
        hit = r["full_test_hit_rate"] * 100
        hits = int(r["full_test_hits"])
        odds = r["full_test_mean_odds"]
        print(f"Step {st} (M{m:2d}): ROI={roi:6.2f}%, Hit={hit:5.2f}% ({hits:3d}R), Odds={odds:4.2f}x")
