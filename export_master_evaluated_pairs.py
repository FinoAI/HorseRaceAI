import os
import json
import pandas as pd
import numpy as np

def create_master_evaluated_pairs():
    # 探索結果ファイル一覧
    files_info = [
        {
            "path": "HorseRaceAI/artifacts_cascade_remaining_1904/remaining_search_all_pairs.csv",
            "source": "remaining_1904_exhaustive",
            "type": "pair"
        },
        {
            "path": "HorseRaceAI/artifacts_cascade_continue_2086/continue_search_all_pairs.csv",
            "source": "continue_search_2086",
            "type": "pair"
        },
        {
            "path": "HorseRaceAI/artifacts_cascade_search_new_pairs/search_new_pairs_all.csv",
            "source": "search_new_pairs",
            "type": "pair"
        },
        {
            "path": "HorseRaceAI/artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv",
            "source": "4origins_exhaustive",
            "type": "exhaustive"
        },
        {
            "path": "HorseRaceAI/artifacts_cascade_unstarted/unstarted_origins_cascade_all_steps.csv",
            "source": "unstarted_origins",
            "type": "chain"
        },
        {
            "path": "HorseRaceAI/artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv",
            "source": "random_cascade_100",
            "type": "chain"
        },
        {
            "path": "HorseRaceAI/artifacts_specific_chain/specific_chain_summary.csv",
            "source": "specific_chain",
            "type": "specific"
        }
    ]

    master_records = {}

    # 1. pair 形式 / exhaustive 形式 / chain 形式から収集
    for info in files_info:
        p = info["path"]
        src = info["source"]
        ftype = info["type"]
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)

        if ftype == "pair":
            for _, r in df.iterrows():
                m1 = int(r["origin_model"])
                m2 = int(r["second_model"])
                key = (m1, m2)
                if key not in master_records:
                    master_records[key] = {
                        "pair_name": f"M{m1}->M{m2}",
                        "origin_model": m1,
                        "second_model": m2,
                        "source_experiment": src,
                        "s1_test_roi": float(r.get("s1_test_roi", np.nan)),
                        "s2_test_roi": float(r.get("s2_test_roi", np.nan)),
                        "delta_test_roi": float(r.get("delta_test_roi", np.nan)),
                        "s1_val_roi": float(r.get("s1_val_roi", np.nan)),
                        "s2_val_roi": float(r.get("s2_val_roi", np.nan)),
                        "delta_val_roi": float(r.get("delta_val_roi", np.nan)),
                        "test_hit_rate": float(r.get("test_hit_rate", np.nan)),
                        "test_hits": int(r.get("test_hits", 0)) if not pd.isna(r.get("test_hits", 0)) else np.nan,
                        "test_mean_odds": float(r.get("test_mean_odds", np.nan)),
                        "test_payout": float(r.get("test_payout", np.nan)),
                        "val_hit_rate": float(r.get("val_hit_rate", np.nan)),
                        "val_hits": int(r.get("val_hits", 0)) if not pd.isna(r.get("val_hits", 0)) else np.nan,
                        "val_mean_odds": float(r.get("val_mean_odds", np.nan)),
                        "val_payout": float(r.get("val_payout", np.nan)),
                        "is_both_improved": bool(r.get("is_both_improved", False)),
                        "is_over_100": bool(r.get("is_over_100", False))
                    }

        elif ftype == "exhaustive":
            for _, r in df.iterrows():
                m1 = int(r["origin_model"])
                m2 = int(r["second_model"])
                key = (m1, m2)
                if key not in master_records:
                    s2_test = float(r.get("test_roi", np.nan))
                    master_records[key] = {
                        "pair_name": f"M{m1}->M{m2}",
                        "origin_model": m1,
                        "second_model": m2,
                        "source_experiment": src,
                        "s1_test_roi": np.nan,
                        "s2_test_roi": s2_test,
                        "delta_test_roi": np.nan,
                        "s1_val_roi": np.nan,
                        "s2_val_roi": np.nan,
                        "delta_val_roi": np.nan,
                        "test_hit_rate": float(r.get("test_hit_rate", np.nan)),
                        "test_hits": int(r.get("test_hits", 0)) if not pd.isna(r.get("test_hits", 0)) else np.nan,
                        "test_mean_odds": float(r.get("test_mean_odds", np.nan)),
                        "test_payout": float(r.get("test_payout", np.nan)),
                        "val_hit_rate": np.nan,
                        "val_hits": np.nan,
                        "val_mean_odds": np.nan,
                        "val_payout": np.nan,
                        "is_both_improved": False,
                        "is_over_100": bool(s2_test >= 1.0)
                    }

        elif ftype == "chain":
            for cid in df["chain_id"].unique():
                cdf = df[df["chain_id"] == cid]
                s1 = cdf[cdf["step"] == 1]
                s2 = cdf[cdf["step"] == 2]
                if len(s1) > 0 and len(s2) > 0:
                    m1 = int(s1["model_idx"].iloc[0])
                    m2 = int(s2["model_idx"].iloc[0])
                    key = (m1, m2)
                    if key not in master_records:
                        r2 = s2.iloc[0]
                        r1 = s1.iloc[0]
                        t_roi_key = "test_roi" if "test_roi" in r2 else "full_test_roi"
                        t_hit_key = "test_hit_rate" if "test_hit_rate" in r2 else "full_test_hit_rate"
                        t_hits_key = "test_hits" if "test_hits" in r2 else "full_test_hits"
                        t_odds_key = "test_mean_odds" if "test_mean_odds" in r2 else "full_test_mean_odds"
                        t_pay_key = "test_payout" if "test_payout" in r2 else "full_test_payout"

                        s1_test = float(r1.get(t_roi_key, np.nan))
                        s2_test = float(r2.get(t_roi_key, np.nan))
                        s1_val = float(r1.get("val_roi", np.nan))
                        s2_val = float(r2.get("val_roi", np.nan))

                        d_test = s2_test - s1_test if not (pd.isna(s2_test) or pd.isna(s1_test)) else np.nan
                        d_val = s2_val - s1_val if not (pd.isna(s2_val) or pd.isna(s1_val)) else np.nan
                        both = bool((d_test > 0) and (d_val > 0)) if not (pd.isna(d_test) or pd.isna(d_val)) else False
                        over = bool((s2_test >= 1.0) or (s2_val >= 1.0))

                        master_records[key] = {
                            "pair_name": f"M{m1}->M{m2}",
                            "origin_model": m1,
                            "second_model": m2,
                            "source_experiment": src,
                            "s1_test_roi": s1_test,
                            "s2_test_roi": s2_test,
                            "delta_test_roi": d_test,
                            "s1_val_roi": s1_val,
                            "s2_val_roi": s2_val,
                            "delta_val_roi": d_val,
                            "test_hit_rate": float(r2.get(t_hit_key, np.nan)),
                            "test_hits": int(r2.get(t_hits_key, 0)) if not pd.isna(r2.get(t_hits_key, 0)) else np.nan,
                            "test_mean_odds": float(r2.get(t_odds_key, np.nan)),
                            "test_payout": float(r2.get(t_pay_key, np.nan)),
                            "val_hit_rate": float(r2.get("val_hit_rate", np.nan)),
                            "val_hits": int(r2.get("val_hits", 0)) if not pd.isna(r2.get("val_hits", 0)) else np.nan,
                            "val_mean_odds": float(r2.get("val_mean_odds", np.nan)),
                            "val_payout": float(r2.get("val_payout", np.nan)),
                            "is_both_improved": both,
                            "is_over_100": over
                        }

        elif ftype == "specific":
            # M17 -> M50
            key = (17, 50)
            if key not in master_records:
                master_records[key] = {
                    "pair_name": "M17->M50",
                    "origin_model": 17,
                    "second_model": 50,
                    "source_experiment": src,
                    "s1_test_roi": 0.7554,
                    "s2_test_roi": 0.8107,
                    "delta_test_roi": 0.0553,
                    "s1_val_roi": 0.8694,
                    "s2_val_roi": 0.8535,
                    "delta_val_roi": -0.0159,
                    "test_hit_rate": 0.2130,
                    "test_hits": 160,
                    "test_mean_odds": 3.80,
                    "test_payout": 60880.0,
                    "val_hit_rate": np.nan,
                    "val_hits": np.nan,
                    "val_mean_odds": np.nan,
                    "val_payout": np.nan,
                    "is_both_improved": False,
                    "is_over_100": False
                }

    # ソートして DataFrame 化
    records_list = sorted(list(master_records.values()), key=lambda x: (x["origin_model"], x["second_model"]))
    df_master = pd.DataFrame(records_list)

    out_csv = "HorseRaceAI/all_evaluated_pairs_master.csv"
    out_json = "HorseRaceAI/all_evaluated_pairs_master.json"
    df_master.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records_list, f, ensure_ascii=False, indent=2)

    print(f"Master file saved successfully!")
    print(f"CSV : {out_csv} ({len(df_master)} unique pairs)")
    print(f"JSON: {out_json}")

if __name__ == "__main__":
    create_master_evaluated_pairs()
