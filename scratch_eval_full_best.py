import os
import sys
import pickle
import numpy as np
import torch
import json

from src.evaluation.metrics import calculate_race_array_payout_metric
from src.models.base_nn import RaceLevelStage1NN

def main():
    with open("artifacts_dirt_1win/cached_dirt_1win_data.pkl", "rb") as f:
        d = pickle.load(f)

    data_va = d["data_va"]
    data_te = d["data_te"]

    odds = np.concatenate([data_va["odds"], data_te["odds"]], axis=0)
    payouts = np.concatenate([data_va["payouts"], data_te["payouts"]], axis=0)
    masks = np.concatenate([data_va["race_masks"], data_te["race_masks"]], axis=0)
    y = np.concatenate([data_va["race_y"], data_te["race_y"]], axis=0)
    total_races = len(y)

    print(f"Total 2025-2026 Combined Races: {total_races}")

    bac_indices = d["bac_col_indices"]
    col_to_idx = d["col_to_idx"]

    # 1. BEST MODEL (M24 + M49)
    def get_feat_idx_best(m):
        with open(f"artifacts_dirt_1win/BEST_MODEL_DIRT_1WIN/features_{m}.json") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_indices) + s_idx

    idx_24 = get_feat_idx_best(24)
    idx_49 = get_feat_idx_best(49)

    X_3d_va = data_va["race_3d"]
    X_3d_te = data_te["race_3d"]
    X_3d_all = np.concatenate([X_3d_va, X_3d_te], axis=0)

    def predict_simple(feat_idx, weight_file):
        X_sub = X_3d_all[:, :, feat_idx]
        dim = 18 * len(feat_idx)
        X_flat = torch.tensor(X_sub.reshape(-1, dim), dtype=torch.float32)
        mask_t = torch.tensor(masks, dtype=torch.float32)

        net = RaceLevelStage1NN(input_dim=dim, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15)
        net.load_state_dict(torch.load(weight_file, map_location="cpu"))
        net.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, total_races, 256):
                bx = X_flat[i:i+256]
                bmask = mask_t[i:i+256]
                logits = net(bx, mask=bmask)
                p = torch.softmax(logits, dim=-1).numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    p24 = predict_simple(idx_24, "artifacts_dirt_1win/BEST_MODEL_DIRT_1WIN/model_24.pth")
    p49 = predict_simple(idx_49, "artifacts_dirt_1win/BEST_MODEL_DIRT_1WIN/model_49.pth")
    p_ens = (p24 + p49) / 2.0

    r24 = calculate_race_array_payout_metric(p24, masks, y, odds, payouts, selection_mode="prob")
    r49 = calculate_race_array_payout_metric(p49, masks, y, odds, payouts, selection_mode="prob")
    rens = calculate_race_array_payout_metric(p_ens, masks, y, odds, payouts, selection_mode="prob")

    odds_24 = r24['total_payout'] / (r24['hits'] * 100) if r24['hits'] > 0 else 0
    odds_49 = r49['total_payout'] / (r49['hits'] * 100) if r49['hits'] > 0 else 0
    odds_ens = rens['total_payout'] / (rens['hits'] * 100) if rens['hits'] > 0 else 0

    print("\n=======================================================")
    print("  [BEST MODEL: M24 + M49] on FULL 1,502 RACES (2025-2026)")
    print("=======================================================")
    print(f"Model 24 : Hit={r24['hit_rate']*100:.2f}% ({r24['hits']}/{total_races}R), MeanOdds={odds_24:.2f}倍, ROI={r24['roi']*100:.2f}%, Payout=¥{r24['total_payout']:,.0f}")
    print(f"Model 49 : Hit={r49['hit_rate']*100:.2f}% ({r49['hits']}/{total_races}R), MeanOdds={odds_49:.2f}倍, ROI={r49['roi']*100:.2f}%, Payout=¥{r49['total_payout']:,.0f}")
    print(f"Ensemble : Hit={rens['hit_rate']*100:.2f}% ({rens['hits']}/{total_races}R), MeanOdds={odds_ens:.2f}倍, ROI={rens['roi']*100:.2f}%, Payout=¥{rens['total_payout']:,.0f}")

    # 2. SPECIFIC CHAIN (M17 -> M50 -> M40 -> M42)
    def get_feat_idx_dirt(m):
        with open(f"artifacts_dirt_1win/features_{m}.json") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_indices) + s_idx

    chain_models = [
        (1, 17, "chain_step_1_m17.pth"),
        (2, 50, "chain_step_2_m50.pth"),
        (3, 40, "chain_step_3_m40.pth"),
        (4, 42, "chain_step_4_m42.pth"),
    ]

    print("\n=======================================================")
    print("  [CHAIN: M17 -> M50 -> M40 -> M42] on FULL 1,502 RACES")
    print("=======================================================")

    prev_p = None
    chain_probs = []

    for step, m_idx, w_file in chain_models:
        f_idx = get_feat_idx_dirt(m_idx)
        X_b = X_3d_all[:, :, f_idx]
        if prev_p is not None:
            X_in = np.concatenate([X_b, prev_p[:, :, np.newaxis]], axis=-1)
        else:
            X_in = X_b

        dim = 18 * X_in.shape[2]
        X_flat = torch.tensor(X_in.reshape(-1, dim), dtype=torch.float32)
        mask_t = torch.tensor(masks, dtype=torch.float32)

        net = RaceLevelStage1NN(input_dim=dim, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15)
        net.load_state_dict(torch.load(os.path.join("artifacts_specific_chain", w_file), map_location="cpu"))
        net.eval()

        probs = []
        with torch.no_grad():
            for i in range(0, total_races, 256):
                bx = X_flat[i:i+256]
                bmask = mask_t[i:i+256]
                logits = net(bx, mask=bmask)
                p = torch.softmax(logits, dim=-1).numpy()
                probs.append(p)
        cur_p = np.concatenate(probs, axis=0)
        prev_p = cur_p
        chain_probs.append(cur_p)

        res = calculate_race_array_payout_metric(cur_p, masks, y, odds, payouts, selection_mode="prob")
        m_odds = res['total_payout'] / (res['hits'] * 100) if res['hits'] > 0 else 0
        print(f"Step {step} (M{m_idx:2d}) : Hit={res['hit_rate']*100:.2f}% ({res['hits']}/{total_races}R), MeanOdds={m_odds:.2f}倍, ROI={res['roi']*100:.2f}%, Payout=¥{res['total_payout']:,.0f}")

    # 累積アンサンブル
    print("\n--- Chain Cumulative Ensemble on FULL 1,502 RACES ---")
    for k in range(1, 5):
        ens_p = np.mean(chain_probs[:k], axis=0)
        res_cum = calculate_race_array_payout_metric(ens_p, masks, y, odds, payouts, selection_mode="prob")
        c_odds = res_cum['total_payout'] / (res_cum['hits'] * 100) if res_cum['hits'] > 0 else 0
        m_names = [str(chain_models[i][1]) for i in range(k)]
        print(f"Steps 1..{k} [{' ,'.join(m_names)}] : Hit={res_cum['hit_rate']*100:.2f}% ({res_cum['hits']}/{total_races}R), MeanOdds={c_odds:.2f}倍, ROI={res_cum['roi']*100:.2f}%, Payout=¥{res_cum['total_payout']:,.0f}")

if __name__ == "__main__":
    main()
