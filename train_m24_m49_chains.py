import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("HorseRaceAI")
from src.utils.helpers import setup_logger, set_seed, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("TrainM24M49Chains")

def train_cascade_step(
    step1_net: nn.Module,
    f1_indices: list,
    f2_indices: list,
    data_tr: dict,
    data_va: dict,
    data_te: dict,
    device: torch.device,
    save_path: str,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 0.001
):
    max_horses = 18
    num_f2 = len(f2_indices) + 1 # 201
    input_dim2 = max_horses * num_f2 # 3618

    def predict_net(net, X_3d, masks):
        net.eval()
        n = len(X_3d)
        input_dim = X_3d.shape[1] * X_3d.shape[2]
        X_flat = torch.tensor(X_3d.reshape(-1, input_dim), dtype=torch.float32)
        mask_t = torch.tensor(masks, dtype=torch.float32)
        eval_bs = 256
        probs = []
        with torch.no_grad():
            for i in range(0, n, eval_bs):
                bx = X_flat[i:i+eval_bs].to(device)
                bm = mask_t[i:i+eval_bs].to(device)
                logits = net(bx, mask=bm)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0)

    # 1. Step 1 による推論
    logger.info("Computing Step 1 predictions on Train/Val/Test...")
    X_tr1 = data_tr["race_3d"][:, :, f1_indices]
    X_va1 = data_va["race_3d"][:, :, f1_indices]
    X_te1 = data_te["race_3d"][:, :, f1_indices]

    p_tr1 = predict_net(step1_net, X_tr1, data_tr["race_masks"])
    p_va1 = predict_net(step1_net, X_va1, data_va["race_masks"])
    p_te1 = predict_net(step1_net, X_te1, data_te["race_masks"])

    # 2. Step 2 入力データ作成 (200特徴量 + Step 1予測値 = 201特徴量)
    logger.info("Constructing Step 2 cascade inputs (201 features)...")
    X_tr2 = np.concatenate([data_tr["race_3d"][:, :, f2_indices], p_tr1[:, :, np.newaxis]], axis=-1)
    X_va2 = np.concatenate([data_va["race_3d"][:, :, f2_indices], p_va1[:, :, np.newaxis]], axis=-1)
    X_te2 = np.concatenate([data_te["race_3d"][:, :, f2_indices], p_te1[:, :, np.newaxis]], axis=-1)

    X_tr2_flat = torch.tensor(X_tr2.reshape(-1, input_dim2), dtype=torch.float32)
    mask_tr = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr = torch.tensor(data_tr["race_y"], dtype=torch.int64)

    X_va2_flat = torch.tensor(X_va2.reshape(-1, input_dim2), dtype=torch.float32)
    mask_va = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    X_te2_flat = torch.tensor(X_te2.reshape(-1, input_dim2), dtype=torch.float32)
    mask_te = torch.tensor(data_te["race_masks"], dtype=torch.float32)

    # Step 2 モデル定義
    net2 = RaceLevelStage1NN(input_dim=input_dim2, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    optimizer = optim.AdamW(net2.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_score = -1.0
    best_state = None
    num_races = len(X_tr2_flat)

    for epoch in range(1, epochs + 1):
        net2.train()
        perm = torch.randperm(num_races)
        for i in range(0, num_races, batch_size):
            idx = perm[i:i+batch_size]
            bx = X_tr2_flat[idx].to(device)
            bm = mask_tr[idx].to(device)
            by = y_tr[idx].to(device)

            optimizer.zero_grad()
            logits = net2(bx, mask=bm)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Val 評価
        val_p = predict_net(net2, X_va2, data_va["race_masks"])
        m_va = calculate_race_array_payout_metric(val_p, data_va["race_masks"], data_va["race_y"], data_va["odds"], data_va["payouts"], "prob")
        val_score = m_va["roi"] * min(1.0, m_va["hit_rate"] / 0.20) + m_va["hit_rate"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu() for k, v in net2.state_dict().items()}

    # 最良モデル復元
    net2.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, save_path)
    logger.info(f"Saved Step 2 model to {save_path}!")

    # 最終評価
    val_p = predict_net(net2, X_va2, data_va["race_masks"])
    test_p = predict_net(net2, X_te2, data_te["race_masks"])
    m_va = calculate_race_array_payout_metric(val_p, data_va["race_masks"], data_va["race_y"], data_va["odds"], data_va["payouts"], "prob")
    m_te = calculate_race_array_payout_metric(test_p, data_te["race_masks"], data_te["race_y"], data_te["odds"], data_te["payouts"], "prob")

    logger.info(f"Final Step 2 Performance -> Val ROI: {m_va['roi']*100:.2f}%, Test ROI: {m_te['roi']*100:.2f}%")
    return net2, m_va, m_te

def main():
    device = get_device()
    set_seed(2026)

    cached_path = "HorseRaceAI/artifacts_dirt_1win/cached_dirt_1win_data.pkl"
    logger.info(f"Loading data from {cached_path}...")
    with open(cached_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    col_to_idx = data_dict["col_to_idx"]
    bac_col_indices = data_dict["bac_col_indices"]

    # Model 24 & Model 49 特徴量インデックス
    with open("HorseRaceAI/artifacts_dirt_1win/features_24.json", "r", encoding="utf-8") as f:
        f24_meta = json.load(f)
    f24_indices = list(bac_col_indices) + [col_to_idx[c] for c in f24_meta["yearly_features"]]

    with open("HorseRaceAI/artifacts_dirt_1win/features_49.json", "r", encoding="utf-8") as f:
        f49_meta = json.load(f)
    f49_indices = list(bac_col_indices) + [col_to_idx[c] for c in f49_meta["yearly_features"]]

    # 1. Model 24 単体 (Step 1) ロード
    net24 = RaceLevelStage1NN(input_dim=18*200, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net24.load_state_dict(torch.load("HorseRaceAI/artifacts_dirt_1win/model_24.pth", map_location=device))

    # 2. Model 49 単体 (Step 1) ロード
    net49 = RaceLevelStage1NN(input_dim=18*200, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net49.load_state_dict(torch.load("HorseRaceAI/artifacts_dirt_1win/model_49.pth", map_location=device))

    # === Train Chain 1: M24 -> M49 ===
    logger.info("\n==========================================")
    logger.info("  Training Chain 1: M24 -> M49")
    logger.info("==========================================")
    save_m24_to_m49 = "HorseRaceAI/artifacts_dirt_1win/cascade_M24_to_M49.pth"
    train_cascade_step(net24, f24_indices, f49_indices, data_tr, data_va, data_te, device, save_m24_to_m49)

    # === Train Chain 2: M49 -> M24 ===
    logger.info("\n==========================================")
    logger.info("  Training Chain 2: M49 -> M24")
    logger.info("==========================================")
    save_m49_to_m24 = "HorseRaceAI/artifacts_dirt_1win/cascade_M49_to_M24.pth"
    train_cascade_step(net49, f49_indices, f24_indices, data_tr, data_va, data_te, device, save_m49_to_m24)

    logger.info("\nAll M24 & M49 cascade chains trained and saved successfully!")

if __name__ == "__main__":
    main()
