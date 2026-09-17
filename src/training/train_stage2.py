import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.models.meta_nn import RaceLevelMetaNN
from src.evaluation.metrics import calculate_race_array_payout_metric
from src.utils.helpers import setup_logger, get_device

logger = setup_logger("TrainStage2")

def build_race_meta_features(preds_list: List[np.ndarray]) -> np.ndarray:
    """
    10個の前段モデルのレース予測（各 N_races × 18）からメタ特徴量を構築
    - 10モデルの予測値 (18 × 10 = 180次元)
    - 各馬番スロットごとの統計特徴: 平均(18), 標準偏差(18), 最大値(18), 最小値(18) = 72次元
    合計 252次元
    """
    # preds_list: 10 elements of shape (N_races, 18)
    P_3d = np.stack(preds_list, axis=-1) # shape: (N_races, 18, 10)
    N_races = P_3d.shape[0]

    # フラット化予測値 (N_races, 180)
    flat_preds = P_3d.reshape(N_races, 18 * len(preds_list))

    # 各馬番の統計量 (N_races, 18)
    mean_val = np.mean(P_3d, axis=-1)
    std_val = np.std(P_3d, axis=-1)
    max_val = np.max(P_3d, axis=-1)
    min_val = np.min(P_3d, axis=-1)

    meta_features = np.hstack([flat_preds, mean_val, std_val, max_val, min_val])
    return meta_features.astype(np.float32)

def train_race_meta_model(
    X_train_meta: np.ndarray,
    race_train: Dict[str, Any],
    X_val_meta: np.ndarray,
    race_val: Dict[str, Any],
    max_horses: int = 18,
    multiplier: int = 32,
    reduction_ratio: float = 0.5,
    dropout: float = 0.1,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.00005,
    epochs: int = 20,
    early_stopping_patience: int = 4,
    eval_metric: str = "payout", # "payout" | "hit_rate" | "hybrid"
    selection_mode: str = "ev_filtered", # "prob" | "ev" | "ev_filtered"
    min_prob_for_ev: float = 0.10,
    loss_weighting: str = "none", # "none" | "payout"
    artifacts_dir: str = "./artifacts_models",
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """
    レース単位入力型 後段メタNN (スタッキングアンサンブル) の学習
    """
    if device is None:
        device = get_device()

    input_dim = X_train_meta.shape[1]
    logger.info(
        f"--- [Meta Model] RaceLevelMetaNN: input_dim={input_dim} | 1st layer={input_dim * multiplier:,} units (x{multiplier}) | "
        f"1/2 reduction ratio={reduction_ratio} | Eval={eval_metric} (mode={selection_mode}) | loss_weighting={loss_weighting} ---"
    )

    model = RaceLevelMetaNN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=multiplier,
        reduction_ratio=reduction_ratio,
        dropout=dropout
    ).to(device)

    logger.info(f"[Meta Model] Total layers: {model.total_layers} (5+ layers deep architecture)")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # 配当重み付き損失の準備
    y_tr = race_train["race_y"]
    mask_tr = race_train["race_masks"]
    if loss_weighting == "payout":
        winner_odds = race_train["odds"][np.arange(len(y_tr)), y_tr]
        race_weights = 1.0 + np.log1p(np.maximum(1.0, winner_odds)).astype(np.float32)
        race_weights_t = torch.tensor(race_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        race_weights_t = None
        criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, "meta_model.pth")

    X_tr_t = torch.tensor(X_train_meta, dtype=torch.float32)
    mask_tr_t = torch.tensor(mask_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64)

    X_va_t = torch.tensor(X_val_meta, dtype=torch.float32)
    mask_va_t = torch.tensor(race_val["race_masks"], dtype=torch.float32)
    y_va = race_val["race_y"]

    num_races = len(X_train_meta)
    batch_size = min(128, num_races)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_races)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_races, batch_size):
            idx = perm[i:i + batch_size]
            b_x = X_tr_t[idx].to(device)
            b_mask = mask_tr_t[idx].to(device)
            b_y = y_tr_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(b_x, mask=b_mask)
            
            if race_weights_t is not None:
                b_w = race_weights_t[idx].to(device)
                loss = (criterion(logits, b_y) * b_w).mean()
            else:
                loss = criterion(logits, b_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # 独立した Validation データ (X_val_meta) による Early Stopping 評価
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t.to(device), mask=mask_va_t.to(device))
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()

        payout_metrics = calculate_race_array_payout_metric(
            probs=val_probs,
            masks=race_val["race_masks"],
            y_true=y_va,
            odds=race_val["odds"],
            payouts=race_val["payouts"],
            selection_mode=selection_mode,
            min_prob=min_prob_for_ev
        )

        if eval_metric == "payout":
            current_score = payout_metrics["roi"]
        elif eval_metric == "hybrid":
            current_score = payout_metrics["hybrid_score"]
        else:
            current_score = payout_metrics["hit_rate"]

        scheduler.step(current_score)
        logger.info(
            f"[Meta Model] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Val Hit Rate: {payout_metrics['hit_rate']*100:.2f}% | Val ROI: {payout_metrics['roi']*100:.2f}% | "
            f"Eval Score ({eval_metric}): {current_score:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"[Meta Model] Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"[Meta Model] Best {eval_metric} Score: {best_score:.4f} (Saved to {best_model_path})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_score

def predict_race_meta_model(
    model: nn.Module,
    X_meta: np.ndarray,
    race_masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """
    メタNNによる最終勝率予測 (N_races, 18)
    """
    model.eval()
    all_probs = []
    num_races = len(X_meta)
    X_t = torch.tensor(X_meta, dtype=torch.float32)
    m_t = torch.tensor(race_masks, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, num_races, batch_size):
            bx = X_t[i:i + batch_size].to(device)
            bm = m_t[i:i + batch_size].to(device)
            logits = model(bx, mask=bm)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)
