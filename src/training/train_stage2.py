import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
from src.models.meta_nn import DeepMetaNN
from src.evaluation.metrics import calculate_payout_metric
from src.utils.helpers import setup_logger, get_device

logger = setup_logger("TrainStage2")

def build_meta_features(preds_list: list) -> np.ndarray:
    """
    10個の前段モデルの予測（各馬の予測確率）からメタ特徴量を構築
    - 各モデルの予測値 (10次元)
    - 統計特徴: 平均、標準偏差、最大値、最小値 (4次元)
    合計 14次元
    """
    P = np.column_stack(preds_list)
    
    mean_val = np.mean(P, axis=1, keepdims=True)
    std_val = np.std(P, axis=1, keepdims=True)
    max_val = np.max(P, axis=1, keepdims=True)
    min_val = np.min(P, axis=1, keepdims=True)
    
    meta_features = np.hstack([P, mean_val, std_val, max_val, min_val])
    return meta_features.astype(np.float32)

def train_meta_model(
    X_train_meta: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_val_meta: np.ndarray,
    y_val: np.ndarray,
    meta_val: pd.DataFrame,
    multiplier: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 0.0005,
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
    後段メタNN（最終予想モデル）の学習
    - X_train_meta でモデルを学習し、完全に独立した X_val_meta で Early Stopping を判定（データリーク・過学習を防止）
    - loss_weighting="payout" による配当重み付き損失に対応
    """
    if device is None:
        device = get_device()

    input_dim = X_train_meta.shape[1]
    logger.info(
        f"--- [Meta Model] DeepMetaNN: input_dim={input_dim}, 1st layer={input_dim * multiplier} | "
        f"Eval={eval_metric} (mode={selection_mode}) | loss_weighting={loss_weighting} ---"
    )

    model = DeepMetaNN(input_dim=input_dim, multiplier=multiplier, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # 配当重み付き損失の準備 (Stage 2)
    if loss_weighting == "payout":
        odds = pd.to_numeric(meta_train.get("Target_確定単勝オッズ", 1.0), errors="coerce").fillna(1.0).values
        sample_weights = np.where(y_train == 1, 1.0 + np.log1p(odds), 1.0).astype(np.float32)
        sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        sample_weights_t = None
        criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, "meta_model.pth")

    X_tr_t = torch.tensor(X_train_meta, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)

    num_samples = len(X_train_meta)
    batch_size = 2048

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            bx = X_tr_t[indices].to(device)
            by = y_tr_t[indices].to(device)

            optimizer.zero_grad()
            logits = model(bx)
            
            if sample_weights_t is not None:
                b_weights = sample_weights_t[indices].to(device)
                loss = (criterion(logits, by) * b_weights).mean()
            else:
                loss = criterion(logits, by)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # 独立した Validation データ (X_val_meta) による Early Stopping 評価
        probs = predict_meta_model(model, X_val_meta, meta_val["RACE_ID"].values, device=device)
        df_eval = meta_val.copy()
        df_eval["prob"] = probs

        payout_metrics = calculate_payout_metric(
            df_eval,
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

def predict_meta_model(model: nn.Module, X_meta: np.ndarray, race_ids: np.ndarray, device: torch.device) -> np.ndarray:
    """
    メタNNによる最終予測。各レース内でSoftmaxを取り、勝率（合計1.0）へ正規化。
    """
    model.eval()
    X_t = torch.tensor(X_meta, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t.to(device)).cpu().numpy()

    probs = np.zeros_like(logits, dtype=np.float32)
    unique_races = np.unique(race_ids)
    
    for r in unique_races:
        idx = np.where(race_ids == r)[0]
        r_logits = logits[idx]
        exp_v = np.exp(r_logits - np.max(r_logits))
        r_prob = exp_v / np.sum(exp_v)
        probs[idx] = r_prob

    return probs
