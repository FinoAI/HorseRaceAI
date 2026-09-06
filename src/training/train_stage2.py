import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
from src.models.meta_nn import DeepMetaNN
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
    X_val_meta: np.ndarray,
    y_val: np.ndarray,
    race_ids_val: np.ndarray,
    multiplier: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.00005,
    epochs: int = 20,
    early_stopping_patience: int = 4,
    artifacts_dir: str = "./artifacts_models",
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """
    後段メタNN（最終予想モデル）の学習
    """
    if device is None:
        device = get_device()

    input_dim = X_val_meta.shape[1]
    logger.info(f"--- [Meta Model] Initializing DeepMetaNN: input_dim={input_dim}, multiplier={multiplier}, 1st layer={input_dim * multiplier} units ---")

    model = DeepMetaNN(input_dim=input_dim, multiplier=multiplier, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    best_val_hit_rate = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, "meta_model.pth")

    X_t = torch.tensor(X_val_meta, dtype=torch.float32)
    y_t = torch.tensor(y_val, dtype=torch.float32)

    num_samples = len(X_val_meta)
    batch_size = 2048

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            bx, by = X_t[indices].to(device), y_t[indices].to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # 評価
        model.eval()
        with torch.no_grad():
            full_logits = model(X_t.to(device)).cpu().numpy()

        df_eval = pd.DataFrame({
            "RACE_ID": race_ids_val,
            "logit": full_logits,
            "target": y_val
        })
        top1 = df_eval.sort_values(["RACE_ID", "logit"], ascending=[True, False]).groupby("RACE_ID").head(1)
        val_hit_rate = (top1["target"] == 1).mean() if len(top1) > 0 else 0.0

        scheduler.step(val_hit_rate)
        logger.info(f"[Meta Model] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Top-1 Hit Rate: {val_hit_rate:.4f}")

        if val_hit_rate > best_val_hit_rate:
            best_val_hit_rate = val_hit_rate
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"[Meta Model] Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"[Meta Model] Best Val Top-1 Hit Rate: {best_val_hit_rate:.4f} (Saved to {best_model_path})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_val_hit_rate

def predict_meta_model(model: nn.Module, X_meta: np.ndarray, race_ids: np.ndarray, device: torch.device) -> np.ndarray:
    """
    メタNNによる最終予測。各レース内でSoftmaxを取り、勝率（合計1.0）へ正規化。
    """
    model.eval()
    X_t = torch.tensor(X_meta, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t.to(device)).cpu().numpy()

    # 高速・安全なレース内Softmax計算
    probs = np.zeros_like(logits, dtype=np.float32)
    unique_races = np.unique(race_ids)
    
    for r in unique_races:
        idx = np.where(race_ids == r)[0]
        r_logits = logits[idx]
        exp_v = np.exp(r_logits - np.max(r_logits))
        r_prob = exp_v / np.sum(exp_v)
        probs[idx] = r_prob

    return probs
