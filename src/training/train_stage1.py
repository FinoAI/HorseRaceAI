import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from src.models.base_nn import DeepStage1NN
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta
from src.utils.helpers import setup_logger, get_device

logger = setup_logger("TrainStage1")

class RaceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, race_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.race_ids = race_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_race_softmax_loss(logits: torch.Tensor, targets: torch.Tensor, race_ids: np.ndarray, device: torch.device):
    """
    レースごとの出走馬グループに対するSoftmax Cross Entropy損失
    """
    unique_races = np.unique(race_ids)
    total_loss = 0.0
    count = 0
    
    for r in unique_races:
        mask = (race_ids == r)
        r_logits = logits[mask]
        r_targets = targets[mask]
        
        if r_targets.sum() == 0:
            continue
            
        # レース内の出走頭数でSoftmax
        log_probs = torch.log_softmax(r_logits, dim=0)
        # 1着馬のインデックス
        loss = -torch.sum(r_targets * log_probs)
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)
    return total_loss / count

def train_single_stage1_model(
    model_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    race_ids_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    race_ids_val: np.ndarray,
    multiplier: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 15,
    early_stopping_patience: int = 3,
    artifacts_dir: str = "./artifacts_models",
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """
    前段1モデルの学習を実行
    """
    if device is None:
        device = get_device()

    input_dim = X_train.shape[1]
    logger.info(f"--- [Model {model_idx}] Initializing DeepStage1NN: input_dim={input_dim}, multiplier={multiplier}, 1st layer={input_dim * multiplier} units ---")
    
    model = DeepStage1NN(input_dim=input_dim, multiplier=multiplier, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    best_val_hit_rate = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")

    # 学習用テンソル
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_va_t = torch.tensor(X_val, dtype=torch.float32)
    
    # バッチ処理（メモリ対策のためチャンク単位）
    batch_size = 4096
    num_samples = len(X_train)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_tr_t[indices].to(device), y_tr_t[indices].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation 評価 (Top-1 的中率)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for vi in range(0, len(X_val), batch_size):
                b_x = X_va_t[vi:vi + batch_size].to(device)
                l = model(b_x)
                val_preds.append(l.cpu().numpy())
        val_logits = np.concatenate(val_preds)
        
        # レースごとのTop1的中率計算
        df_val_eval = pd.DataFrame({
            "RACE_ID": race_ids_val,
            "logit": val_logits,
            "target": y_val
        })
        top1 = df_val_eval.sort_values(["RACE_ID", "logit"], ascending=[True, False]).groupby("RACE_ID").head(1)
        val_hit_rate = (top1["target"] == 1).mean() if len(top1) > 0 else 0.0

        scheduler.step(val_hit_rate)
        logger.info(f"[Model {model_idx}] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Top-1 Hit Rate: {val_hit_rate:.4f}")

        if val_hit_rate > best_val_hit_rate:
            best_val_hit_rate = val_hit_rate
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"[Model {model_idx}] Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"[Model {model_idx}] Best Val Top-1 Hit Rate: {best_val_hit_rate:.4f} (Saved to {best_model_path})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_val_hit_rate

def predict_stage1_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X_t[i:i + batch_size].to(device)
            l = model(bx)
            preds.append(torch.sigmoid(l).cpu().numpy())
    return np.concatenate(preds)
