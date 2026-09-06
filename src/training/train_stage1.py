import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from src.models.base_nn import DeepStage1NN
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta
from src.evaluation.metrics import calculate_payout_metric, calculate_top1_hit_rate
from src.utils.helpers import setup_logger, get_device

logger = setup_logger("TrainStage1")

def train_single_stage1_model(
    model_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_val: pd.DataFrame,
    multiplier: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 15,
    early_stopping_patience: int = 3,
    eval_metric: str = "payout", # "payout" | "hit_rate" | "hybrid"
    selection_mode: str = "ev_filtered", # "prob" | "ev" | "ev_filtered"
    min_prob_for_ev: float = 0.10,
    loss_weighting: str = "none",
    artifacts_dir: str = "./artifacts_models",
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """
    前段1モデルの学習を実行 (払戻金最大化 / 期待値最大化オプション対応)
    """
    if device is None:
        device = get_device()

    input_dim = X_train.shape[1]
    logger.info(f"--- [Model {model_idx}] DeepStage1NN: input_dim={input_dim}, 1st layer={input_dim * multiplier} | Eval={eval_metric} (mode={selection_mode}) ---")
    
    model = DeepStage1NN(input_dim=input_dim, multiplier=multiplier, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # 配当重み付き損失の準備
    if loss_weighting == "payout":
        odds = pd.to_numeric(meta_train["Target_確定単勝オッズ"], errors="coerce").fillna(1.0).values
        # 1着かつ高配当のサンプルに重みを付与 (log1pで急激な発散を抑制)
        sample_weights = np.where(y_train == 1, 1.0 + np.log1p(odds), 1.0).astype(np.float32)
        sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        sample_weights_t = None
        criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_va_t = torch.tensor(X_val, dtype=torch.float32)
    
    batch_size = 4096
    num_samples = len(X_train)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_tr_t[indices].to(device)
            batch_y = y_tr_t[indices].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            
            if sample_weights_t is not None:
                b_weights = sample_weights_t[indices].to(device)
                loss = (criterion(logits, batch_y) * b_weights).mean()
            else:
                loss = criterion(logits, batch_y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation 評価
        model.eval()
        val_preds = []
        with torch.no_grad():
            for vi in range(0, len(X_val), batch_size):
                b_x = X_va_t[vi:vi + batch_size].to(device)
                l = model(b_x)
                val_preds.append(torch.sigmoid(l).cpu().numpy())
        val_probs = np.concatenate(val_preds)
        
        df_val_eval = meta_val.copy()
        df_val_eval["prob"] = val_probs

        # 評価指標の算出
        payout_metrics = calculate_payout_metric(
            df_val_eval,
            selection_mode=selection_mode,
            min_prob=min_prob_for_ev
        )

        if eval_metric == "payout":
            current_score = payout_metrics["roi"] # 回収率 (払戻金/投資額)
        elif eval_metric == "hybrid":
            current_score = payout_metrics["hybrid_score"] # 回収率 × 的中率ペナルティ
        else:
            current_score = payout_metrics["hit_rate"] # 的中率

        scheduler.step(current_score)
        logger.info(
            f"[Model {model_idx}] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Hit Rate: {payout_metrics['hit_rate']*100:.2f}% | ROI: {payout_metrics['roi']*100:.2f}% | "
            f"Eval Score ({eval_metric}): {current_score:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"[Model {model_idx}] Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"[Model {model_idx}] Best {eval_metric} Score: {best_score:.4f} (Saved to {best_model_path})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_score

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
