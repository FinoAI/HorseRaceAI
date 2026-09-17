import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric
from src.utils.helpers import setup_logger, get_device

logger = setup_logger("TrainStage1")

def train_single_stage1_model(
    model_idx: int,
    race_train: Dict[str, Any],
    race_val: Dict[str, Any],
    max_horses: int = 18,
    multiplier: int = 32,
    reduction_ratio: float = 0.5,
    dropout: float = 0.2,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.0001,
    epochs: int = 15,
    early_stopping_patience: int = 3,
    eval_metric: str = "payout", # "payout" | "hit_rate" | "hybrid"
    selection_mode: str = "ev_filtered", # "prob" | "ev" | "ev_filtered"
    min_prob_for_ev: float = 0.10,
    loss_weighting: str = "none", # "none" | "payout"
    artifacts_dir: str = "./artifacts_models",
    device: torch.device = None
) -> Tuple[nn.Module, float]:
    """
    レース単位入力型 前段NNモデル (Model idx) の学習
    """
    if device is None:
        device = get_device()

    X_tr = race_train["race_X"]
    mask_tr = race_train["race_masks"]
    y_tr = race_train["race_y"]

    X_va = race_val["race_X"]
    mask_va = race_val["race_masks"]
    y_va = race_val["race_y"]

    input_dim = X_tr.shape[1]
    feature_dim = input_dim // max_horses
    logger.info(
        f"--- [Model {model_idx}] RaceLevelStage1NN: input_dim={input_dim} (18 × {feature_dim} feats) | "
        f"1st layer={input_dim * multiplier:,} units (x{multiplier}) | 1/2 reduction ratio={reduction_ratio} | "
        f"Eval={eval_metric} (mode={selection_mode}) ---"
    )

    model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=multiplier,
        reduction_ratio=reduction_ratio,
        dropout=dropout
    ).to(device)

    logger.info(f"[Model {model_idx}] Total layers: {model.total_layers} (5+ layers deep architecture)")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # 配当重み付き損失の準備
    if loss_weighting == "payout":
        # 1着馬のオッズを取得
        winner_odds = race_train["odds"][np.arange(len(y_tr)), y_tr]
        race_weights = 1.0 + np.log1p(np.maximum(1.0, winner_odds)).astype(np.float32)
        race_weights_t = torch.tensor(race_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        race_weights_t = None
        criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    mask_tr_t = torch.tensor(mask_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64)

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    mask_va_t = torch.tensor(mask_va, dtype=torch.float32)

    num_races = len(X_tr)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation 評価
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t.to(device), mask=mask_va_t.to(device))
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()

        payout_metrics = calculate_race_array_payout_metric(
            probs=val_probs,
            masks=mask_va,
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
            f"[Model {model_idx}] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
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
                logger.info(f"[Model {model_idx}] Early stopping triggered at epoch {epoch}")
                break

    logger.info(f"[Model {model_idx}] Best {eval_metric} Score: {best_score:.4f} (Saved to {best_model_path})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, best_score

def predict_race_stage1(
    model: nn.Module,
    race_X: np.ndarray,
    race_masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """
    レース単位の勝率配列 (N_races, 18) を推論
    """
    model.eval()
    all_probs = []
    num_races = len(race_X)
    X_t = torch.tensor(race_X, dtype=torch.float32)
    m_t = torch.tensor(race_masks, dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, num_races, batch_size):
            bx = X_t[i:i + batch_size].to(device)
            bm = m_t[i:i + batch_size].to(device)
            logits = model(bx, mask=bm)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)
