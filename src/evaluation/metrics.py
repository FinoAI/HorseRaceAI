import numpy as np
import pandas as pd
from typing import Dict, Any, Union

def calculate_top1_hit_rate(df_preds: pd.DataFrame) -> Dict[str, float]:
    """
    各レースで予測確率（またはスコア）が1位の馬を購入した場合の単勝的中率を算出
    df_preds must contain: ['RACE_ID', 'prob', 'Target_着順']
    """
    top1_horses = df_preds.sort_values(["RACE_ID", "prob"], ascending=[True, False]).groupby("RACE_ID").head(1)
    
    total_races = len(top1_horses)
    if total_races == 0:
        return {"total_races": 0, "hits": 0, "hit_rate": 0.0}
    
    hits = (top1_horses["Target_着順"] == 1).sum()
    hit_rate = hits / total_races
    return {
        "total_races": int(total_races),
        "hits": int(hits),
        "hit_rate": float(hit_rate)
    }

def calculate_payout_metric(
    df_preds: pd.DataFrame,
    selection_mode: str = "prob",
    min_prob: float = 0.10,
    unit_bet: int = 100
) -> Dict[str, Any]:
    """
    Target_単勝 * (Target_着順 == 1 ? 1 : 0) による払戻金最大化評価関数 (DataFrame版)
    """
    df = df_preds.copy()
    odds = pd.to_numeric(df.get("Target_確定単勝オッズ", 0), errors="coerce").fillna(0.0)
    df["odds"] = odds
    df["EV"] = df["prob"] * odds
    
    if selection_mode == "ev":
        selected = df.sort_values(["RACE_ID", "EV"], ascending=[True, False]).groupby("RACE_ID").head(1).copy()
    elif selection_mode == "ev_filtered":
        filtered = df[df["prob"] >= min_prob]
        selected_filtered = filtered.sort_values(["RACE_ID", "EV"], ascending=[True, False]).groupby("RACE_ID").head(1)
        missing_races = set(df["RACE_ID"].unique()) - set(selected_filtered["RACE_ID"].unique())
        if missing_races:
            fallback = df[df["RACE_ID"].isin(missing_races)].sort_values(["RACE_ID", "prob"], ascending=[True, False]).groupby("RACE_ID").head(1)
            selected = pd.concat([selected_filtered, fallback], ignore_index=True)
        else:
            selected = selected_filtered
    else: # "prob"
        selected = df.sort_values(["RACE_ID", "prob"], ascending=[True, False]).groupby("RACE_ID").head(1).copy()

    total_races = len(selected)
    if total_races == 0:
        return {
            "selection_mode": selection_mode,
            "total_races": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "total_investment": 0,
            "total_payout": 0.0,
            "roi": 0.0,
            "hybrid_score": 0.0
        }

    is_winner = (selected["Target_着順"] == 1)
    raw_payout = pd.to_numeric(selected.get("Target_単勝", 0.0), errors="coerce").fillna(0.0)
    effective_payouts = np.where(is_winner, raw_payout, 0.0)
    total_payout = float(np.sum(effective_payouts))
    total_investment = total_races * unit_bet

    hits = int(is_winner.sum())
    hit_rate = hits / total_races
    roi = total_payout / total_investment if total_investment > 0 else 0.0
    hit_penalty = min(1.0, hit_rate / 0.20)
    hybrid_score = roi * hit_penalty

    return {
        "selection_mode": selection_mode,
        "total_races": total_races,
        "hits": hits,
        "hit_rate": float(hit_rate),
        "total_investment": total_investment,
        "total_payout": float(total_payout),
        "roi": float(roi),
        "hybrid_score": float(hybrid_score)
    }

def calculate_race_array_payout_metric(
    probs: np.ndarray,      # shape (N_races, 18)
    masks: np.ndarray,      # shape (N_races, 18)
    y_true: np.ndarray,     # shape (N_races,) 1着馬スロットインデックス
    odds: np.ndarray,       # shape (N_races, 18)
    payouts: np.ndarray,    # shape (N_races, 18)
    selection_mode: str = "prob",
    min_prob: float = 0.10,
    unit_bet: int = 100
) -> Dict[str, Any]:
    """
    レース単位の配列 (N_races, 18) から直接高速に払戻金最大化指標を計算
    """
    num_races = len(probs)
    if num_races == 0:
        return {"total_races": 0, "hits": 0, "hit_rate": 0.0, "total_investment": 0, "total_payout": 0.0, "roi": 0.0, "hybrid_score": 0.0}

    # 非出走馬のスコアを無効化
    valid_probs = np.where(masks > 0, probs, -1.0)
    ev_matrix = valid_probs * odds

    selected_slots = np.zeros(num_races, dtype=int)
    for i in range(num_races):
        r_probs = valid_probs[i]
        r_ev = ev_matrix[i]
        
        if selection_mode == "ev":
            selected_slots[i] = np.argmax(r_ev)
        elif selection_mode == "ev_filtered":
            qualifying = np.where((r_probs >= min_prob) & (masks[i] > 0))[0]
            if len(qualifying) > 0:
                best_idx = qualifying[np.argmax(r_ev[qualifying])]
                selected_slots[i] = best_idx
            else:
                selected_slots[i] = np.argmax(r_probs)
        else: # "prob"
            selected_slots[i] = np.argmax(r_probs)

    hits = (selected_slots == y_true)
    hit_count = int(hits.sum())
    hit_rate = hit_count / num_races

    # 的中したレースのみ payout を獲得
    obtained_payouts = np.where(hits, payouts[np.arange(num_races), selected_slots], 0.0)
    total_payout = float(np.sum(obtained_payouts))
    total_investment = num_races * unit_bet
    roi = total_payout / total_investment if total_investment > 0 else 0.0

    hit_penalty = min(1.0, hit_rate / 0.20)
    hybrid_score = roi * hit_penalty

    return {
        "selection_mode": selection_mode,
        "total_races": num_races,
        "hits": hit_count,
        "hit_rate": float(hit_rate),
        "total_investment": total_investment,
        "total_payout": float(total_payout),
        "roi": float(roi),
        "hybrid_score": float(hybrid_score)
    }
