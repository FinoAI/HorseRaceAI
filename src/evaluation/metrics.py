import numpy as np
import pandas as pd
from typing import Dict, Any

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
    Target_単勝 * (Target_着順 == 1 ? 1 : 0) による払戻金最大化評価関数
    
    selection_mode:
      - 'prob': 予測確率 P が最大の馬を選択
      - 'ev': 期待値 EV = P * 確定オッズ が最大の馬を選択 (高配当狙い)
      - 'ev_filtered': P >= min_prob を満たす馬の中で EV が最大の馬を選択 (的中率担保+高配当)
    """
    df = df_preds.copy()
    odds = pd.to_numeric(df.get("Target_確定単勝オッズ", 0), errors="coerce").fillna(0.0)
    df["odds"] = odds
    df["EV"] = df["prob"] * odds
    
    if selection_mode == "ev":
        # 期待値最大の馬を選択
        selected = df.sort_values(["RACE_ID", "EV"], ascending=[True, False]).groupby("RACE_ID").head(1).copy()
    elif selection_mode == "ev_filtered":
        # P >= min_prob の中で EV 最大の馬を選択。全馬基準未満なら prob 最大馬を代替選択
        filtered = df[df["prob"] >= min_prob]
        selected_filtered = filtered.sort_values(["RACE_ID", "EV"], ascending=[True, False]).groupby("RACE_ID").head(1)
        
        # 該当がないレースは prob 最大馬でフォールバック
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

    # 着順が1以外の馬は払戻金 0
    is_winner = (selected["Target_着順"] == 1)
    raw_payout = pd.to_numeric(selected.get("Target_単勝", 0.0), errors="coerce").fillna(0.0)
    
    # 払戻金額 = Target_単勝 * (Target_着順 == 1)
    effective_payouts = np.where(is_winner, raw_payout, 0.0)
    total_payout = float(np.sum(effective_payouts))
    total_investment = total_races * unit_bet

    hits = int(is_winner.sum())
    hit_rate = hits / total_races
    roi = total_payout / total_investment if total_investment > 0 else 0.0

    # 的中率20%を満たしているかを考慮したハイブリッド指標
    # 的中率が20%未満の場合は比例ペナルティ
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
