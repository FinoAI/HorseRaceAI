import numpy as np
import pandas as pd
from typing import Dict

def calculate_top1_hit_rate(df_preds: pd.DataFrame) -> Dict[str, float]:
    """
    各レースで予測確率（またはスコア）が1位の馬を購入した場合の単勝的中率を算出
    df_preds must contain: ['RACE_ID', 'prob', 'Target_着順']
    """
    # レースごとに予測確率最大の馬を抽出
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
