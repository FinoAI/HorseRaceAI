import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.utils.helpers import setup_logger

logger = setup_logger("BetSimulation")

class BetSimulator:
    def __init__(self, unit_bet: int = 100):
        self.unit_bet = unit_bet

    def simulate_flat_bet(self, df_preds: pd.DataFrame) -> Dict[str, Any]:
        """
        全レースTop 1予測馬を単勝ベタ買いした場合の成績
        """
        top1 = df_preds.sort_values(["RACE_ID", "prob"], ascending=[True, False]).groupby("RACE_ID").head(1).copy()
        total_bets = len(top1) * self.unit_bet
        
        # 的中判定と払戻金計算
        hits = (top1["Target_着順"] == 1).sum()
        # Target_単勝 は 100円あたりの払戻金 (NaNなら 0)
        payouts = top1["Target_単勝"].fillna(0.0).sum()
        
        hit_rate = hits / len(top1) if len(top1) > 0 else 0.0
        roi = payouts / total_bets if total_bets > 0 else 0.0
        
        return {
            "strategy": "Flat Bet (Top 1)",
            "num_races": len(top1),
            "num_bets": len(top1),
            "total_investment": total_bets,
            "total_payout": payouts,
            "hits": int(hits),
            "hit_rate": float(hit_rate),
            "roi": float(roi)
        }

    def simulate_ev_threshold(
        self,
        df_preds: pd.DataFrame,
        ev_threshold: float = 1.1,
        min_prob: float = 0.15,
        top_only: bool = True
    ) -> Dict[str, Any]:
        """
        期待値 (EV = 予測確率 * 確定単勝オッズ) と最小確率閾値に基づく購入シミュレーション
        """
        df = df_preds.copy()
        odds = pd.to_numeric(df["Target_確定単勝オッズ"], errors="coerce").fillna(0.0)
        df["EV"] = df["prob"] * odds
        
        if top_only:
            # レース内Top1馬のみを候補とする
            candidates = df.sort_values(["RACE_ID", "prob"], ascending=[True, False]).groupby("RACE_ID").head(1).copy()
        else:
            candidates = df.copy()

        # フィルター条件
        selected = candidates[(candidates["EV"] >= ev_threshold) & (candidates["prob"] >= min_prob)].copy()
        
        num_bets = len(selected)
        total_investment = num_bets * self.unit_bet
        
        if num_bets == 0:
            return {
                "strategy": f"EV >= {ev_threshold} (min_prob={min_prob})",
                "ev_threshold": ev_threshold,
                "min_prob": min_prob,
                "num_bets": 0,
                "total_investment": 0,
                "total_payout": 0.0,
                "hits": 0,
                "hit_rate": 0.0,
                "roi": 0.0
            }

        hits = (selected["Target_着順"] == 1).sum()
        payouts = selected["Target_単勝"].fillna(0.0).sum()
        
        hit_rate = hits / num_bets
        roi = payouts / total_investment
        
        return {
            "strategy": f"EV >= {ev_threshold} (min_prob={min_prob})",
            "ev_threshold": ev_threshold,
            "min_prob": min_prob,
            "num_bets": num_bets,
            "total_investment": total_investment,
            "total_payout": payouts,
            "hits": int(hits),
            "hit_rate": float(hit_rate),
            "roi": float(roi)
        }

    def grid_search_strategies(
        self,
        df_preds: pd.DataFrame,
        ev_grid: List[float] = [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5],
        prob_grid: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30]
    ) -> pd.DataFrame:
        """
        グリッドサーチを行い、回収率100%超 & 的中率20%超の戦略を探索
        """
        results = []
        # ベタ買い
        results.append(self.simulate_flat_bet(df_preds))

        for ev in ev_grid:
            for prob in prob_grid:
                res = self.simulate_ev_threshold(df_preds, ev_threshold=ev, min_prob=prob, top_only=True)
                results.append(res)

        df_res = pd.DataFrame(results)
        return df_res
