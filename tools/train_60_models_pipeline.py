import os
import sys
import struct
import json
import random
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import setup_logger, set_seed, load_config, get_device, save_json
from src.models.base_nn import RaceLevelStage1NN
from src.training.train_stage1 import predict_race_stage1

logger = setup_logger("Train60Models")

torch.set_num_threads(8)

def save_meta_bin(meta_dict: Dict[str, Any], out_path: str):
    N = len(meta_dict["race_ids"])
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", N, 18))
        for i in range(N):
            y = int(meta_dict["y"][i])
            f.write(struct.pack("<i", y))
            f.write(meta_dict["masks"][i].astype(np.float32).tobytes())
            f.write(meta_dict["odds"][i].astype(np.float32).tobytes())
            f.write(meta_dict["payouts"][i].astype(np.float32).tobytes())
    logger.info(f"Saved metadata to {out_path} ({N} races)")

def sample_diverse_features(
    model_idx: int,
    bac_cols: List[str],
    yearly_cols: List[str],
    bac_count: int = 6,
    yearly_count: int = 4,
    random_seed: int = 42
) -> List[str]:
    rng = random.Random(random_seed + model_idx * 23)
    k_bac = min(bac_count, len(bac_cols))
    k_yr = min(yearly_count, len(yearly_cols))
    sel_bac = rng.sample(bac_cols, k_bac)
    sel_yr = rng.sample(yearly_cols, k_yr)
    return sel_bac + sel_yr

def main():
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Compute Device: {device} (PyTorch threads: {torch.get_num_threads()})")

    data_dir = cfg["data"]["data_dir"]
    data_c_dir = "data_c"
    os.makedirs(data_c_dir, exist_ok=True)
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. 競馬予想に真に有効な特徴量プール（禁止カラム・リークカラム除外済み）
    logger.info("=== Initializing Strong Predictive Feature Pool ===")
    
    # 競走馬個体の能力・前走成績・指数・陣営評価（27列の強力な候補プール）
    strong_horse_candidates = [
        "KYI_騎手指数", "KYI_情報指数", "KYI_総合指数", "KYI_脚質", "KYI_距離適性", "KYI_上昇度", "KYI_ローテーション",
        "KYI_調教指数", "KYI_厩舎指数", "KYI_テン指数", "KYI_ペース指数", "KYI_上がり指数", "KYI_位置指数",
        "KYI_総合情報◎", "KYI_総合情報○", "KYI_総合情報▲", "KYI_総合情報△",
        "KYI_特定情報◎", "KYI_特定情報○", "KYI_特定情報▲",
        "SUBV1_着順", "SUBV1_確定単勝オッズ", "SUBV1_頭数",
        "SUBV2_着順", "SUBV2_確定単勝オッズ",
        "SUBV3_着順", "SUBV3_確定単勝オッズ"
    ]

    # BAC_KAB レース環境特徴量プール (20列)
    df_bac = pd.read_csv(os.path.join(data_dir, "BAC_KAB.csv"), dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)
    
    bac_cands = [
        "距離", "芝ダ障害コード", "右左", "内外", "条件", "重量", "グレード",
        "芝馬場状態コード", "ダ馬場状態コード", "天候コード", "連続何日目",
        "直線馬場差最内", "直線馬場差内", "直線馬場差中", "直線馬場差外", "直線馬場差大外"
    ]
    logger.info(f"Available features: BAC={len(bac_cands)}, Horse-Level={len(strong_horse_candidates)}")

    num_total_models = 60
    models_features = {}
    needed_yearly_cols = set([
        "KYI_RACE_KEY", "KYI_馬番", "Target_着順", "Target_確定単勝オッズ", "Target_単勝"
    ])

    for m in range(1, num_total_models + 1):
        feats = sample_diverse_features(m, bac_cands, strong_horse_candidates, bac_count=6, yearly_count=4)
        models_features[m] = feats
        save_json({"model_idx": m, "features": feats}, os.path.join(artifacts_dir, f"features_model_{m}.json"))
        for f_name in feats:
            if f_name in strong_horse_candidates:
                needed_yearly_cols.add(f_name)

    logger.info(f"Pre-selected 60 models features. Unique yearly columns to load: {len(needed_yearly_cols)}")

    # 2. データのロード (Train: 2010-2023, Val & Test: 2024-2025)
    train_years = cfg["data"]["train_years"] # [2010..2023]
    val_test_years = cfg["data"]["val_test_years"] # [2024, 2025]
    logger.info(f"=== Loading Train Data ({train_years[0]}-{train_years[-1]}) and Val/Test ({val_test_years}) ===")

    use_cols_list = list(needed_yearly_cols)

    # 2.1 Train データ (2010-2023)
    train_dfs = []
    t_start = time.time()
    for y in train_years:
        fpath = os.path.join(data_dir, f"{y}.csv")
        df_y = pd.read_csv(fpath, usecols=use_cols_list, dtype={"KYI_RACE_KEY": str}, low_memory=False)
        df_y["RACE_ID"] = df_y["KYI_RACE_KEY"].astype(str).str.zfill(10).str[:8]
        train_dfs.append(df_y)
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c in bac_cands]
    df_train = pd.merge(df_train_raw, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
    del train_dfs, df_train_raw
    logger.info(f"Loaded Train: shape={df_train.shape}, races={df_train['RACE_ID'].nunique()} in {time.time()-t_start:.1f}s")

    # 2.2 Val & Test データ (2024-2025)
    t_vt = time.time()
    val_test_dfs = []
    for y in val_test_years:
        fpath = os.path.join(data_dir, f"{y}.csv")
        df_y = pd.read_csv(fpath, usecols=use_cols_list, dtype={"KYI_RACE_KEY": str}, low_memory=False)
        df_y["RACE_ID"] = df_y["KYI_RACE_KEY"].astype(str).str.zfill(10).str[:8]
        val_test_dfs.append(df_y)
    df_val_test_raw = pd.concat(val_test_dfs, ignore_index=True)
    df_val_test = pd.merge(df_val_test_raw, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
    del val_test_dfs, df_val_test_raw

    # 特徴量カラムを一括で float32 に変換してキャッシュ
    logger.info("Converting all selected feature columns to float32...")
    all_needed_feats = set()
    for m in range(1, num_total_models + 1):
        all_needed_feats.update(models_features[m])
    
    for c in all_needed_feats:
        if c in df_train.columns:
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(0.0).astype(np.float32)
        if c in df_val_test.columns:
            df_val_test[c] = pd.to_numeric(df_val_test[c], errors="coerce").fillna(0.0).astype(np.float32)

    # レース単位でシャッフル分割 (Val 50% / Test 50%)
    all_vt_races = np.array(df_val_test["RACE_ID"].drop_duplicates().tolist())
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(all_vt_races)

    n_val_races = int(len(all_vt_races) * cfg["data"]["val_ratio"])
    val_race_ids = list(all_vt_races[:n_val_races])
    test_race_ids = list(all_vt_races[n_val_races:])

    df_val = df_val_test[df_val_test["RACE_ID"].isin(set(val_race_ids))].reset_index(drop=True)
    df_test = df_val_test[df_val_test["RACE_ID"].isin(set(test_race_ids))].reset_index(drop=True)
    logger.info(f"Loaded Val/Test in {time.time()-t_vt:.1f}s: Val={len(val_race_ids)} races, Test={len(test_race_ids)} races")

    max_horses = 18

    # 3. Val / Test の固定スロットインデックスおよびメタデータを事前構築
    def build_meta_and_indices(df: pd.DataFrame, ordered_race_ids: List[str]):
        num_races = len(ordered_race_ids)
        r_map = {r: i for i, r in enumerate(ordered_race_ids)}
        r_idx = np.array([r_map.get(r, -1) for r in df["RACE_ID"].values], dtype=np.int32)
        horse_nums = pd.to_numeric(df["KYI_馬番"], errors="coerce").fillna(0).values.astype(np.int32)
        slots = horse_nums - 1
        valid = (r_idx >= 0) & (slots >= 0) & (slots < max_horses)

        masks = np.zeros((num_races, max_horses), dtype=np.float32)
        odds = np.zeros((num_races, max_horses), dtype=np.float32)
        payouts = np.zeros((num_races, max_horses), dtype=np.float32)
        y = np.zeros(num_races, dtype=np.int64)

        vr = r_idx[valid]
        vs = slots[valid]
        masks[vr, vs] = 1.0

        raw_odds = pd.to_numeric(df["Target_確定単勝オッズ"], errors="coerce").fillna(0.0).values.astype(np.float32)
        odds[vr, vs] = raw_odds[valid]

        raw_pay = pd.to_numeric(df["Target_単勝"], errors="coerce").fillna(0.0).values.astype(np.float32)
        payouts[vr, vs] = raw_pay[valid]

        orders = pd.to_numeric(df["Target_着順"], errors="coerce").fillna(99).values.astype(np.int32)
        win_m = (orders[valid] == 1)
        y[vr[win_m]] = vs[win_m]

        return {
            "race_ids": ordered_race_ids,
            "y": y,
            "masks": masks,
            "odds": odds,
            "payouts": payouts,
            "valid": valid,
            "vr": vr,
            "vs": vs
        }

    val_meta = build_meta_and_indices(df_val, val_race_ids)
    test_meta = build_meta_and_indices(df_test, test_race_ids)

    save_meta_bin(val_meta, os.path.join(data_c_dir, "val_meta.bin"))
    save_meta_bin(test_meta, os.path.join(data_c_dir, "test_meta.bin"))

    # 4. 60モデルの高速学習 & 予測出力
    logger.info(f"\n=== Training {num_total_models} Diverse Models (2010-2023 Train -> 2024-2025 Val/Test) ===")
    all_train_races = np.array(df_train["RACE_ID"].drop_duplicates().tolist())
    logger.info(f"Total available train races: {len(all_train_races)}")

    all_val_preds = []
    all_test_preds = []

    samples_per_model = 2500 # 各モデル2500レース (2010-2023年をカバー)
    batch_size = 256

    for m_idx in range(1, num_total_models + 1):
        t_m0 = time.time()
        feats = models_features[m_idx]
        D = len(feats)

        # モデルごとの特徴量標準化統計量
        f_tr_sub = df_train[feats].values
        means = np.mean(f_tr_sub, axis=0)
        stds = np.std(f_tr_sub, axis=0)
        stds[stds == 0] = 1.0

        # モデルごとのブートストラップレースサンプリング (2010-2023から2500レース)
        m_rng = np.random.default_rng(cfg["data"]["random_seed"] + m_idx * 17)
        m_train_race_ids = m_rng.choice(all_train_races, size=samples_per_model, replace=False).tolist()
        df_train_m = df_train[df_train["RACE_ID"].isin(set(m_train_race_ids))]

        tr_meta = build_meta_and_indices(df_train_m, m_train_race_ids)
        norm_tr = (df_train_m[feats].values - means) / stds
        race_tr_X_3d = np.zeros((samples_per_model, max_horses, D), dtype=np.float32)
        race_tr_X_3d[tr_meta["vr"], tr_meta["vs"], :] = norm_tr[tr_meta["valid"]]
        race_tr_X = race_tr_X_3d.reshape(samples_per_model, max_horses * D)

        # Val / Test 特徴量テンソルの瞬間生成
        norm_va = (df_val[feats].values - means) / stds
        race_va_X_3d = np.zeros((len(val_race_ids), max_horses, D), dtype=np.float32)
        race_va_X_3d[val_meta["vr"], val_meta["vs"], :] = norm_va[val_meta["valid"]]
        race_va_X = race_va_X_3d.reshape(len(val_race_ids), max_horses * D)

        norm_te = (df_test[feats].values - means) / stds
        race_te_X_3d = np.zeros((len(test_race_ids), max_horses, D), dtype=np.float32)
        race_te_X_3d[test_meta["vr"], test_meta["vs"], :] = norm_te[test_meta["valid"]]
        race_te_X = race_te_X_3d.reshape(len(test_race_ids), max_horses * D)

        # モデル定義 (180次元 -> 5760ユニット -> ピラミッド8層)
        model = RaceLevelStage1NN(
            input_dim=max_horses * D,
            max_horses=max_horses,
            multiplier=32,
            reduction_ratio=0.5,
            dropout=0.2
        ).to(device)

        # 高速学習 (2 epochs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(race_tr_X, dtype=torch.float32)
        m_tr_t = torch.tensor(tr_meta["masks"], dtype=torch.float32)
        y_tr_t = torch.tensor(tr_meta["y"], dtype=torch.long)

        model.train()
        for ep in range(2):
            perm = torch.randperm(samples_per_model)
            for i in range(0, samples_per_model, batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                out = model(X_tr_t[idx].to(device), m_tr_t[idx].to(device))
                loss = criterion(out, y_tr_t[idx].to(device))
                loss.backward()
                optimizer.step()

        # 推論 (Val: 3454 races, Test: 3455 races)
        model.eval()
        with torch.no_grad():
            v_out = predict_race_stage1(model, race_va_X, val_meta["masks"], device=device)
            t_out = predict_race_stage1(model, race_te_X, test_meta["masks"], device=device)

        all_val_preds.append(v_out.astype(np.float32))
        all_test_preds.append(t_out.astype(np.float32))

        # 保存
        torch.save(model.state_dict(), os.path.join(artifacts_dir, f"model_{m_idx}.pth"))

        if m_idx % 5 == 0 or m_idx == 1:
            logger.info(f"[Progress] Completed Model {m_idx}/{num_total_models} in {time.time()-t_m0:.2f}s")

    # 5. C言語用の予測テンソルバイナリの保存
    val_preds_arr = np.stack(all_val_preds, axis=0) # (60, N_val, 18)
    test_preds_arr = np.stack(all_test_preds, axis=0) # (60, N_test, 18)

    val_preds_bin_path = os.path.join(data_c_dir, "val_preds_60.bin")
    test_preds_bin_path = os.path.join(data_c_dir, "test_preds_60.bin")

    with open(val_preds_bin_path, "wb") as f:
        f.write(struct.pack("<III", num_total_models, len(val_race_ids), max_horses))
        f.write(val_preds_arr.tobytes())

    with open(test_preds_bin_path, "wb") as f:
        f.write(struct.pack("<III", num_total_models, len(test_race_ids), max_horses))
        f.write(test_preds_arr.tobytes())

    logger.info(f"Successfully generated C-ready predictions: {val_preds_bin_path} and {test_preds_bin_path}")
    logger.info(f"Val shape: {val_preds_arr.shape}, Test shape: {test_preds_arr.shape}")

if __name__ == "__main__":
    main()
