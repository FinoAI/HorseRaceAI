"""
migrate_pths_to_gdrive.py
========================================================================
全ての .pth ファイルを Google Drive に安全に退避し、
上位15アンサンブルで使用するモデルのみをローカルに保持するスクリプト
========================================================================
"""

import os
import sys
import glob
import shutil
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import setup_logger

logger = setup_logger("MigratePthsToGDrive")

GDRIVE_BASE = "/Users/hino/Google Drive/マイドライブ/KeibaData_Archive/HorseRaceAI"

def main():
    if not os.path.exists(GDRIVE_BASE):
        logger.error(f"Google Drive archive path not found: {GDRIVE_BASE}")
        return

    # 1. 上位15アンサンブルで使われているモデルIDを取得
    super_ens_csv = os.path.join(CURRENT_DIR, "artifacts_turf_1win/super_ensemble_val_test_kuroji.csv")
    df = pd.read_csv(super_ens_csv)
    top15 = df.head(15)

    keep_model_ids = set()
    for combo in top15["combo"]:
        for p in combo.split("+"):
            keep_model_ids.add(int(p.replace("M", "")))

    keep_model_ids = sorted(list(keep_model_ids))
    logger.info(f"Top 15 ensembles use {len(keep_model_ids)} models: {keep_model_ids}")

    # ローカル保持対象のパス（正規化された絶対パス）
    keep_paths = set()
    for m in keep_model_ids:
        p = os.path.abspath(os.path.join(CURRENT_DIR, f"artifacts_turf_1win/model_{m}.pth"))
        keep_paths.add(p)

    # ダートBESTモデルも念のため保持
    dirt_best_m24 = os.path.abspath(os.path.join(CURRENT_DIR, "artifacts_dirt_1win/BEST_MODEL_DIRT_1WIN/model_24.pth"))
    dirt_best_m49 = os.path.abspath(os.path.join(CURRENT_DIR, "artifacts_dirt_1win/BEST_MODEL_DIRT_1WIN/model_49.pth"))
    if os.path.exists(dirt_best_m24):
        keep_paths.add(dirt_best_m24)
    if os.path.exists(dirt_best_m49):
        keep_paths.add(dirt_best_m49)

    logger.info(f"Local KEEP target files count: {len(keep_paths)}")

    # 2. 全 .pth ファイルを走査
    all_pths = sorted(glob.glob(os.path.join(CURRENT_DIR, "**/*.pth"), recursive=True))
    all_pths = [os.path.abspath(p) for p in all_pths]
    total_files = len(all_pths)
    total_bytes = sum(os.path.getsize(p) for p in all_pths)
    total_gb = total_bytes / (1024**3)

    logger.info(f"Found {total_files} .pth files in total ({total_gb:.2f} GB)")

    migrated_count = 0
    skipped_count = 0
    deleted_count = 0
    kept_count = 0
    freed_bytes = 0

    t_start = time.time()

    for idx, src_path in enumerate(all_pths, 1):
        rel_path = os.path.relpath(src_path, CURRENT_DIR)
        dest_path = os.path.join(GDRIVE_BASE, rel_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        src_size = os.path.getsize(src_path)

        # コピー済み判定（サイズ一致）
        already_copied = False
        if os.path.exists(dest_path):
            if os.path.getsize(dest_path) == src_size:
                already_copied = True
                skipped_count += 1

        # コピー実行
        if not already_copied:
            t0 = time.time()
            shutil.copy2(src_path, dest_path)
            dt = time.time() - t0
            migrated_count += 1
            dest_size = os.path.getsize(dest_path)
            if dest_size != src_size:
                logger.error(f"Integrity check failed for {rel_path}! (src: {src_size}, dest: {dest_size})")
                continue

        # ローカル保持・削除の判定
        if src_path in keep_paths:
            kept_count += 1
            # 保持対象はローカルに残す
        else:
            # 退避が完了したのでローカルから安全に削除し容量解放
            os.remove(src_path)
            deleted_count += 1
            freed_bytes += src_size

        if idx % 50 == 0 or idx == total_files:
            elapsed = time.time() - t_start
            freed_gb = freed_bytes / (1024**3)
            logger.info(
                f"[{idx:3d}/{total_files} ({idx/total_files*100:5.1f}%)] "
                f"コピー済: {migrated_count} | スキップ: {skipped_count} | "
                f"ローカル保持: {kept_count} | 削除解放: {deleted_count} ({freed_gb:6.2f} GB) | "
                f"経過: {elapsed/60:4.1f}分"
            )

    elapsed_total = time.time() - t_start
    final_freed_gb = freed_bytes / (1024**3)

    logger.info("==================================================================")
    logger.info("  ★ 全 .pth ファイルの Google Drive 退避処理が完了いたしました！ ★")
    logger.info(f"  総ファイル数     : {total_files} files")
    logger.info(f"  Google Drive 退避: {migrated_count + skipped_count} files (完全退避)")
    logger.info(f"  ローカル保持     : {kept_count} files (上位15アンサンブル使用モデル)")
    logger.info(f"  ローカル解放容量 : {deleted_count} files ({final_freed_gb:.2f} GB 解放)")
    logger.info(f"  総所要時間       : {elapsed_total/60:.2f} 分")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
