import os
import sys
import subprocess
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.base_nn import RaceLevelStage1NN
from tools.export_weights_to_bin import export_model_to_bin

def main():
    print("=== PyTorch vs C/Metal Inference Numerical Equivalence Verification ===")
    
    num_feats = 10
    max_horses = 18
    input_dim = max_horses * num_feats
    
    torch.manual_seed(123)
    np.random.seed(123)
    
    # 1. PyTorch モデル初期化
    model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=32,
        reduction_ratio=0.5,
        dropout=0.0 # 推論時はDropoutなし
    )
    model.eval()
    
    weights_path = "artifacts_models/verify_weights.bin"
    export_model_to_bin(model, weights_path)
    
    # 2. テスト入力作成 (14頭出走、4頭未出走)
    running_horses = 14
    mask = np.zeros(max_horses, dtype=np.float32)
    mask[:running_horses] = 1.0
    
    x = np.zeros(input_dim, dtype=np.float32)
    for h in range(running_horses):
        x[h*num_feats : (h+1)*num_feats] = np.random.uniform(-1.0, 1.0, size=num_feats)
        
    # 3. PyTorch での推論計算
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits_pt = model(x_t, mask=mask_t)
        probs_pt = torch.softmax(logits_pt, dim=-1).squeeze(0).numpy()
        
    print(f"PyTorch Output (Top 3): {np.argsort(probs_pt)[::-1][:3] + 1} with probs: {np.sort(probs_pt)[::-1][:3]}")
    
    # 4. 入力データをバイナリで保存し、C/Metal 実行プログラムから推論
    input_bin_path = "artifacts_models/verify_input.bin"
    output_bin_path = "artifacts_models/verify_output.bin"
    
    with open(input_bin_path, "wb") as f:
        f.write(x.tobytes())
        f.write(mask.tobytes())
        
    # C/Metal 検証用コマンド実行 (main.c またはテストハーネス)
    # ここではテスト用 C スクリプトをコンパイルして実行
    test_c_src = """
    #include <stdio.h>
    #include <stdlib.h>
    #include "horse_race_metal.h"
    
    int main(int argc, char* argv[]) {
        HorseRaceMetalContext* ctx = horse_race_metal_init(argv[1], argv[2]);
        FILE* fi = fopen(argv[3], "rb");
        float x[180];
        float mask[18];
        fread(x, sizeof(float), 180, fi);
        fread(mask, sizeof(float), 18, fi);
        fclose(fi);
        
        float probs[18];
        horse_race_metal_predict(ctx, x, mask, probs);
        
        FILE* fo = fopen(argv[4], "wb");
        fwrite(probs, sizeof(float), 18, fo);
        fclose(fo);
        
        horse_race_metal_free(ctx);
        return 0;
    }
    """
    with open("build/verify_runner.c", "w") as f:
        f.write(test_c_src)
        
    subprocess.run([
        "clang", "-O3", "build/verify_runner.c", "build/horse_race_metal.o",
        "-Iinclude", "-framework", "Foundation", "-framework", "Metal",
        "-o", "bin/verify_runner"
    ], check=True)
    
    subprocess.run([
        "./bin/verify_runner",
        weights_path, "src/metal/kernels.metal", input_bin_path, output_bin_path
    ], check=True)
    
    # 5. C/Metal 出力を読み込み、PyTorch 出力と比較
    probs_metal = np.fromfile(output_bin_path, dtype=np.float32)
    print(f"Metal Output   (Top 3): {np.argsort(probs_metal)[::-1][:3] + 1} with probs: {np.sort(probs_metal)[::-1][:3]}")
    
    diff = np.abs(probs_pt - probs_metal)
    max_diff = np.max(diff)
    print(f"Max Absolute Error between PyTorch and Metal GPU: {max_diff:.8f}")
    
    assert max_diff < 1e-4, f"Difference too large: {max_diff}"
    print(">>> VERIFICATION SUCCESSFUL: PyTorch and Apple Metal outputs match perfectly! <<<")

if __name__ == "__main__":
    main()
