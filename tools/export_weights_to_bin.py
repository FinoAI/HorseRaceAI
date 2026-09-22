import os
import sys
import struct
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.base_nn import RaceLevelStage1NN

def export_model_to_bin(model: RaceLevelStage1NN, output_bin_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    with open(output_bin_path, "wb") as f:
        magic = b"HRM1"
        input_dim = model.input_dim
        max_horses = model.max_horses
        num_blocks = len(model.blocks)
        
        f.write(magic)
        f.write(struct.pack("<III", input_dim, max_horses, num_blocks))
        
        first_layer_dim = model.input_layer[0].out_features
        f.write(struct.pack("<II", input_dim, first_layer_dim))
        
        for block in model.blocks:
            in_d = block.linear.in_features
            out_d = block.linear.out_features
            f.write(struct.pack("<II", in_d, out_d))
            
        head_in = model.head.in_features
        head_out = model.head.out_features
        f.write(struct.pack("<II", head_in, head_out))
        
        # input_layer
        w_in = model.input_layer[0].weight.detach().cpu().numpy().astype(np.float32)
        b_in = model.input_layer[0].bias.detach().cpu().numpy().astype(np.float32)
        w_ln0 = model.input_layer[1].weight.detach().cpu().numpy().astype(np.float32)
        b_ln0 = model.input_layer[1].bias.detach().cpu().numpy().astype(np.float32)
        
        f.write(w_in.tobytes())
        f.write(b_in.tobytes())
        f.write(w_ln0.tobytes())
        f.write(b_ln0.tobytes())
        
        # blocks
        for block in model.blocks:
            w_lin = block.linear.weight.detach().cpu().numpy().astype(np.float32)
            b_lin = block.linear.bias.detach().cpu().numpy().astype(np.float32)
            w_norm = block.norm.weight.detach().cpu().numpy().astype(np.float32)
            b_norm = block.norm.bias.detach().cpu().numpy().astype(np.float32)
            w_res = block.residual_proj.weight.detach().cpu().numpy().astype(np.float32)
            b_res = block.residual_proj.bias.detach().cpu().numpy().astype(np.float32)
            
            f.write(w_lin.tobytes())
            f.write(b_lin.tobytes())
            f.write(w_norm.tobytes())
            f.write(b_norm.tobytes())
            f.write(w_res.tobytes())
            f.write(b_res.tobytes())
            
        # head
        w_head = model.head.weight.detach().cpu().numpy().astype(np.float32)
        b_head = model.head.bias.detach().cpu().numpy().astype(np.float32)
        
        f.write(w_head.tobytes())
        f.write(b_head.tobytes())

    size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    print(f"Exported model weights successfully to {output_bin_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=10, help="特徴量数 (デフォルト10: テスト高速化用)")
    parser.add_argument("--out", type=str, default="artifacts_models/model_weights_test.bin", help="出力先パス")
    args = parser.parse_args()
    
    max_horses = 18
    input_dim = max_horses * args.features
    print(f"Creating model: {max_horses} horses x {args.features} features = {input_dim} inputs, 1st layer = {input_dim * 32}")
    
    torch.manual_seed(42)
    model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=32,
        reduction_ratio=0.5,
        dropout=0.2
    )
    export_model_to_bin(model, args.out)
