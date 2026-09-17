import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ResNetBlock(nn.Module):
    """
    LayerNorm + GELU + Dropout + 残差接続ブロック
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_proj(x)
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + res

class RaceLevelStage1NN(nn.Module):
    """
    レース単位入力型 前段ニューラルネットワークモデル
    - 入力: 出走頭数(18) × 特徴量数 (D)
    - 第1隠れ層: 入力パラメータ数 × 32倍ユニット
    - 隠れ層減衰: 最大1/2ずつ段階的に縮小 (急激な1/10圧縮を禁止)
    - 5段以上のピラミッド型ディープアーキテクチャ
    - 出力: 18頭分の勝馬ロジット (非出走馬マスキング対応)
    """
    def __init__(
        self,
        input_dim: int,
        max_horses: int = 18,
        multiplier: int = 32,
        reduction_ratio: float = 0.5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_horses = max_horses
        first_layer_dim = input_dim * multiplier
        
        # 入力層: input_dim -> input_dim * 32
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.LayerNorm(first_layer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 隠れ層を最大1/2 (reduction_ratio=0.5) ずつ段階的に縮小
        # 最低5段以上のディープ構造を形成
        layer_dims = []
        curr_dim = first_layer_dim
        
        # 最終層の直前まで半分ずつ縮小 (最低5段を保証)
        while curr_dim > max_horses * 8 or len(layer_dims) < 4:
            next_dim = max(max_horses * 4, int(curr_dim * reduction_ratio))
            if next_dim >= curr_dim:
                break
            layer_dims.append(next_dim)
            curr_dim = next_dim

        self.blocks = nn.ModuleList()
        in_dim = first_layer_dim
        for out_dim in layer_dims:
            self.blocks.append(ResNetBlock(in_dim, out_dim, dropout=dropout))
            in_dim = out_dim
            
        self.head = nn.Linear(in_dim, max_horses)
        self.total_layers = 1 + len(self.blocks) + 1 # input_layer + blocks + head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: shape (batch_size, max_horses * D)
        mask: shape (batch_size, max_horses) (出走馬: 1.0, 未出走: 0.0)
        """
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h) # shape (batch_size, max_horses)
        
        if mask is not None:
            # 非出走スロットに大きな負の値を加えてSoftmaxで確率ゼロ化
            masked_logits = logits + (1.0 - mask) * -1e9
            return masked_logits
        return logits
