import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.models.base_nn import ResNetBlock

class RaceLevelMetaNN(nn.Module):
    """
    レース単位入力型 後段メタニューラルネットワークモデル (スタッキングアンサンブル)
    - 入力: 10モデルのレース予測勝率 (18 × 10 = 180) + 各馬番のアンサンブル統計量 (18 × 4 = 72) = 計252次元
    - 第1隠れ層: 入力パラメータ数 × 32倍ユニット
    - 隠れ層減衰: 最大1/2ずつ段階的に縮小 (急激な1/10圧縮を禁止)
    - 5段以上のピラミッド型ディープアーキテクチャ
    - 出力: 18頭分の最終勝馬ロジット (非出走馬マスキング対応)
    """
    def __init__(
        self,
        input_dim: int,
        max_horses: int = 18,
        multiplier: int = 32,
        reduction_ratio: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_horses = max_horses
        first_layer_dim = input_dim * multiplier
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.LayerNorm(first_layer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        layer_dims = []
        curr_dim = first_layer_dim
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
        self.total_layers = 1 + len(self.blocks) + 1

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        
        if mask is not None:
            masked_logits = logits + (1.0 - mask) * -1e9
            return masked_logits
        return logits
