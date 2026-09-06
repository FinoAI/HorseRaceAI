import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_nn import ResNetBlock

class DeepMetaNN(nn.Module):
    """
    後段メタニューラルネットワークモデル（LLMスタッキングアンサンブル）
    - 前段10モデルの予測スコア（および統計特徴量）を統合
    - 5段以上のディープ構造
    - 入力第1層: 入力パラメータ数 × 32倍ユニット
    """
    def __init__(self, input_dim: int, multiplier: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        first_layer_dim = input_dim * multiplier
        
        # 5層以上の多層構造
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.LayerNorm(first_layer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.block1 = ResNetBlock(first_layer_dim, 256, dropout=dropout)
        self.block2 = ResNetBlock(256, 128, dropout=dropout)
        self.block3 = ResNetBlock(128, 64, dropout=dropout)
        self.block4 = ResNetBlock(64, 32, dropout=dropout)
        
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)   # Layer 1: in -> in * 32
        h = self.block1(h)        # Layer 2: -> 256
        h = self.block2(h)        # Layer 3: -> 128
        h = self.block3(h)        # Layer 4: -> 64
        h = self.block4(h)        # Layer 5: -> 32
        logits = self.head(h).squeeze(-1) # Layer 6: -> 1
        return logits
