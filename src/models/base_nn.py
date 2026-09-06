import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    LLMやModern Deep Learningで用いられるLayerNorm + GELU + Dropout + 残差接続ブロック
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

class DeepStage1NN(nn.Module):
    """
    前段ニューラルネットワークモデル
    - 5段以上のディープ構造
    - 入力第1層: 入力パラメータ数 × 32倍ユニット
    - LayerNorm, GELU, Dropout, 残差接続
    """
    def __init__(self, input_dim: int, multiplier: int = 32, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        first_layer_dim = input_dim * multiplier
        
        # 5層以上の多層構造（Layer 1 〜 Layer 5 + Head）
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.LayerNorm(first_layer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 隠れ層 4段 (計5段以上の構造)
        self.block1 = ResNetBlock(first_layer_dim, 2048, dropout=dropout)
        self.block2 = ResNetBlock(2048, 1024, dropout=dropout)
        self.block3 = ResNetBlock(1024, 512, dropout=dropout)
        self.block4 = ResNetBlock(512, 256, dropout=dropout)
        
        # 最終出力ヘッド (スコア出力)
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.input_layer(x)   # Layer 1: input -> input * 32
        h = self.block1(h)        # Layer 2: -> 2048
        h = self.block2(h)        # Layer 3: -> 1024
        h = self.block3(h)        # Layer 4: -> 512
        features = self.block4(h) # Layer 5: -> 256
        logits = self.head(features).squeeze(-1) # Layer 6: -> 1
        
        if return_features:
            return logits, features
        return logits
