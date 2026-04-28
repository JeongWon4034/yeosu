"""
MarineDebrisGNN: LSTM 시간 인코더 + GAT 그래프 예측 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.model.lstm_encoder import LSTMEncoder


class MarineDebrisGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 7,
        lstm_input_dim: int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_embed: int = 32,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm_enc = LSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=0.2,
            embed_dim=lstm_embed,
        )

        in_dim = node_feat_dim + lstm_embed  # 6 + 32 = 38

        self.gat1 = GATConv(
            in_channels=in_dim,
            out_channels=gat_hidden,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
            edge_dim=5,
        )
        self.gat2 = GATConv(
            in_channels=gat_hidden * gat_heads,  # 256
            out_channels=32,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=5,
        )
        self.out = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, env_seq: torch.Tensor, return_attention: bool = False):
        """
        Args:
            data: PyG Data (x, edge_index, edge_attr)
            env_seq: [1, seq_len, feat]  — 단일 시퀀스 (배치 차원 포함)
            return_attention: True이면 (pred, attn_weights) 반환
        Returns:
            pred: [N]  노드별 예측값
        """
        # 1. LSTM → 시간 임베딩 [1, 32]
        time_emb = self.lstm_enc(env_seq)          # [1, 32]
        time_emb = time_emb.expand(data.x.size(0), -1)  # [N, 32]

        # 2. 노드 피처 concat → [N, 38]
        x = torch.cat([data.x, time_emb], dim=-1)

        # 3. GAT Layer 1
        if return_attention:
            x, (edge_idx_ret, attn1) = self.gat1(
                x, data.edge_index, data.edge_attr, return_attention_weights=True
            )
        else:
            x = self.gat1(x, data.edge_index, data.edge_attr)

        x = F.elu(x)
        x = self.dropout(x)

        # 4. GAT Layer 2
        if return_attention:
            x, (_, attn2) = self.gat2(
                x, data.edge_index, data.edge_attr, return_attention_weights=True
            )
        else:
            x = self.gat2(x, data.edge_index, data.edge_attr)

        # 5. 출력: GNN residual + source_count anchor (고정 스케일)
        # source_count anchor: sc_norm * y_max → source_count 기반 기준 예측
        sc_anchor = data.x[:, 6] * data.sc_max  # [N]
        residual  = self.out(x).squeeze(-1)      # [N] — GNN이 잔차 학습
        pred = sc_anchor + residual

        if return_attention:
            return pred, (edge_idx_ret, attn1, attn2)
        return pred
