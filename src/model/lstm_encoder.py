"""
LSTM 시계열 인코더
입력: [batch, seq_len, input_dim]  (환경변수 시퀀스)
출력: [batch, embed_dim]           (시간 임베딩 벡터)
"""
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            embed: [batch, embed_dim]
        """
        _, (h_n, _) = self.lstm(x)   # h_n: [num_layers, batch, hidden_dim]
        last_hidden = h_n[-1]         # [batch, hidden_dim]  (마지막 레이어)
        return self.proj(last_hidden) # [batch, embed_dim]
