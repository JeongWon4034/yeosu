"""
환경변수 시계열 전처리 + 슬라이딩 윈도우 + env_sequences.pt 저장
"""
import os
import sys
sys.path.insert(0, "/Users/jeongwon/yeosu")

import numpy as np
import pandas as pd
import torch

ROOT = "/Users/jeongwon/yeosu"
FINAL = os.path.join(ROOT, "final_data")

FEATURE_COLS = ["tide", "tide_pred", "wind_speed", "wind_dir",
                "u_wind", "v_wind", "u_current", "v_current"]
WINDOW = 168   # 7일 × 24h
STEP   = 24    # 1일 스텝

# ─────────────────────────────────────────────────────────
# 1. 로드 & 정렬
# ─────────────────────────────────────────────────────────
env = pd.read_csv(os.path.join(FINAL, "env_timeseries_merged.csv"),
                  parse_dates=["datetime"])
env = env.sort_values("datetime").reset_index(drop=True)
print(f"원본 shape: {env.shape}")
print(f"기간: {env['datetime'].min()} ~ {env['datetime'].max()}")

# datetime 기준 1시간 간격 reindex → 빈 시간 선형보간
env = env.set_index("datetime")
full_idx = pd.date_range(env.index.min(), env.index.max(), freq="1h")
env = env.reindex(full_idx)
env[FEATURE_COLS] = env[FEATURE_COLS].interpolate(method="linear", limit_direction="both")
env = env.reset_index().rename(columns={"index": "datetime"})
print(f"reindex 후 shape: {env.shape}")

# ─────────────────────────────────────────────────────────
# 2. MinMaxScaler (0~1), 스케일러 파라미터 저장
# ─────────────────────────────────────────────────────────
data_np = env[FEATURE_COLS].values.astype(np.float32)  # [T, 9]

feat_min = data_np.min(axis=0)   # [9]
feat_max = data_np.max(axis=0)   # [9]
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0   # 분모 0 방지

data_scaled = (data_np - feat_min) / feat_range  # [T, 9]
print(f"\n정규화 완료: min={data_scaled.min():.4f}, max={data_scaled.max():.4f}")

# ─────────────────────────────────────────────────────────
# 3. 슬라이딩 윈도우 → [N, 168, 9]
# ─────────────────────────────────────────────────────────
T = len(data_scaled)
starts = range(0, T - WINDOW + 1, STEP)
sequences = np.stack([data_scaled[s:s + WINDOW] for s in starts])  # [N, 168, 9]
print(f"슬라이딩 윈도우: T={T}, 윈도우={WINDOW}, 스텝={STEP} → {len(sequences)}개 시퀀스")
print(f"sequences shape: {sequences.shape}")

seq_tensor = torch.tensor(sequences, dtype=torch.float32)

# ─────────────────────────────────────────────────────────
# 4. 저장
# ─────────────────────────────────────────────────────────
save_dict = {
    "sequences": seq_tensor,          # [N, 168, 9]
    "feature_cols": FEATURE_COLS,
    "scaler_min": torch.tensor(feat_min, dtype=torch.float32),
    "scaler_max": torch.tensor(feat_max, dtype=torch.float32),
    "window": WINDOW,
    "step": STEP,
}
pt_path = os.path.join(FINAL, "env_sequences.pt")
torch.save(save_dict, pt_path)
print(f"\n저장: {pt_path}")

# ─────────────────────────────────────────────────────────
# 5. LSTMEncoder forward pass 검증
# ─────────────────────────────────────────────────────────
from src.model.lstm_encoder import LSTMEncoder

N_FEAT = len(FEATURE_COLS)   # 실제 피처 수 (8)
model = LSTMEncoder(input_dim=N_FEAT, hidden_dim=64, num_layers=2, dropout=0.2, embed_dim=32)
model.eval()

# 실제 시퀀스 첫 배치 (8개)
batch = seq_tensor[:8]
print(f"\n[Forward pass 검증]")
print(f"  입력 shape:  {batch.shape}   (batch=8, seq=168, feat={N_FEAT})")

with torch.no_grad():
    out = model(batch)

print(f"  출력 shape:  {out.shape}    (batch=8, embed=32)")
print(f"  출력 통계: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
print(f"\n  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print("\n완료!")
