"""
MarineDebrisGNN LOOCV 학습 스크립트
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/jeongwon/yeosu")

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.model.marine_debris_gnn import MarineDebrisGNN

ROOT = "/Users/jeongwon/yeosu"
FINAL = os.path.join(ROOT, "final_data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 300
PATIENCE    = 30
PHYS_LAMBDA = 0.01
DEVICE      = torch.device("cpu")

# ─────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────
graph_data = torch.load(os.path.join(FINAL, "graph_data.pt"), weights_only=False)
graph_data = graph_data.to(DEVICE)

env_dict   = torch.load(os.path.join(FINAL, "env_sequences.pt"), weights_only=False)
sequences  = env_dict["sequences"].to(DEVICE)   # [453, 168, 8]

islands    = pd.read_csv(os.path.join(FINAL, "island_merged.csv"))
N_RP       = 127   # 방출점 노드 수
N_IS       = 23    # 섬 노드 수

# 물리 제약용 최대값
SC_MAX = float(islands["source_count"].max())   # 34385.0
print(f"source_count 최댓값 (물리 제약): {SC_MAX}")

# 결측 대체 행 (LOOCV 제외)
IMPUTED_NAMES = {"해남묵동리", "여수반월", "마산봉암"}
island_names  = islands["지역명"].tolist()
# LOOCV에 사용할 섬 인덱스 (결측 3개 제외 → 20개)
loocv_indices = [i for i, n in enumerate(island_names) if n not in IMPUTED_NAMES]
print(f"LOOCV 대상 섬: {len(loocv_indices)}개 (결측 3개 제외)")

# 대표 env 시퀀스: 전체 평균 (단일 시퀀스로 고정)
env_mean_seq = sequences.mean(dim=0, keepdim=True)  # [1, 168, 8]

# y 값 (섬 노드 순서: 노드 127~149)
y_all = graph_data.y[N_RP:].cpu().numpy()   # [23]


# ─────────────────────────────────────────────────────────
# 학습 함수 (단일 fold)
# ─────────────────────────────────────────────────────────
def train_one_fold(test_island_idx: int):
    """
    test_island_idx: 0~22 중 하나 (섬 로컬 인덱스)
    Returns: (predicted_value, true_value, train_loss_history)
    """
    model = MarineDebrisGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse_fn = nn.MSELoss()

    # train_mask: 섬 노드에서 test fold 제외
    train_mask = graph_data.train_mask.clone()
    train_mask[N_RP + test_island_idx] = False   # test 섬 제외

    best_loss   = float("inf")
    best_state  = None
    patience_cnt = 0
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        pred = model(graph_data, env_mean_seq)   # [150]

        # train loss (섬 22개)
        train_pred = pred[train_mask]
        train_true = graph_data.y[train_mask]
        loss = mse_fn(train_pred, train_true)

        # 물리 제약 정규화
        sc_max_t = torch.tensor(SC_MAX, device=DEVICE)
        phys_penalty = PHYS_LAMBDA * torch.mean(torch.relu(pred[graph_data.train_mask] - sc_max_t))
        loss = loss + phys_penalty

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    # 추론
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(graph_data, env_mean_seq)

    pred_val = pred[N_RP + test_island_idx].item()
    true_val = graph_data.y[N_RP + test_island_idx].item()
    return pred_val, true_val, loss_history


# ─────────────────────────────────────────────────────────
# LOOCV 실행
# ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"LOOCV 시작 ({len(loocv_indices)}개 fold)")
print(f"{'='*60}")

results = []
all_loss_histories = []

for fold_idx, isl_idx in enumerate(loocv_indices):
    name = island_names[isl_idx]
    pred_val, true_val, history = train_one_fold(isl_idx)
    results.append({
        "fold": fold_idx + 1,
        "island": name,
        "actual": true_val,
        "predicted": pred_val,
        "error": pred_val - true_val,
        "abs_error": abs(pred_val - true_val),
        "epochs_run": len(history),
    })
    all_loss_histories.append(history)
    print(f"  Fold {fold_idx+1:2d} | {name:10s} | actual={true_val:7.0f} | pred={pred_val:7.1f} | err={pred_val-true_val:+7.1f} | ep={len(history)}")

# ─────────────────────────────────────────────────────────
# 평가 지표
# ─────────────────────────────────────────────────────────
df = pd.DataFrame(results)
actuals = df["actual"].values
preds   = df["predicted"].values

mae   = np.mean(np.abs(preds - actuals))
rmse  = np.sqrt(np.mean((preds - actuals) ** 2))
ss_res = np.sum((actuals - preds) ** 2)
ss_tot = np.sum((actuals - actuals.mean()) ** 2)
r2    = 1 - ss_res / ss_tot
r_pearson, p_val = stats.pearsonr(actuals, preds)

print(f"\n{'='*60}")
print(f"LOOCV 평가 결과 (n={len(df)})")
print(f"  MAE      : {mae:.2f}")
print(f"  RMSE     : {rmse:.2f}")
print(f"  R²       : {r2:.4f}")
print(f"  Pearson r: {r_pearson:.4f}  (p={p_val:.4f})")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────
# 전체 데이터로 최종 모델 학습 (저장용)
# ─────────────────────────────────────────────────────────
print("\n최종 모델 학습 중 (전체 23개 섬)...")
final_model = MarineDebrisGNN().to(DEVICE)
optimizer   = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_fn      = nn.MSELoss()

best_loss  = float("inf")
best_state = None
patience_cnt = 0
final_loss_history = []

for epoch in range(EPOCHS):
    final_model.train()
    optimizer.zero_grad()

    pred = final_model(graph_data, env_mean_seq)
    train_pred = pred[graph_data.train_mask]
    train_true = graph_data.y[graph_data.train_mask]
    loss = mse_fn(train_pred, train_true)

    sc_max_t = torch.tensor(SC_MAX, device=DEVICE)
    phys_penalty = PHYS_LAMBDA * torch.mean(torch.relu(pred[graph_data.train_mask] - sc_max_t))
    loss = loss + phys_penalty

    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    final_loss_history.append(loss_val)

    if loss_val < best_loss:
        best_loss = loss_val
        best_state = copy.deepcopy(final_model.state_dict())
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break

final_model.load_state_dict(best_state)
print(f"최종 모델 학습 완료 ({len(final_loss_history)} epochs, best loss={best_loss:.4f})")

torch.save({
    "model_state_dict": final_model.state_dict(),
    "model_config": dict(
        node_feat_dim=6, lstm_input_dim=8, lstm_hidden=64,
        lstm_layers=2, lstm_embed=32, gat_hidden=64, gat_heads=4, dropout=0.3
    ),
    "loocv_mae": mae,
    "loocv_rmse": rmse,
    "loocv_r2": r2,
}, os.path.join(OUT, "model_namhae.pt"))
print(f"저장: output/model_namhae.pt")

# ─────────────────────────────────────────────────────────
# Attention weights 추출
# ─────────────────────────────────────────────────────────
final_model.eval()
with torch.no_grad():
    _, (edge_idx_ret, attn1, attn2) = final_model(
        graph_data, env_mean_seq, return_attention=True
    )

# attn1: [E, heads] → 헤드 평균
attn1_mean = attn1.cpu().numpy().mean(axis=1)   # [E]
attn2_mean = attn2.cpu().numpy().flatten()       # [E]
ei = edge_idx_ret.cpu().numpy()                 # [2, E]

src_nodes  = ei[0]
dst_nodes  = ei[1]
dst_island_idx = dst_nodes - N_RP               # 섬 로컬 인덱스

attn_df = pd.DataFrame({
    "src_node": src_nodes,
    "dst_node": dst_nodes,
    "island_name": [island_names[i] if 0 <= i < N_IS else "?" for i in dst_island_idx],
    "release_point_id": [int(rp_id) + 1 if s < N_RP else -1 for s, rp_id in zip(src_nodes, src_nodes)],
    "attn_layer1_mean": attn1_mean,
    "attn_layer2": attn2_mean,
})
attn_path = os.path.join(OUT, "attention_weights.csv")
attn_df.to_csv(attn_path, index=False, encoding="utf-8-sig")
print(f"저장: output/attention_weights.csv  ({len(attn_df)}개 엣지)")

# ─────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────

# 1. 학습 곡선 (LOOCV fold별 + 최종)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i, hist in enumerate(all_loss_histories):
    ax.plot(hist, alpha=0.35, linewidth=0.8, color="steelblue")
ax.plot(final_loss_history, color="crimson", linewidth=1.5, label="Final model")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE + phys)")
ax.set_title("Training Loss Curves\n(grey=LOOCV folds, red=final)")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Pred vs Actual scatter
ax2 = axes[1]
ax2.scatter(actuals, preds, s=60, alpha=0.8, color="steelblue", edgecolors="white", linewidths=0.5)
lim = max(actuals.max(), preds.max()) * 1.1
ax2.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Perfect")
# 회귀선
slope, intercept, *_ = stats.linregress(actuals, preds)
x_line = np.array([0, lim])
ax2.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.2, label=f"Fit (r={r_pearson:.3f})")
for _, row in df.iterrows():
    ax2.annotate(row["island"][:4], (row["actual"], row["predicted"]),
                 fontsize=7, alpha=0.7, ha="left", va="bottom",
                 xytext=(3, 3), textcoords="offset points")
ax2.set_xlabel("Actual (개)")
ax2.set_ylabel("Predicted (개)")
ax2.set_title(f"LOOCV Pred vs Actual\nMAE={mae:.0f}  RMSE={rmse:.0f}  R²={r2:.3f}")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_loss.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(OUT, "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: output/training_loss.png")
print("저장: output/pred_vs_actual.png")

# LOOCV 결과 CSV
csv_path = os.path.join(OUT, "loocv_results.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"저장: output/loocv_results.csv")

print("\n모든 출력 완료!")
