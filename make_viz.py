"""
기획서용 시각화 자료 일괄 생성 스크립트
output3/ 에 fig01~fig10 저장
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT  = "/Users/jeongwon/yeosu"
FINAL = os.path.join(ROOT, "final_data")
OUT   = os.path.join(ROOT, "output3")

# ── 한글 폰트 설정
import matplotlib.font_manager as fm
candidates = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
]
for c in candidates:
    if os.path.exists(c):
        fm.fontManager.addfont(c)
        prop = fm.FontProperties(fname=c)
        plt.rcParams["font.family"] = prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드
islands   = pd.read_csv(os.path.join(FINAL, "island_merged.csv"))
loocv     = pd.read_csv(os.path.join(OUT,   "loocv_results.csv"))
attn      = pd.read_csv(os.path.join(OUT,   "attention_weights.csv"))
rp        = pd.read_csv(os.path.join(FINAL, "release_points_encoded.csv"))

N_RP = len(rp)  # 127
island_names = islands["지역명"].tolist()

# 어텐션: island_name 기준으로 dst_node 매핑
attn["island_name"] = attn["dst_node"].apply(
    lambda d: island_names[d - N_RP] if 0 <= (d - N_RP) < len(island_names) else "?"
)

BLUE   = "#1a6faf"
RED    = "#d94f3c"
GREEN  = "#2e8b57"
ORANGE = "#e67e22"
GRAY   = "#888888"
NAVY   = "#1a3a5c"

print("시각화 생성 시작...")


# ══════════════════════════════════════════════════════════
# Fig 01: source_count vs 수량 상관관계
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))

sc = islands["source_count"].values
y  = islands["수량(개)"].values
r, p = stats.pearsonr(sc, y)
slope, intercept, *_ = stats.linregress(sc, y)

colors = plt.cm.RdYlGn_r(y / y.max())
sc_plot = ax.scatter(sc, y, c=y, cmap="RdYlGn_r", s=120, zorder=5,
                     edgecolors="white", linewidths=0.8, alpha=0.9)
cbar = plt.colorbar(sc_plot, ax=ax, shrink=0.8)
cbar.set_label("실측 수거량 (개)", fontsize=11)

x_line = np.linspace(0, sc.max() * 1.05, 200)
ax.plot(x_line, slope * x_line + intercept, color=RED, linewidth=2,
        linestyle="--", label=f"회귀선 (r={r:.3f})")
ax.plot(x_line, x_line, color=GRAY, linewidth=1, linestyle=":", alpha=0.5, label="y=x")

for i, row in islands.iterrows():
    if row["수량(개)"] > 1500 or row["source_count"] > 3000:
        ax.annotate(row["지역명"], (row["source_count"], row["수량(개)"]),
                    fontsize=8.5, xytext=(8, 4), textcoords="offset points",
                    color=NAVY, fontweight="bold")

ax.set_xlabel("MOHID 입자 도달량 (source_count)", fontsize=12)
ax.set_ylabel("실측 해양쓰레기 수거량 (개)", fontsize=12)
ax.set_title(f"MOHID 시뮬레이션 입자량 vs 실측 수거량\nPearson r = {r:.4f}  (p < 0.0001)", fontsize=13, pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig01_data_correlation.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig01 완료")


# ══════════════════════════════════════════════════════════
# Fig 02: 섬 지도 — 위치 + 실측 수거량 버블맵
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax_i, (ax, col, title, cmap_name) in enumerate(zip(
    axes,
    ["수량(개)", "source_count"],
    ["실측 수거량 (개)", "MOHID 입자 도달량"],
    ["RdYlGn_r", "Blues"],
)):
    vals = islands[col].values
    sizes = (vals / vals.max()) * 800 + 60

    sc_map = ax.scatter(
        islands["Longitude"], islands["Latitude"],
        c=vals, cmap=cmap_name, s=sizes, alpha=0.85,
        edgecolors="white", linewidths=0.8, zorder=5
    )
    cbar = plt.colorbar(sc_map, ax=ax, shrink=0.75)
    cbar.set_label(title, fontsize=10)

    for _, row in islands.iterrows():
        if row[col] > vals.mean() * 1.5:
            ax.annotate(row["지역명"],
                        (row["Longitude"], row["Latitude"]),
                        fontsize=7.5, xytext=(5, 3),
                        textcoords="offset points", color=NAVY)

    # 방출점도 표시
    ax.scatter(rp["lon"], rp["lat"], c=GRAY, s=15, alpha=0.4,
               marker="^", zorder=3, label="방출점 (127개)")

    ax.set_xlabel("경도", fontsize=10)
    ax.set_ylabel("위도", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_facecolor("#f0f4f8")

fig.suptitle("남해안 섬 분포 및 해양쓰레기 현황", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig02_island_map.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig02 완료")


# ══════════════════════════════════════════════════════════
# Fig 03: LOOCV 예측 vs 실측 (고품질 버전)
# ══════════════════════════════════════════════════════════
actuals = loocv["actual"].values
preds   = loocv["predicted"].values
r2      = 1 - np.sum((actuals - preds)**2) / np.sum((actuals - actuals.mean())**2)
mae     = np.mean(np.abs(preds - actuals))
rmse    = np.sqrt(np.mean((preds - actuals)**2))
rp_val, _ = stats.pearsonr(actuals, preds)

fig, ax = plt.subplots(figsize=(9, 8))
colors_pts = plt.cm.RdYlGn_r(np.abs(loocv["error"].values) / np.abs(loocv["error"].values).max())
for i, row in loocv.iterrows():
    ax.scatter(row["actual"], row["predicted"],
               color=colors_pts[i], s=130, edgecolors="white", linewidths=0.8, zorder=5, alpha=0.9)
    ax.annotate(row["island"],
                (row["actual"], row["predicted"]),
                fontsize=8, xytext=(5, 3), textcoords="offset points",
                color=NAVY, alpha=0.85)

lim = max(actuals.max(), preds.max()) * 1.08
ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, alpha=0.5, label="Perfect (y=x)")
slope2, intercept2, *_ = stats.linregress(actuals, preds)
x2 = np.array([0, lim])
ax.plot(x2, slope2 * x2 + intercept2, color=RED, linewidth=2,
        label=f"회귀선 (r={rp_val:.3f})")

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("실측 수거량 (개)", fontsize=12)
ax.set_ylabel("예측 수거량 (개)", fontsize=12)
ax.set_title(
    f"LOOCV 예측 성능  —  GNN+LSTM (Residual 구조)\n"
    f"MAE={mae:.0f}   RMSE={rmse:.0f}   R²={r2:.3f}   r={rp_val:.3f}",
    fontsize=13, pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
      norm=plt.Normalize(vmin=0, vmax=np.abs(loocv["error"].values).max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
cbar.set_label("|오차| 크기", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig03_loocv_pred_vs_actual.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig03 완료")


# ══════════════════════════════════════════════════════════
# Fig 04: 섬별 예측 오차 (오차 막대 + 실측/예측 비교)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

loocv_s = loocv.sort_values("actual")
x_pos = np.arange(len(loocv_s))
w = 0.35

ax = axes[0]
bars_a = ax.bar(x_pos - w/2, loocv_s["actual"], w, label="실측", color=BLUE, alpha=0.85, zorder=3)
bars_p = ax.bar(x_pos + w/2, loocv_s["predicted"], w, label="예측", color=ORANGE, alpha=0.85, zorder=3)
ax.set_xticks(x_pos)
ax.set_xticklabels(loocv_s["island"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("수거량 (개)", fontsize=11)
ax.set_title("섬별 실측 수거량 vs 예측 수거량", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#fafafa")

ax2 = axes[1]
err = loocv_s["error"].values
bar_colors = [RED if e < 0 else GREEN for e in err]
ax2.bar(x_pos, err, color=bar_colors, alpha=0.8, zorder=3)
ax2.axhline(0, color="black", linewidth=1.2)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(loocv_s["island"], rotation=40, ha="right", fontsize=9)
ax2.set_ylabel("오차 (예측-실측)", fontsize=11)
ax2.set_title("예측 오차 (양수=과대예측, 음수=과소예측)", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor("#fafafa")

patch_red   = mpatches.Patch(color=RED,   label="과소예측")
patch_green = mpatches.Patch(color=GREEN, label="과대예측")
ax2.legend(handles=[patch_red, patch_green], fontsize=9)

fig.tight_layout(pad=2.5)
fig.savefig(os.path.join(OUT, "fig04_error_by_island.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig04 완료")


# ══════════════════════════════════════════════════════════
# Fig 05: 모델 성능 비교 (이전 vs 이후)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

models = ["블랙박스\nAI (기존)", "GNN+LSTM\n(v1)", "두둥실\nGNN+LSTM\n(Residual)"]
r2_vals = [-0.24, -0.19, 0.917]
mae_vals = [1254, 1273, 516]
r_vals  = [-0.45, -0.43, 0.958]

bar_colors = [GRAY, "#f0a500", BLUE]

for ax_i, (ax, vals, title, unit, better) in enumerate(zip(
    axes,
    [r2_vals, mae_vals, r_vals],
    ["R² (결정계수)", "MAE (평균절대오차)", "Pearson r (상관계수)"],
    ["", " 개", ""],
    ["high", "low", "high"],
)):
    bar_c = []
    for v in vals:
        if better == "high":
            bar_c.append(BLUE if v == max(vals) else (GRAY if v == min(vals) else "#f0a500"))
        else:
            bar_c.append(BLUE if v == min(vals) else (GRAY if v == max(vals) else "#f0a500"))

    bars = ax.bar(models, vals, color=bar_c, alpha=0.85,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if better=="high" else 10),
                f"{v:.3f}{unit}" if unit=="" else f"{v:.0f}{unit}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(min(vals)*1.3 if min(vals)<0 else -0.1,
                max(vals)*1.25 if max(vals)>0 else 0.1)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#fafafa")
    ax.tick_params(axis="x", labelsize=9)

fig.suptitle("모델 성능 개선 비교", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig05_model_comparison.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig05 완료")


# ══════════════════════════════════════════════════════════
# Fig 06: 섬별 위험도 랭킹 (수거량 기준)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 9))

isl_sorted = islands.sort_values("수량(개)", ascending=True)
y_pos = np.arange(len(isl_sorted))

norm = plt.Normalize(isl_sorted["수량(개)"].min(), isl_sorted["수량(개)"].max())
bar_colors = plt.cm.RdYlGn_r(norm(isl_sorted["수량(개)"].values))

bars = ax.barh(y_pos, isl_sorted["수량(개)"], color=bar_colors, alpha=0.9,
               edgecolor="white", linewidth=0.6)
ax2_twin = ax.twinx()
ax2_twin.barh(y_pos, isl_sorted["source_count"], alpha=0.0)

for i, (_, row) in enumerate(isl_sorted.iterrows()):
    ax.text(row["수량(개)"] + 50, i, f'{row["수량(개)"]:,}개', va="center", fontsize=9)
    ax.text(2, i, f'  sc={row["source_count"]:.0f}', va="center", fontsize=8, color=GRAY)

ax.set_yticks(y_pos)
ax.set_yticklabels(isl_sorted["지역명"], fontsize=10)
ax.set_xlabel("실측 수거량 (개)", fontsize=12)
ax.set_title("섬별 해양쓰레기 수거량 랭킹\n(색상: 위험도 高=적색 / MOHID 입자량 표기)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
ax.set_facecolor("#fafafa")

sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
cbar.set_label("위험도", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig06_island_ranking.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig06 완료")


# ══════════════════════════════════════════════════════════
# Fig 07: 방출점 유형 분포 + 지역별 분포
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# 파이차트
type_counts = {
    "해수욕장\n(beach)":  rp["type_beach"].sum(),
    "항구\n(port)":      rp["type_port"].sum(),
    "하천\n(river)":     rp["type_river"].sum(),
    "어업\n(fishery)":   rp["type_fishery"].sum(),
}
type_colors = [ORANGE, BLUE, GREEN, RED]
wedges, texts, autotexts = axes[0].pie(
    type_counts.values(), labels=type_counts.keys(),
    colors=type_colors, autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
axes[0].set_title(f"방출점(오염원) 유형별 분포\n(총 {len(rp)}개)", fontsize=13, fontweight="bold")

# 지리적 분포 산점도
type_map = {
    "type_beach":   (ORANGE, "해수욕장", "o"),
    "type_port":    (BLUE,   "항구",    "s"),
    "type_river":   (GREEN,  "하천",    "^"),
    "type_fishery": (RED,    "어업",    "D"),
}
for col, (c, label, marker) in type_map.items():
    mask = rp[col] == 1
    axes[1].scatter(rp.loc[mask, "lon"], rp.loc[mask, "lat"],
                    c=c, s=60, marker=marker, alpha=0.75, label=label, zorder=4)

axes[1].set_xlabel("경도", fontsize=11)
axes[1].set_ylabel("위도", fontsize=11)
axes[1].set_title("방출점 지리적 분포 (127개소)", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10, loc="upper left")
axes[1].grid(True, alpha=0.25)
axes[1].set_facecolor("#f0f4f8")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig07_release_points.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig07 완료")


# ══════════════════════════════════════════════════════════
# Fig 08: 어텐션 히트맵 (상위 섬 Top-8 × 상위 방출점)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 8))

top_islands = loocv.nlargest(8, "actual")["island"].tolist()
attn_top = attn[attn["island_name"].isin(top_islands)].copy()

pivot = attn_top.pivot_table(
    index="island_name", columns="src_node",
    values="attn_layer2", aggfunc="mean"
)
# 각 섬에서 top-15 방출점만 선택
top_cols = pivot.mean(axis=0).nlargest(15).index
pivot_sub = pivot[top_cols].fillna(0)

# 방출점 번호 → 유형 레이블
def get_type(node_id):
    row = rp[rp["id"] == node_id + 1]
    if len(row) == 0:
        return f"RP{node_id}"
    r = row.iloc[0]
    if r["type_beach"]:   return f"해변#{node_id}"
    if r["type_port"]:    return f"항구#{node_id}"
    if r["type_river"]:   return f"하천#{node_id}"
    return f"어업#{node_id}"

col_labels = [get_type(c) for c in pivot_sub.columns]
im = ax.imshow(pivot_sub.values, cmap="YlOrRd", aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(pivot_sub.index)))
ax.set_yticklabels(pivot_sub.index, fontsize=11)
plt.colorbar(im, ax=ax, label="어텐션 가중치 (GAT Layer 2)")

for i in range(len(pivot_sub.index)):
    for j in range(len(col_labels)):
        v = pivot_sub.values[i, j]
        if v > 0.005:
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if v < 0.03 else "white")

ax.set_title("GAT 어텐션 가중치 히트맵\n(상위 수거량 8개 섬 × 주요 방출점)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("방출점 (오염원)", fontsize=11)
ax.set_ylabel("섬 (예측 대상)", fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig08_attention_heatmap.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig08 완료")


# ══════════════════════════════════════════════════════════
# Fig 09: 시스템 파이프라인 다이어그램
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16); ax.set_ylim(0, 7)
ax.axis("off")
ax.set_facecolor("#f8f9fa")

def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=10.5, sub=""):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor="white", linewidth=2, zorder=3)
    ax.add_patch(box)
    if sub:
        ax.text(x + w/2, y + h*0.62, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                color=text_color, fontsize=8.5, alpha=0.9, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=2, mutation_scale=18), zorder=2)

# Row 1: 데이터 입력
row1_y = 4.8
draw_box(ax, 0.3,  row1_y, 2.5, 1.5, "해류·조위\n풍속 데이터",   "#4a90d9", fontsize=10)
draw_box(ax, 3.1,  row1_y, 2.5, 1.5, "MOHID\n시뮬레이션",       "#2c6e49", fontsize=10)
draw_box(ax, 5.9,  row1_y, 2.5, 1.5, "실측 수거\n데이터 (23곳)", "#e67e22", fontsize=10)
draw_box(ax, 8.7,  row1_y, 2.5, 1.5, "섬 위경도\n(335개)",      "#7b2d8b", fontsize=10)
draw_box(ax, 11.5, row1_y, 2.5, 1.5, "방출점\n(127개소)",       "#c0392b", fontsize=10)

# Row 2: 처리
row2_y = 2.8
draw_box(ax, 1.5, row2_y, 2.8, 1.3, "LSTM 인코더\n시계열 학습",  "#2980b9", fontsize=10)
draw_box(ax, 4.7, row2_y, 2.8, 1.3, "GNN 그래프\n공간 학습",    "#1a6b3c", fontsize=10)
draw_box(ax, 7.9, row2_y, 2.8, 1.3, "전이학습\nFine-tuning",   "#d35400", fontsize=10)
draw_box(ax,11.1, row2_y, 2.8, 1.3, "Residual\n예측 모델",     "#6c3483", fontsize=10)

# Row 3: 출력
draw_box(ax, 5.0, 0.5, 6.0, 1.7,
         "대시보드 — 예측 기반 의사결정 지원",
         NAVY, fontsize=12,
         sub="유입 경로 시각화  |  위험도 랭킹  |  정화 경로 최적화  |  자동 알림")

# 화살표
for (x1,y1,x2,y2) in [
    (1.55,4.8,2.3,4.1), (4.35,4.8,5.1,4.1), (7.15,4.8,5.7,4.1),
    (9.95,4.8,8.0,4.1), (12.75,4.8,12.5,4.1),
    (2.9, 2.8,6.0,2.2), (6.1, 2.8,7.5,2.2), (9.3, 2.8,8.7,2.2), (12.5,2.8,9.5,2.2),
]:
    draw_arrow(ax, x1, y1, x2, y2)

ax.text(8.0, 6.7, "두둥실 시스템 파이프라인",
        ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

# 라벨
for (lx, ly, txt) in [(0.3,6.4,"① 데이터 수집"), (4.7,6.4,"② 모델 학습"),
                       (10.5,6.4,"③ 예측·서비스")]:
    ax.text(lx, ly, txt, fontsize=11, color=GRAY, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig09_pipeline_diagram.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig09 완료")


# ══════════════════════════════════════════════════════════
# Fig 10: 환경변수 시계열 (조위·풍속·해류)
# ══════════════════════════════════════════════════════════
env = pd.read_csv(os.path.join(FINAL, "env_timeseries_merged.csv"), parse_dates=["datetime"])
env_recent = env[env["datetime"] >= "2026-01-01"].copy()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

plots = [
    ("tide",       "조위 (m)",        BLUE,   "조위 (관측)"),
    ("wind_speed", "풍속 (m/s)",      ORANGE, "풍속"),
    ("u_current",  "해류 u 성분 (m/s)", GREEN,  "동서 해류"),
]
for ax_i, (ax, (col, ylabel, color, label)) in enumerate(zip(axes, plots)):
    ax.plot(env_recent["datetime"], env_recent[col],
            color=color, linewidth=0.9, alpha=0.85, label=label)
    ax.fill_between(env_recent["datetime"], env_recent[col], alpha=0.15, color=color)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#fafafa")

axes[0].set_title("2026년 1분기 남해안 해양 환경변수 시계열\n(조위 / 풍속 / 동서해류 성분)", fontsize=13, fontweight="bold")
axes[-1].set_xlabel("날짜", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_env_timeseries.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig10 완료")


# ══════════════════════════════════════════════════════════
# Fig 11: 모델 구조 시각화 (LSTM + GAT + Residual)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15); ax.set_ylim(0, 6)
ax.axis("off")
ax.set_facecolor("#f8f9fa")

def box(ax, x, y, w, h, text, fc, tc="white", fs=10, bold=True, sub=""):
    p = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                        facecolor=fc, edgecolor="white", linewidth=1.8, zorder=3)
    ax.add_patch(p)
    yc = y+h*0.62 if sub else y+h/2
    ax.text(x+w/2, yc, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)
    if sub:
        ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center",
                color=tc, fontsize=8, alpha=0.85, zorder=4)

def arr(ax, x1, y1, x2, y2, color="#555", lw=1.8):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16))

# Input
box(ax, 0.2, 3.5, 2.0, 2.0, "환경변수\n시계열", "#4a90d9", fs=10, sub="[168, 8]\n7일 × 8변수")
box(ax, 0.2, 0.5, 2.0, 2.5, "그래프\n노드·엣지", "#2c6e49", fs=10,
    sub="노드: 150개\n엣지: 1207개\nsc_norm 포함")

# LSTM
box(ax, 2.8, 3.5, 2.2, 2.0, "LSTM\nEncoder", "#1565c0", fs=10.5,
    sub="2층, hidden=64\n→ embed [32]")

# GAT
box(ax, 5.8, 0.5, 2.4, 5.0, "GAT\nLayer 1", "#1a6b3c", fs=10.5,
    sub="in=39, out=64\nheads=4, edge_dim=5")
box(ax, 8.8, 0.5, 2.4, 5.0, "GAT\nLayer 2", "#2e7d32", fs=10.5,
    sub="in=256, out=32\nheads=1")

# Residual & Output
box(ax, 11.8, 3.0, 2.5, 2.0, "sc_anchor\n(고정)", "#e67e22", fs=10,
    sub="sc_norm × y_max")
box(ax, 11.8, 0.5, 2.5, 2.0, "GNN\nresidual", "#6c3483", fs=10, sub="Linear(32→1)")
box(ax, 14.3, 1.5, 0.5, 2.0, "+", NAVY, fs=16)

# Arrows
arr(ax, 2.2, 4.5,  2.8, 4.5, "#4a90d9")
arr(ax, 5.0, 4.5,  5.8, 3.0, "#1565c0")
arr(ax, 2.2, 1.75, 5.8, 1.75, "#2c6e49")
arr(ax, 8.2, 3.0,  8.8, 3.0, "#1a6b3c")
arr(ax, 11.2, 3.0, 11.8, 4.0, "#2e7d32")
arr(ax, 11.2, 1.75, 11.8, 1.5, "#2e7d32")
arr(ax, 14.3, 4.0, 14.55, 3.5, "#e67e22")
arr(ax, 14.3, 1.5, 14.55, 2.0, "#6c3483")

ax.text(14.55, 2.5, "예측\n수거량", ha="center", va="center",
        fontsize=11, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.3", fc="#dce8f5", ec=NAVY, lw=1.5))

ax.text(7.5, 5.7, "MarineDebrisGNN 모델 구조 (LSTM + GAT + Residual)",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_model_architecture.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig11 완료")


# ══════════════════════════════════════════════════════════
# Fig 12: 전이학습 전략 개념도
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_facecolor("#f8f9fa")

# Stage 1
box(ax, 0.3, 1.5, 3.2, 2.5, "Pre-training", "#1a6faf", fs=12,
    sub="남해안 32개 관측소\n일반 유입 패턴 학습")
box(ax, 0.3, 0.2, 3.2, 1.0, "데이터: 32개 지점", "#5aa0d0", tc="white", fs=9.5)

# Arrow
arr(ax, 3.5, 2.75, 4.5, 2.75, "#555", lw=2.5)
ax.text(4.0, 3.2, "전이", ha="center", fontsize=11, color=NAVY, fontweight="bold")

# Stage 2
box(ax, 4.5, 1.5, 3.2, 2.5, "Fine-tuning", "#2e8b57", fs=12,
    sub="여수 지역 특화\n지형·박람회·산업 반영")
box(ax, 4.5, 0.2, 3.2, 1.0, "데이터: 4개 지점", "#5ab08a", tc="white", fs=9.5)

# Arrow
arr(ax, 7.7, 2.75, 8.7, 2.75, "#555", lw=2.5)
ax.text(8.2, 3.2, "예측", ha="center", fontsize=11, color=NAVY, fontweight="bold")

# Stage 3
box(ax, 8.7, 1.5, 3.8, 2.5, "여수 다도해 예측", NAVY, fs=12,
    sub="335개 섬 위험도 산출\n6~24시간 예측")
box(ax, 8.7, 0.2, 3.8, 1.0, "정화 인력·선박 배치 최적화", "#4a6a8a", tc="white", fs=9)

# 문제/해결 박스
ax.text(1.9, 4.7, "문제: 여수 데이터 희소 (4개 지점)",
        ha="center", fontsize=10.5, color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec=RED, lw=1.2))
ax.text(5.1, 4.7, "해결: 남해안 일반 패턴으로 보완",
        ha="center", fontsize=10.5, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec=GREEN, lw=1.2))

ax.text(6.5, 5.2, "전이학습(Transfer Learning) 전략 — 데이터 부족 극복",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig12_transfer_learning.png"), dpi=160, bbox_inches="tight")
plt.close()
print("  fig12 완료")

print(f"\n완료! output3/ 에 fig01~fig12 저장됨")
print("\n".join(sorted([f for f in os.listdir(OUT) if f.startswith("fig")])))
