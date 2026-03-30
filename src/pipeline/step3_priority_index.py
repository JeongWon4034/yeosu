"""
Step 3: 여수형 정화 우선순위 지수 (Cleanup Priority Index)
============================================================
2026 여수 세계섬박람회 — 해양쓰레기 예측 하이브리드 모델 (Grey-box)

공식:
    Priority = (α × Risk) + (β × Vulnerability) - (γ × Accessibility)

    Risk          : GNN/LSTM 모델 출력 — 해양쓰레기 유입 위험도 (0~1)
    Vulnerability : 박람회 민감도 × 생태 취약도 (expo 근접 + 생태 가중치)
    Accessibility : 접근 용이성 (가까운 항만 거리의 역수로 대리)

담당 기능:
  1. PriorityIndexCalculator  — α/β/γ 가중치 기반 CPI 산출
  2. CleanupLoss              — 모델 학습용 커스텀 손실함수
       - 관측 노드: MSE 손실
       - 미관측 노드: 우선순위 정규화 보조 손실
  3. report_top_islands()     — 상위 N개 섬 우선순위 리포팅

Usage:
    from src.pipeline.step3_priority_index import PriorityIndexCalculator, report_top_islands

    calc = PriorityIndexCalculator(alpha=0.5, beta=0.3, gamma=0.2)
    cpi_df = calc.compute(graph_data, gdf_nodes)
    report_top_islands(cpi_df, top_n=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 우선순위 지수 계산기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PriorityIndexCalculator:
    """
    여수형 정화 우선순위 지수 (CPI) 산출기.

    Priority = (α × Risk) + (β × Vulnerability) - (γ × Accessibility)

    Parameters
    ----------
    alpha : Risk 가중치      (기본 0.5) — 유입 위험도 중시
    beta  : Vulnerability 가중치 (기본 0.3) — 박람회/생태 취약도
    gamma : Accessibility 가중치 (기본 0.2) — 접근성 보너스 (높을수록 우선순위↓)

    가중치 조정 가이드:
      박람회 직전 / 미디어 노출 최대화 시기 → beta 상향 (0.5)
      예산 제약 강할 때                      → gamma 상향 (0.4, 멀고 어려운 섬 후순위)
      순수 환경 복원 목적                    → alpha 상향 (0.7)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"α+β+γ=1.0 이어야 합니다. 현재: {alpha+beta+gamma:.3f}"
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    def compute(
        self,
        graph_data:  dict,
        gdf_nodes,
        risk_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        CPI 산출.

        Parameters
        ----------
        graph_data  : step2_graph_builder.GraphBuilder.build() 반환값
        gdf_nodes   : step1의 GeoDataFrame (지역명, is_yeosu, x_5179, y_5179 등)
        risk_scores : 모델 예측 위험도 [N] (None이면 동적 피처 기반 STUB 프록시 사용)

        Returns
        -------
        DataFrame: 지역명, Risk, Vulnerability, Accessibility, CPI, rank
        """
        X       = graph_data["X"]            # [N, 11]
        N       = len(gdf_nodes)
        gdf     = gdf_nodes.reset_index(drop=True)

        # ── Risk ─────────────────────────────────────────────────────────────
        if risk_scores is not None:
            risk = self._minmax(np.array(risk_scores, dtype=np.float32))
        else:
            # [STUB] 모델 없을 때: particle_count(X[:,4]) + label_count(X[:,10]) 합성
            # [교체] GNN/LSTM 모델 출력 텐서를 numpy로 변환해서 넘기면 됨
            raw_risk = X[:, 4] * 0.7 + X[:, 10] * 0.3
            risk = self._minmax(raw_risk)

        # ── Vulnerability ────────────────────────────────────────────────────
        # = 박람회 근접도 × 체류시간 프록시 (높을수록 쓰레기가 오래 머뭄)
        expo_bonus     = gdf["is_yeosu"].astype(float).values * 0.4   # 여수 섬 추가 가중
        residence_norm = self._minmax(X[:, 5])                         # residence_time_n
        vulnerability  = self._minmax(expo_bonus + residence_norm)

        # ── Accessibility ────────────────────────────────────────────────────
        # = port_dist_km 의 역수 정규화 (가까울수록 접근 쉬움 → 값 높음 → 우선순위 패널티)
        port_dist_n    = X[:, 0]                                        # 이미 minmax 완료
        accessibility  = self._minmax(1.0 - port_dist_n)               # 가까울수록 1

        # ── CPI 산출 ─────────────────────────────────────────────────────────
        cpi = self.alpha * risk + self.beta * vulnerability - self.gamma * accessibility
        cpi = self._minmax(cpi)  # 0~1 재정규화

        df = pd.DataFrame({
            "지역명":          gdf["지역명"].values,
            "grid_code":       gdf["grid_code"].values,
            "node_type":       gdf["node_type"].values,
            "is_yeosu":        gdf["is_yeosu"].values,
            "Risk":            risk,
            "Vulnerability":   vulnerability,
            "Accessibility":   accessibility,
            "CPI":             cpi,
            "id_count":        gdf["id_count"].values,
        })
        df["rank"] = df["CPI"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("rank").reset_index(drop=True)

        return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 커스텀 손실 함수 (PyTorch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if _HAS_TORCH:

    class CleanupPriorityLoss(nn.Module):
        """
        여수형 정화 우선순위 최적화를 위한 커스텀 손실 함수.

        L_total = λ_sup  × L_supervised   (관측 노드 MSE)
                + λ_rank × L_rank          (우선순위 순서 보존 — ListMLE)
                + λ_reg  × L_reg           (미관측 노드 smoothness 정규화)

        Parameters
        ----------
        lambda_sup  : 지도학습(MSE) 가중치  (기본 0.6)
        lambda_rank : 순위 손실 가중치       (기본 0.3)
        lambda_reg  : 공간 정규화 가중치     (기본 0.1)
        """

        def __init__(
            self,
            lambda_sup:  float = 0.6,
            lambda_rank: float = 0.3,
            lambda_reg:  float = 0.1,
        ):
            super().__init__()
            assert abs(lambda_sup + lambda_rank + lambda_reg - 1.0) < 1e-5, \
                "λ_sup + λ_rank + λ_reg = 1.0 이어야 합니다."
            self.l_sup  = lambda_sup
            self.l_rank = lambda_rank
            self.l_reg  = lambda_reg
            self.mse    = nn.MSELoss()

        def forward(
            self,
            pred:       "torch.Tensor",   # [N] 모델 예측 위험도
            target:     "torch.Tensor",   # [N] 실제 수거량 (정규화)
            train_mask: "torch.Tensor",   # [N] bool
            edge_index: "torch.Tensor",   # [2, E]
            cpi_target: Optional["torch.Tensor"] = None,  # [N] 정답 CPI (있으면 사용)
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            pred        : GNN 출력 — 각 노드의 위험도 예측값 [N]
            target      : 실제 정규화 수거량 [N]
            train_mask  : 관측 노드 마스크 [N] bool
            edge_index  : 방향성 그래프 엣지 [2, E]
            cpi_target  : (선택) 관측 노드의 실제 CPI 순위 텐서
            """
            # ── L_supervised: 관측 노드 MSE ──────────────────────────────
            L_sup = self.mse(pred[train_mask], target[train_mask])

            # ── L_rank: ListMLE — 상위 노드 순서 보존 ───────────────────
            # 관측 노드의 예측값과 실제 수거량 간 순위 일치도 최대화
            if train_mask.sum() > 1:
                pred_obs   = pred[train_mask]
                target_obs = target[train_mask]
                # ListMLE: log P(target 순서 | pred score)
                log_softmax_pred = torch.log_softmax(pred_obs, dim=0)
                # target을 내림차순 정렬 → 예측 log-prob 합산
                sorted_idx  = torch.argsort(target_obs, descending=True)
                L_rank_vals = []
                for k in range(len(sorted_idx)):
                    remaining   = sorted_idx[k:]
                    log_denom   = torch.logsumexp(pred_obs[remaining], dim=0)
                    L_rank_vals.append(pred_obs[sorted_idx[k]] - log_denom)
                L_rank = -torch.stack(L_rank_vals).mean()
            else:
                L_rank = torch.tensor(0.0)

            # ── L_reg: 인접 노드 예측값 Smoothness ──────────────────────
            src, dst = edge_index[0], edge_index[1]
            L_reg = ((pred[src] - pred[dst]) ** 2).mean()

            L_total = self.l_sup * L_sup + self.l_rank * L_rank + self.l_reg * L_reg

            return L_total

        def breakdown(
            self,
            pred, target, train_mask, edge_index
        ) -> dict:
            """손실 구성요소별 값 반환 (디버깅/모니터링용)."""
            with torch.no_grad():
                L_sup  = self.mse(pred[train_mask], target[train_mask]).item()
                src, dst = edge_index[0], edge_index[1]
                L_reg  = ((pred[src] - pred[dst]) ** 2).mean().item()
            return {
                "L_supervised": round(L_sup,  6),
                "L_rank":       "N/A (use forward)",
                "L_reg":        round(L_reg,  6),
                "weights":      dict(sup=self.l_sup, rank=self.l_rank, reg=self.l_reg),
            }

else:
    # torch 없을 때 더미 클래스
    class CleanupPriorityLoss:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch가 필요합니다: pip install torch")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 상위 N개 섬 우선순위 리포팅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report_top_islands(
    cpi_df:    pd.DataFrame,
    top_n:     int = 5,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    상위 top_n개 섬의 CPI 리포트 출력 및 시각화.

    Parameters
    ----------
    cpi_df    : PriorityIndexCalculator.compute() 반환값
    top_n     : 출력할 상위 섬 수 (기본 5)
    save_path : 차트 저장 경로 (None이면 저장 안 함)

    Returns
    -------
    DataFrame: 상위 top_n개 섬 상세 정보
    """
    top = cpi_df.head(top_n).copy()

    # ── 콘솔 리포트 ──────────────────────────────────────────────────────────
    SEP = "=" * 68
    print(f"\n{SEP}")
    print(f"  🏆 여수형 정화 우선순위 지수 (CPI) — 상위 {top_n}개 섬")
    print(SEP)
    print(f"  {'순위':>4}  {'지역명':12}  {'CPI':>6}  {'Risk':>6}  "
          f"{'Vuln':>6}  {'Access':>6}  {'관측':>4}  {'수거량':>8}")
    print("  " + "-" * 64)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for _, row in top.iterrows():
        medal   = medals.get(row["rank"], "  ")
        obs_tag = "✅" if row["node_type"] == "observed" else "🔮"
        yeosu   = " 🌊" if row["is_yeosu"] else ""
        print(
            f"  {medal} {row['rank']:>2}위  "
            f"{row['지역명']:12}{yeosu}  "
            f"{row['CPI']:.4f}  "
            f"{row['Risk']:.4f}  "
            f"{row['Vulnerability']:.4f}  "
            f"{row['Accessibility']:.4f}  "
            f"{obs_tag}  "
            f"{int(row['id_count']):>8,}개"
        )

    print(f"\n  📌 Action Items:")
    for _, row in top.iterrows():
        priority_reason = []
        if row["Risk"] > 0.6:
            priority_reason.append("유입 위험 높음")
        if row["Vulnerability"] > 0.6:
            priority_reason.append("박람회/생태 취약")
        if row["Accessibility"] < 0.3:
            priority_reason.append("접근 어려움 (선박 배치 필요)")
        reason_str = " / ".join(priority_reason) if priority_reason else "종합 지수 상위"
        print(f"   {row['rank']:>2}위 {row['지역명']:10} → {reason_str}")

    print(f"\n{SEP}\n")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    _plot_cpi_chart(cpi_df, top_n, save_path)

    return top


def _plot_cpi_chart(
    cpi_df:    pd.DataFrame,
    top_n:     int,
    save_path: Optional[str],
) -> None:
    """CPI 구성요소 스택 바 차트 + 전체 순위 산점도."""
    top = cpi_df.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1b2a")
    for ax in axes:
        ax.set_facecolor("#0d1b2a")
        ax.tick_params(colors="#8892b0", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3f5f")

    # ── 좌: 스택 바 (구성요소별 기여도) ─────────────────────────────────────
    ax1 = axes[0]
    labels = [f"{r['지역명']} {'🌊' if r['is_yeosu'] else ''}" for _, r in top.iterrows()]
    x = np.arange(len(labels))

    alpha_c = "#E63946"  # Risk
    beta_c  = "#F4A261"  # Vulnerability
    gamma_c = "#457B9D"  # Accessibility (음수)

    b1 = ax1.bar(x, top["Risk"].values,          color=alpha_c, alpha=0.85, label="α × Risk")
    b2 = ax1.bar(x, top["Vulnerability"].values, color=beta_c,  alpha=0.85, label="β × Vulnerability",
                 bottom=top["Risk"].values)
    b3 = ax1.bar(x, -top["Accessibility"].values, color=gamma_c, alpha=0.65, label="−γ × Accessibility")

    ax1.plot(x, top["CPI"].values, "w-o", lw=2, ms=7, zorder=5, label="CPI (최종)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
    ax1.set_ylabel("지수값", color="#8892b0", fontsize=9)
    ax1.set_title(f"CPI 구성요소 분해 — 상위 {top_n}개 섬",
                  color="white", fontsize=11, pad=10)
    ax1.legend(fontsize=8, facecolor="#1e2d3d", edgecolor="#2a3f5f", labelcolor="white")
    ax1.axhline(0, color="#2a3f5f", linewidth=0.8)
    ax1.grid(True, color="#1e3a5f", alpha=0.3, linestyle="--", axis="y")

    # ── 우: Risk vs Vulnerability 산점도 (전체 노드) ──────────────────────────
    ax2 = axes[1]
    is_obs  = cpi_df["node_type"] == "observed"
    is_yeos = cpi_df["is_yeosu"]

    for mask, color, label, marker in [
        (is_obs  &  is_yeos, "#E63946", "여수 관측",   "o"),
        (is_obs  & ~is_yeos, "#F4A261", "여타 관측",   "o"),
        (~is_obs &  is_yeos, "#ff6b6b", "여수 미관측", "^"),
        (~is_obs & ~is_yeos, "#457B9D", "여타 미관측", "^"),
    ]:
        sub = cpi_df[mask]
        sc = ax2.scatter(
            sub["Risk"], sub["Vulnerability"],
            c=sub["CPI"], cmap="plasma",
            vmin=0, vmax=1,
            s=120 if "관측" in label else 70,
            marker=marker, alpha=0.88, edgecolors="white", linewidths=0.4,
            zorder=5, label=label,
        )

    # 상위 top_n 레이블
    for _, row in top.iterrows():
        ax2.annotate(
            f"#{int(row['rank'])} {row['지역명']}",
            (row["Risk"], row["Vulnerability"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color="white", zorder=6,
        )

    # 사분면 구분선
    ax2.axvline(0.5, color="#2a3f5f", lw=0.8, linestyle="--")
    ax2.axhline(0.5, color="#2a3f5f", lw=0.8, linestyle="--")
    ax2.text(0.75, 0.92, "긴급 정화 구역", color="#E63946", fontsize=8,
             transform=ax2.transAxes, ha="center",
             bbox=dict(boxstyle="round", facecolor="#1e2d3d", alpha=0.7))
    ax2.text(0.25, 0.08, "저우선 구역", color="#457B9D", fontsize=8,
             transform=ax2.transAxes, ha="center",
             bbox=dict(boxstyle="round", facecolor="#1e2d3d", alpha=0.7))

    plt.colorbar(sc, ax=ax2, label="CPI").ax.yaxis.set_tick_params(color="#8892b0")
    ax2.set_xlabel("Risk (유입 위험도)", color="#8892b0", fontsize=9)
    ax2.set_ylabel("Vulnerability (취약도)", color="#8892b0", fontsize=9)
    ax2.set_title("전체 노드 Risk × Vulnerability 분포",
                  color="white", fontsize=11, pad=10)
    ax2.legend(fontsize=8, facecolor="#1e2d3d", edgecolor="#2a3f5f", labelcolor="white")
    ax2.grid(True, color="#1e3a5f", alpha=0.3, linestyle="--")

    plt.suptitle(
        "여수형 정화 우선순위 지수 (CPI) 분석 — 2026 여수 세계섬박람회",
        color="white", fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1b2a", edgecolor="none")
        print(f"💾 CPI 차트 저장: {save_path}")
    plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단독 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))

    from src.pipeline.step1_data_pipeline import SpatioTemporalDataPipeline
    from src.pipeline.step2_graph_builder  import MOHIDDataLoader, GraphBuilder

    # ── Step 1 ────────────────────────────────────────────────────────────────
    pipeline = SpatioTemporalDataPipeline(str(_ROOT / "final_data"))
    pipeline.load_nodes()
    pipeline.load_auxiliary()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    loader  = MOHIDDataLoader(str(_ROOT / "final_data"))
    loader.load_stub()
    builder = GraphBuilder(
        gdf_nodes    = pipeline.gdf_nodes,
        mohid_loader = loader,
        gdf_ports    = pipeline.gdf_ports,
        gdf_rivers   = pipeline.gdf_rivers,
    )
    graph_data = builder.build()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    calc   = PriorityIndexCalculator(alpha=0.5, beta=0.3, gamma=0.2)
    cpi_df = calc.compute(graph_data, pipeline.gdf_nodes)

    top5 = report_top_islands(
        cpi_df,
        top_n=5,
        save_path=str(_ROOT / "output" / "step3_cpi_chart.png"),
    )

    # 손실함수 구조 확인
    if _HAS_TORCH:
        loss_fn = CleanupPriorityLoss(lambda_sup=0.6, lambda_rank=0.3, lambda_reg=0.1)
        print(f"\n[커스텀 손실함수 구조]\n{loss_fn}")
        print("  L = 0.6×MSE(관측) + 0.3×ListMLE(순위) + 0.1×Smooth(정규화)")
