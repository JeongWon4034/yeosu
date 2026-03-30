"""
Step 1: 시공간 데이터 파이프라인 및 EDA
===========================================
2026 여수 세계섬박람회 — 해양쓰레기 예측 하이브리드 모델 (Grey-box)

담당 기능:
  1. test_data_1.csv 로드 → 관측/미관측 노드 분류
  2. 위경도 기반 고유 격자 코드(Grid Code) 생성
  3. 정적(Matplotlib) + 인터랙티브(Folium) 지도 시각화
  4. 보조 데이터(어항, 하천 방출점, 남해 정답지) 통합
  5. EDA 리포트 + 격자별 집계 테이블 반환

Usage:
    from src.pipeline.step1_data_pipeline import SpatioTemporalDataPipeline

    pipeline = SpatioTemporalDataPipeline(data_root="final_data")
    gdf = pipeline.run(
        save_static="output/step1_static_map.png",
        save_interactive="output/step1_interactive_map.html",
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point

# ── macOS 한글 폰트 설정 ─────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 격자 코드 생성기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GridCodeGenerator:
    """
    위경도 기반 고유 격자 코드 생성기

    • 해상도 0.05° ≈ 5.5km (MOHID 격자와 연계 가능한 구조)
    • 포맷: G{lat_idx:04d}N{lon_idx:04d}E
      예) 여수(34.72°N, 127.77°E) → G0344N0075E
    • 남해안 기준 원점: 33.0°N, 124.0°E (신안 서쪽 해상)
    """

    LAT_ORIGIN: float = 33.0
    LON_ORIGIN: float = 124.0

    def __init__(self, lat_resolution: float = 0.05, lon_resolution: float = 0.05):
        self.lat_res = lat_resolution
        self.lon_res = lon_resolution

    def encode(self, lat: float, lon: float) -> str:
        """좌표 → 격자 코드"""
        lat_idx = int((lat - self.LAT_ORIGIN) / self.lat_res)
        lon_idx = int((lon - self.LON_ORIGIN) / self.lon_res)
        return f"G{lat_idx:04d}N{lon_idx:04d}E"

    def decode(self, grid_code: str) -> Tuple[float, float]:
        """격자 코드 → 격자 중심 좌표 (lat, lon)"""
        body = grid_code.lstrip("G")
        lat_str, lon_str = body.split("N")
        lon_str = lon_str.rstrip("E")
        lat_idx, lon_idx = int(lat_str), int(lon_str)
        lat = self.LAT_ORIGIN + (lat_idx + 0.5) * self.lat_res
        lon = self.LON_ORIGIN + (lon_idx + 0.5) * self.lon_res
        return lat, lon

    def get_neighbors(self, grid_code: str, radius: int = 1) -> list[str]:
        """인접 격자 코드 목록 (8방향 × radius)"""
        body = grid_code.lstrip("G")
        lat_str, lon_str = body.split("N")
        lat_idx = int(lat_str)
        lon_idx = int(lon_str.rstrip("E"))
        return [
            f"G{lat_idx + dlat:04d}N{lon_idx + dlon:04d}E"
            for dlat in range(-radius, radius + 1)
            for dlon in range(-radius, radius + 1)
            if not (dlat == 0 and dlon == 0)
        ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 노드 분류기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NodeClassifier:
    """
    관측 유무에 따른 노드 분류

    • observed   : id_count 또는 source_count 값이 존재하는 노드
                   → 전이학습의 레이블(Label)로 사용
    • unobserved : 두 컬럼 모두 결측 → 모델 예측 대상(Inference target)
    """

    OBSERVED = "observed"
    UNOBSERVED = "unobserved"

    @classmethod
    def classify(
        cls,
        df: pd.DataFrame,
        count_col: str = "id_count",
        source_col: str = "source_count",
    ) -> pd.DataFrame:
        df = df.copy()
        has_data = df[count_col].notna() | df[source_col].notna()
        df["node_type"] = np.where(has_data, cls.OBSERVED, cls.UNOBSERVED)
        # 결측값 → 0 (GNN 피처 텐서에서 마스크로 구분)
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
        df[source_col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)
        return df

    @classmethod
    def summary(cls, gdf: gpd.GeoDataFrame) -> dict:
        vc = gdf["node_type"].value_counts()
        total = len(gdf)
        obs = vc.get(cls.OBSERVED, 0)
        return {
            "total": total,
            "observed": obs,
            "unobserved": vc.get(cls.UNOBSERVED, 0),
            "coverage_pct": round(obs / total * 100, 1) if total else 0.0,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 메인 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpatioTemporalDataPipeline:
    """
    Step 1 전체 파이프라인

    Parameters
    ----------
    data_root : str | Path
        final_data/ 폴더 경로
    """

    # ── 시각화 스타일 팔레트 ──────────────────────────────────────────────────
    _PALETTE = {
        "observed":   dict(color="#E63946", marker="o", label="관측 노드 (수거 기록 있음)", zorder=6),
        "unobserved": dict(color="#457B9D", marker="^", label="미관측 노드 (예측 대상)",   zorder=5),
        "port":       dict(color="#2A9D8F", marker="s", label="어항/항만",                zorder=4),
        "river":      dict(color="#F4A261", marker="D", label="하천 방출점(쓰레기 소스)", zorder=4),
    }
    _BG_COLOR = "#0d1b2a"
    _GRID_COLOR = "#1e3a5f"

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.grid_gen = GridCodeGenerator()

        # 로드된 데이터 저장
        self.gdf_nodes: Optional[gpd.GeoDataFrame] = None
        self.gdf_ports: Optional[gpd.GeoDataFrame] = None
        self.gdf_rivers: Optional[gpd.GeoDataFrame] = None
        self.gdf_truth: Optional[gpd.GeoDataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # 데이터 로드
    # ─────────────────────────────────────────────────────────────────────────

    def load_nodes(self, filename: str = "test_data_1.csv") -> gpd.GeoDataFrame:
        """
        핵심 노드 데이터 로드 및 전처리

        파생 컬럼:
          grid_code  — 0.05° 격자 코드
          node_type  — 'observed' / 'unobserved'
          is_yeosu   — 여수 관할 여부 (bool)
          x_5179     — UTM-K X 좌표 (거리 계산용)
          y_5179     — UTM-K Y 좌표 (거리 계산용)
        """
        path = self.data_root / filename
        df = pd.read_csv(path)

        df = NodeClassifier.classify(df)
        df["grid_code"] = df.apply(
            lambda r: self.grid_gen.encode(r["Latitude"], r["Longitude"]), axis=1
        )

        geometry = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        gdf["is_yeosu"] = gdf["지역명"].str.contains("여수", na=False)

        # UTM-K 투영 좌표 저장 (Step 2 Edge 거리 계산에 사용)
        gdf_proj = gdf.to_crs("EPSG:5179")
        gdf["x_5179"] = gdf_proj.geometry.x
        gdf["y_5179"] = gdf_proj.geometry.y

        self.gdf_nodes = gdf
        stats = NodeClassifier.summary(gdf)
        print(f"✅ 노드 로드 완료 | 전체: {stats['total']}개  "
              f"관측: {stats['observed']}개  미관측: {stats['unobserved']}개  "
              f"커버리지: {stats['coverage_pct']}%")
        return gdf

    def load_auxiliary(self) -> None:
        """보조 공간 데이터(어항 Shapefile, 하천 방출점, 남해 정답지) 로드"""
        shp_dir = self.data_root / "Mohid_prepare_data" / "여수섬프로젝트"

        # 어항 위치 (쓰레기 취약도 피처로 사용)
        port_path = shp_dir / "jeonnam_fishing_port_status.shp"
        if port_path.exists():
            self.gdf_ports = gpd.read_file(str(port_path)).to_crs("EPSG:4326")
            print(f"✅ 어항 Shapefile 로드: {len(self.gdf_ports)}개")

        # 하천 방출점 (MOHID 입자 릴리즈 소스)
        river_path = shp_dir / "namhae_sources.shp"
        if river_path.exists():
            self.gdf_rivers = gpd.read_file(str(river_path)).to_crs("EPSG:4326")
            print(f"✅ 하천 방출점 Shapefile 로드: {len(self.gdf_rivers)}개")

        # 남해 정답지 — 전이학습 사전학습(Pre-training) 레이블
        truth_path = self.data_root / "남해_정답지데이터.csv"
        if truth_path.exists():
            df_t = pd.read_csv(str(truth_path))
            geom = [Point(lon, lat) for lon, lat in zip(df_t["Longitude"], df_t["Latitude"])]
            self.gdf_truth = gpd.GeoDataFrame(df_t, geometry=geom, crs="EPSG:4326")
            print(f"✅ 남해 정답지 로드: {len(self.gdf_truth)}개 지점")

    # ─────────────────────────────────────────────────────────────────────────
    # 정적 지도 (Matplotlib)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_static_map(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (16, 9),
    ) -> None:
        """
        서브플롯 2개짜리 정적 지도 생성

        좌) 남해안 전체 뷰 — Pre-training 노드 분포
        우) 여수 확대 뷰 — Fine-tuning 대상, 수거량 버블 오버레이
        """
        assert self.gdf_nodes is not None, "load_nodes()를 먼저 호출하세요."

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize, facecolor=self._BG_COLOR
        )

        for ax in (ax1, ax2):
            ax.set_facecolor(self._BG_COLOR)
            ax.tick_params(colors="#8892b0", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a3f5f")

        # ── 좌측: 남해안 전체 ──────────────────────────────────────────────
        self._scatter_nodes(ax1, label_fmt="name")

        if self.gdf_ports is not None:
            p = self.gdf_ports
            ax1.scatter(
                p.geometry.x, p.geometry.y,
                **{k: self._PALETTE["port"][k] for k in ("color", "marker", "zorder")},
                s=35, alpha=0.65, label=self._PALETTE["port"]["label"],
                edgecolors="none",
            )

        if self.gdf_rivers is not None:
            r = self.gdf_rivers
            ax1.scatter(
                r.geometry.x, r.geometry.y,
                **{k: self._PALETTE["river"][k] for k in ("color", "marker", "zorder")},
                s=30, alpha=0.65, label=self._PALETTE["river"]["label"],
                edgecolors="none",
            )

        self._style_axis(
            ax1,
            title="남해안 전체 노드 분포\n( Pre-training 영역 )",
            legend_loc="lower left",
        )
        self._draw_grid_lines(ax1)

        # ── 우측: 여수 확대 ────────────────────────────────────────────────
        yeosu = self.gdf_nodes[self.gdf_nodes["is_yeosu"]]
        all_nodes_near_yeosu = self.gdf_nodes[
            (self.gdf_nodes.geometry.x > 127.3) & (self.gdf_nodes.geometry.x < 128.3) &
            (self.gdf_nodes.geometry.y > 33.8) & (self.gdf_nodes.geometry.y < 35.0)
        ]

        if len(yeosu) > 0:
            bnds = yeosu.total_bounds  # [minx, miny, maxx, maxy]
            buf = 0.35
            ax2.set_xlim(bnds[0] - buf, bnds[2] + buf)
            ax2.set_ylim(bnds[1] - buf, bnds[3] + buf)

        # 여수 주변 전체 노드
        self._scatter_nodes(ax2, subset=all_nodes_near_yeosu, label_fmt="name+grid", node_scale=1.6)

        # 수거량 버블 오버레이 (관측 노드만)
        obs_nearby = all_nodes_near_yeosu[
            (all_nodes_near_yeosu["node_type"] == "observed") &
            (all_nodes_near_yeosu["id_count"] > 0)
        ]
        if len(obs_nearby) > 0:
            bubble_sizes = (obs_nearby["id_count"] / obs_nearby["id_count"].max() * 600).clip(lower=30)
            ax2.scatter(
                obs_nearby.geometry.x, obs_nearby.geometry.y,
                c=self._PALETTE["observed"]["color"],
                s=bubble_sizes, alpha=0.18, zorder=3,
                label="수거량 (버블 크기)",
            )

        self._style_axis(
            ax2,
            title="여수 관할 노드 확대\n( Fine-tuning 영역 )",
            legend_loc="lower left",
        )
        self._draw_grid_lines(ax2)

        # ── 하단 통계 텍스트 ───────────────────────────────────────────────
        stats = NodeClassifier.summary(self.gdf_nodes)
        yeosu_obs = self.gdf_nodes[self.gdf_nodes["is_yeosu"] & (self.gdf_nodes["node_type"] == "observed")]
        total_debris = int(self.gdf_nodes["id_count"].sum())
        info = (
            f"전체 노드 {stats['total']}개  |  관측 {stats['observed']}개  |  "
            f"미관측 {stats['unobserved']}개  |  커버리지 {stats['coverage_pct']}%  |  "
            f"여수 관측 노드 {len(yeosu_obs)}개  |  총 수거량 {total_debris:,}개"
        )
        fig.text(
            0.5, 0.015, info, ha="center", fontsize=9, color="#8892b0",
            bbox=dict(boxstyle="round", facecolor="#1e2d3d", alpha=0.85, edgecolor="#2a3f5f"),
        )

        plt.suptitle(
            "해양쓰레기 예측 노드 현황 — 2026 여수 세계섬박람회",
            color="white", fontsize=14, y=0.98,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight",
                facecolor=self._BG_COLOR, edgecolor="none",
            )
            print(f"💾 정적 맵 저장: {save_path}")
        plt.show()

    def _scatter_nodes(
        self,
        ax: plt.Axes,
        subset: Optional[gpd.GeoDataFrame] = None,
        label_fmt: str = "name",
        node_scale: float = 1.0,
    ) -> None:
        """노드 타입별 산점도 + 레이블 렌더링 (내부 헬퍼)"""
        data = subset if subset is not None else self.gdf_nodes

        for ntype in ("observed", "unobserved"):
            s = data[data["node_type"] == ntype]
            if s.empty:
                continue
            style = self._PALETTE[ntype]
            base_size = 110 if ntype == "observed" else 65
            ax.scatter(
                s.geometry.x, s.geometry.y,
                c=style["color"], marker=style["marker"],
                s=base_size * node_scale,
                alpha=0.92, zorder=style["zorder"],
                edgecolors="white", linewidths=0.4,
                label=style["label"],
            )
            for _, row in s.iterrows():
                label = (
                    f"{row['지역명']}\n[{row['grid_code']}]"
                    if label_fmt == "name+grid"
                    else row["지역명"]
                )
                ax.annotate(
                    label,
                    (row.geometry.x, row.geometry.y),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=5.5 if label_fmt == "name+grid" else 6.0,
                    color="#ccd6f6", zorder=7,
                )

    def _style_axis(self, ax: plt.Axes, title: str, legend_loc: str) -> None:
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_xlabel("Longitude (°E)", color="#8892b0", fontsize=8)
        ax.set_ylabel("Latitude (°N)",  color="#8892b0", fontsize=8)
        ax.legend(
            loc=legend_loc, fontsize=7,
            facecolor="#1e2d3d", edgecolor="#2a3f5f", labelcolor="white",
        )
        ax.grid(True, color=self._GRID_COLOR, alpha=0.4, linestyle="--", linewidth=0.5)

    def _draw_grid_lines(self, ax: plt.Axes) -> None:
        """0.05° 격자 배경선 렌더링"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 아직 limits가 기본값(0,1)이면 스킵
        if abs(xlim[1] - xlim[0]) < 0.1:
            return

        res = self.grid_gen.lat_res  # lat_res == lon_res == 0.05

        lat0 = np.floor(ylim[0] / res) * res
        for lat in np.arange(lat0, ylim[1] + res, res):
            ax.axhline(lat, color=self._GRID_COLOR, linewidth=0.3, alpha=0.5, zorder=1)

        lon0 = np.floor(xlim[0] / res) * res
        for lon in np.arange(lon0, xlim[1] + res, res):
            ax.axvline(lon, color=self._GRID_COLOR, linewidth=0.3, alpha=0.5, zorder=1)

    # ─────────────────────────────────────────────────────────────────────────
    # 인터랙티브 지도 (Folium)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_interactive_map(self, save_path: Optional[str] = None) -> folium.Map:
        """
        Folium 인터랙티브 지도 생성

        • 관측 노드 (빨강) / 미관측 노드 (파랑) 구분
        • 클릭 팝업: 지역명, 격자 코드, 수거량, 여수 관할 여부
        • 레이어 토글: 관측/미관측/어항/격자 독립 제어
        """
        assert self.gdf_nodes is not None, "load_nodes()를 먼저 호출하세요."

        center = [self.gdf_nodes.geometry.y.mean(), self.gdf_nodes.geometry.x.mean()]
        m = folium.Map(location=center, zoom_start=8, tiles="CartoDB dark_matter")

        obs_layer   = folium.FeatureGroup(name="관측 노드 (수거 기록)", show=True)
        unobs_layer = folium.FeatureGroup(name="미관측 노드 (예측 대상)", show=True)
        port_layer  = folium.FeatureGroup(name="어항/항만", show=False)
        river_layer = folium.FeatureGroup(name="하천 방출점", show=False)

        # 노드 마커
        for _, row in self.gdf_nodes.iterrows():
            is_obs = row["node_type"] == "observed"
            color  = "#E63946" if is_obs else "#457B9D"
            type_label = (
                '<span style="color:#E63946"><b>관측</b></span>'
                if is_obs else
                '<span style="color:#457B9D"><b>미관측 (예측 대상)</b></span>'
            )
            debris_line = (
                f"<br><b>쓰레기 수거량:</b> {int(row['id_count']):,}개"
                if is_obs and row["id_count"] > 0 else ""
            )
            yeosu_badge = " 🌊 여수 관할" if row["is_yeosu"] else ""
            popup_html = f"""
            <div style="font-family:sans-serif;min-width:220px;padding:4px">
              <h4 style="margin:0;color:{color}">{row['지역명']}{yeosu_badge}</h4>
              <hr style="margin:5px 0;border-color:#333">
              <table style="width:100%;font-size:12px">
                <tr><td><b>격자 코드</b></td><td><code>{row['grid_code']}</code></td></tr>
                <tr><td><b>좌표</b></td><td>{row['Latitude']:.4f}°N, {row['Longitude']:.4f}°E</td></tr>
                <tr><td><b>노드 유형</b></td><td>{type_label}</td></tr>
              </table>
              {debris_line}
            </div>
            """
            marker = folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=9 if is_obs else 6,
                color="white", weight=0.6,
                fill=True, fill_color=color, fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"[{row['node_type']}] {row['지역명']}  |  {row['grid_code']}",
            )
            (obs_layer if is_obs else unobs_layer).add_child(marker)

        # 어항 마커
        if self.gdf_ports is not None:
            for _, row in self.gdf_ports.iterrows():
                if row.geometry is not None:
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=4, color="#2A9D8F",
                        fill=True, fill_color="#2A9D8F", fill_opacity=0.75,
                        tooltip="어항/항만",
                    ).add_to(port_layer)

        # 하천 방출점 마커
        if self.gdf_rivers is not None:
            for _, row in self.gdf_rivers.iterrows():
                if row.geometry is not None:
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=4, color="#F4A261",
                        fill=True, fill_color="#F4A261", fill_opacity=0.75,
                        tooltip="하천 방출점",
                    ).add_to(river_layer)

        for layer in (obs_layer, unobs_layer, port_layer, river_layer):
            layer.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(save_path)
            print(f"🗺️  인터랙티브 맵 저장: {save_path}")

        return m

    # ─────────────────────────────────────────────────────────────────────────
    # EDA 리포트
    # ─────────────────────────────────────────────────────────────────────────

    def run_eda(self) -> pd.DataFrame:
        """
        EDA 리포트 출력 + 격자별 집계 테이블 반환

        반환값은 Step 2 그래프 구성의 Node Feature 설계에 활용됩니다.
        """
        assert self.gdf_nodes is not None, "load_nodes()를 먼저 호출하세요."
        gdf = self.gdf_nodes

        SEP = "=" * 65
        print(f"\n{SEP}")
        print("  📊 EDA 리포트 — 해양쓰레기 예측 노드 데이터")
        print(SEP)

        # ── [1] 전체 노드 목록 ─────────────────────────────────────────────
        print("\n[1] 전체 노드 목록")
        display_cols = ["지역명", "Latitude", "Longitude",
                        "id_count", "source_count", "node_type",
                        "grid_code", "is_yeosu"]
        print(gdf[display_cols].to_string(index=False))

        # ── [2] 노드 타입 분포 ────────────────────────────────────────────
        print("\n[2] 노드 타입 분포")
        for ntype, cnt in gdf["node_type"].value_counts().items():
            bar = "█" * cnt
            print(f"  {ntype:12s} : {cnt:3d}  {bar}")

        # ── [3] 관측 노드 수거량 Top-N ────────────────────────────────────
        print("\n[3] 관측 노드 수거량 순위 (id_count 기준)")
        obs = gdf[gdf["node_type"] == "observed"].sort_values("id_count", ascending=False)
        print(obs[["지역명", "id_count", "source_count", "grid_code", "is_yeosu"]].to_string(index=False))

        # ── [4] 격자 코드 통계 ────────────────────────────────────────────
        print("\n[4] 격자 코드별 노드 집계")
        grid_summary = (
            gdf.groupby("grid_code")
            .agg(
                node_count=("지역명", "count"),
                observed_count=("node_type", lambda x: (x == "observed").sum()),
                total_debris=("id_count", "sum"),
                is_yeosu=("is_yeosu", "any"),
                regions=("지역명", lambda x: ", ".join(x)),
            )
            .reset_index()
            .sort_values("total_debris", ascending=False)
        )
        print(grid_summary.to_string(index=False))

        # ── [5] 공간 범위 요약 ────────────────────────────────────────────
        bounds = gdf.total_bounds
        print(f"\n[5] 공간 범위")
        print(f"  경도 : {bounds[0]:.3f}°E  ~  {bounds[2]:.3f}°E  (범위 {bounds[2]-bounds[0]:.2f}°)")
        print(f"  위도 : {bounds[1]:.3f}°N  ~  {bounds[3]:.3f}°N  (범위 {bounds[3]-bounds[1]:.2f}°)")

        # ── [6] 격자 코드 해상도 검증 ─────────────────────────────────────
        print("\n[6] 격자 코드 encode → decode 검증 (상위 3개)")
        for _, row in gdf.head(3).iterrows():
            decoded = self.grid_gen.encode(row["Latitude"], row["Longitude"])
            lat_d, lon_d = self.grid_gen.decode(decoded)
            error_m = (
                ((row["Latitude"] - lat_d) ** 2 + (row["Longitude"] - lon_d) ** 2) ** 0.5
                * 111_000
            )
            print(
                f"  {row['지역명']:12s}  코드: {decoded}  "
                f"오차: {error_m:.0f}m (격자 중심과의 거리, 최대 {0.05/2*111_000:.0f}m 이내 정상)"
            )

        print(f"\n{SEP}\n")
        return grid_summary

    # ─────────────────────────────────────────────────────────────────────────
    # 전체 실행 진입점
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        save_static: Optional[str] = None,
        save_interactive: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Step 1 파이프라인 일괄 실행

        Returns
        -------
        gpd.GeoDataFrame
            grid_code, node_type, is_yeosu, x_5179, y_5179 가 추가된 노드 GDF
            → Step 2(그래프 구성)의 입력으로 직접 사용
        """
        print("=" * 65)
        print("  🚀 Step 1: 시공간 데이터 파이프라인 시작")
        print("=" * 65 + "\n")

        self.load_nodes()
        self.load_auxiliary()
        self.run_eda()
        self.plot_static_map(save_path=save_static)
        self.plot_interactive_map(save_path=save_interactive)

        print("✅ Step 1 완료 — GeoDataFrame을 Step 2로 전달합니다.\n")
        return self.gdf_nodes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단독 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # 프로젝트 루트 기준 경로 자동 설정
    _ROOT = Path(__file__).resolve().parents[2]
    _DATA = _ROOT / "final_data"
    _OUT  = _ROOT / "output"

    pipeline = SpatioTemporalDataPipeline(data_root=str(_DATA))
    gdf_result = pipeline.run(
        save_static=str(_OUT / "step1_static_map.png"),
        save_interactive=str(_OUT / "step1_interactive_map.html"),
    )

    print("\n[반환된 GeoDataFrame 컬럼]")
    print(gdf_result.columns.tolist())
    print(f"\n[Step 2 입력 샘플]")
    print(gdf_result[["지역명", "grid_code", "node_type", "is_yeosu",
                       "x_5179", "y_5179", "id_count"]].head(8))
