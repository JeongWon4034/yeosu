"""
Step 2: GNN을 위한 그래프 구조 설계
=====================================
2026 여수 세계섬박람회 — 해양쓰레기 예측 하이브리드 모델 (Grey-box)

담당 기능:
  1. MOHIDDataLoader  — MOHID 결과값 로드 인터페이스
       .load_stub()   : 현재 사용 중인 샘플 데이터 (namhae_water_flow_2025_Q1.txt)
       .load_real()   : 실제 MOHID 시뮬레이션 결과로 교체할 진입점  ← 추후 여기만 수정
  2. NodeFeatureBuilder — 정적/동적 피처 텐서 [N, F] 구성
  3. GraphBuilder       — 해수유동 벡터 기반 방향성 그래프 (edge_index, edge_attr)
  4. build_pyg_data()   — PyTorch Geometric Data 객체 최종 조립

MOHID 실데이터 교체 방법:
  loader = MOHIDDataLoader(data_root)
  loader.load_real(
      particle_csv  = "path/to/mohid_particle_output.csv",   # 노드별 도달 입자량
      flow_nc       = "path/to/mohid_uvw.nc",                # 유향/유속 NetCDF
      residence_csv = "path/to/mohid_residence_time.csv",    # 체류 시간
  )
  builder = GraphBuilder(gdf_nodes, loader)
  data = builder.build_pyg_data()

Usage:
    from src.pipeline.step1_data_pipeline import SpatioTemporalDataPipeline
    from src.pipeline.step2_graph_builder  import MOHIDDataLoader, GraphBuilder

    pipeline = SpatioTemporalDataPipeline("final_data")
    gdf = pipeline.run()

    loader  = MOHIDDataLoader("final_data")
    loader.load_stub()                       # 샘플 데이터 사용
    builder = GraphBuilder(gdf, loader)
    data    = builder.build_pyg_data()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# PyTorch / PyG — 없을 경우 numpy fallback 안내
try:
    import torch
    from torch_geometric.data import Data as PygData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    warnings.warn(
        "torch_geometric 미설치. build_pyg_data()는 dict로 fallback 반환됩니다.\n"
        "pip install torch-geometric 후 재실행하면 PygData 객체를 반환합니다.",
        stacklevel=2,
    )

from shapely.geometry import Point


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. MOHID 데이터 로더 — 실데이터 교체 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MOHIDDataLoader:
    """
    MOHID 시뮬레이션 결과를 노드별 동적 피처로 변환하는 로더.

    현재 상태: load_stub() — 기존 해수유동 샘플 데이터로 보간(IDW)
    교체 방법: load_real()  — 실제 MOHID 출력 파일 경로를 넘기면 됨

    동적 피처 컬럼 (self.dynamic_df):
        node_idx        : GeoDataFrame 인덱스와 매핑
        particle_count  : 해당 노드에 도달한 입자 수
        residence_time  : 평균 체류 시간 (hr)
        u_mean          : 평균 동서 유속 (m/s)
        v_mean          : 평균 남북 유속 (m/s)
        flow_speed      : 평균 유속 크기 (m/s)
        flow_dir_deg    : 평균 유향 (도, 0=동쪽, CCW)
    """

    # ── 실데이터가 왔을 때 기대하는 컬럼명 스펙 ──────────────────────────────
    REAL_PARTICLE_COLS  = ["node_id", "particle_count", "residence_time_hr"]
    REAL_FLOW_COLS      = ["time", "lat", "lon", "u_current_ms", "v_current_ms"]

    def __init__(self, data_root: str | Path):
        self.data_root   = Path(data_root)
        self.dynamic_df: Optional[pd.DataFrame] = None
        self.flow_df:    Optional[pd.DataFrame] = None   # 격자 전체 유동장
        self._is_stub    = True

    # ─────────────────────────────────────────────────────────────────────────
    # STUB: 기존 샘플 데이터 사용 (현재 단계)
    # ─────────────────────────────────────────────────────────────────────────

    def load_stub(
        self,
        flow_file: str = "Mohid_prepare_data/namhae_water_flow_2025_Q1.txt",
    ) -> None:
        """
        [STUB] 샘플 해수유동 데이터로 동적 피처 생성.

        실제 MOHID 입자 궤적 결과가 없으므로:
          - particle_count  : id_count × 난수 스케일 (임시 프록시)
          - residence_time  : 유속 역수 기반 추정값 (임시 프록시)
          - u_mean, v_mean  : 노드 위치에서 IDW 보간한 실측 유동값

        MOHID 실데이터가 오면 load_real()로 교체하세요.
        """
        path = self.data_root / flow_file
        if not path.exists():
            raise FileNotFoundError(f"유동장 파일 없음: {path}")

        df = pd.read_csv(str(path), parse_dates=["time"])
        df = df.dropna(subset=["u_current_ms", "v_current_ms"])

        # 시간 평균 유동장 (격자 전체)
        self.flow_df = (
            df.groupby(["lat", "lon"])
            .agg(u_mean=("u_current_ms", "mean"), v_mean=("v_current_ms", "mean"))
            .reset_index()
        )
        self.flow_df["flow_speed"]   = np.hypot(self.flow_df["u_mean"], self.flow_df["v_mean"])
        self.flow_df["flow_dir_deg"] = np.degrees(np.arctan2(self.flow_df["v_mean"], self.flow_df["u_mean"]))

        self._is_stub = True
        print(f"✅ [STUB] 유동장 로드: {len(self.flow_df):,}개 격자점 "
              f"(시간 범위: {df['time'].min().date()} ~ {df['time'].max().date()})")

    def interpolate_to_nodes(self, gdf: gpd.GeoDataFrame, k: int = 5) -> pd.DataFrame:
        """
        IDW(역거리 가중 보간)로 각 노드 위치의 유동값(u, v) 추정.

        Parameters
        ----------
        gdf : 노드 GeoDataFrame (Latitude, Longitude 컬럼 필요)
        k   : 보간에 사용할 최근접 격자점 수

        Returns
        -------
        DataFrame: node_idx, u_mean, v_mean, flow_speed, flow_dir_deg,
                   particle_count(stub), residence_time(stub)
        """
        assert self.flow_df is not None, "load_stub() 또는 load_real()을 먼저 호출하세요."

        flow_lats = self.flow_df["lat"].values
        flow_lons = self.flow_df["lon"].values

        records = []
        np.random.seed(42)  # STUB 재현성

        for idx, row in gdf.iterrows():
            # 최근접 k개 격자점 거리 계산
            dlat = flow_lats - row["Latitude"]
            dlon = flow_lons - row["Longitude"]
            dist = np.hypot(dlat, dlon) + 1e-9

            knn_idx = np.argpartition(dist, min(k, len(dist) - 1))[:k]
            weights = 1.0 / dist[knn_idx] ** 2
            weights /= weights.sum()

            u  = float(np.dot(weights, self.flow_df["u_mean"].values[knn_idx]))
            v  = float(np.dot(weights, self.flow_df["v_mean"].values[knn_idx]))
            spd = float(np.hypot(u, v))
            dr  = float(np.degrees(np.arctan2(v, u)))

            # ── STUB 프록시 피처 ──────────────────────────────────────────
            # [교체] particle_count → MOHID 입자 도달량 CSV 직접 매핑
            # [교체] residence_time → MOHID 체류시간 CSV 직접 매핑
            stub_particles = (
                row["id_count"] * np.random.uniform(50, 100)
                if row["id_count"] > 0
                else spd * np.random.uniform(1000, 5000)
            )
            stub_residence = (1.0 / (spd + 0.01)) * np.random.uniform(0.8, 1.2) * 10.0

            records.append({
                "node_idx":      idx,
                "particle_count": stub_particles,   # [STUB] → 실측값 교체 대상
                "residence_time": stub_residence,   # [STUB] → 실측값 교체 대상
                "u_mean":         u,
                "v_mean":         v,
                "flow_speed":     spd,
                "flow_dir_deg":   dr,
            })

        self.dynamic_df = pd.DataFrame(records).set_index("node_idx")
        print(f"✅ [STUB] 노드별 동적 피처 생성 완료: {len(self.dynamic_df)}개 노드")
        return self.dynamic_df

    # ─────────────────────────────────────────────────────────────────────────
    # REAL: 실데이터 교체 진입점 — 추후 여기만 구현
    # ─────────────────────────────────────────────────────────────────────────

    def load_real(
        self,
        particle_csv:  str,
        flow_nc:       Optional[str] = None,
        residence_csv: Optional[str] = None,
    ) -> None:
        """
        [REAL] 실제 MOHID 시뮬레이션 결과 로드.

        Parameters
        ----------
        particle_csv  : MOHID 입자 궤적 집계 CSV
                        필수 컬럼: node_id, particle_count, residence_time_hr
        flow_nc       : MOHID 유동장 NetCDF (u, v, lat, lon, time)
                        없으면 particle_csv의 u/v 컬럼 사용
        residence_csv : 노드별 평균 체류시간 CSV (particle_csv에 포함된 경우 생략 가능)

        교체 후 이어서 호출:
            loader.interpolate_to_nodes(gdf)   # 내부 dynamic_df 갱신
        """
        # ── particle_csv 로드 ──────────────────────────────────────────────
        df_p = pd.read_csv(particle_csv)
        missing = [c for c in self.REAL_PARTICLE_COLS if c not in df_p.columns]
        if missing:
            raise ValueError(
                f"particle_csv에 필수 컬럼 누락: {missing}\n"
                f"필요 컬럼: {self.REAL_PARTICLE_COLS}"
            )

        # ── flow NetCDF 로드 (xarray 사용) ────────────────────────────────
        if flow_nc:
            try:
                import xarray as xr
                ds = xr.open_dataset(flow_nc)
                u = ds["u"].mean(dim="time").values
                v = ds["v"].mean(dim="time").values
                lats = ds["lat"].values
                lons = ds["lon"].values
                self.flow_df = pd.DataFrame({
                    "lat": lats.ravel(),
                    "lon": lons.ravel(),
                    "u_mean": u.ravel(),
                    "v_mean": v.ravel(),
                })
                self.flow_df["flow_speed"]   = np.hypot(self.flow_df["u_mean"], self.flow_df["v_mean"])
                self.flow_df["flow_dir_deg"] = np.degrees(
                    np.arctan2(self.flow_df["v_mean"], self.flow_df["u_mean"])
                )
            except ImportError:
                raise ImportError("pip install xarray netcdf4 가 필요합니다.")

        self._real_particle_df = df_p
        self._is_stub = False
        print(f"✅ [REAL] MOHID 입자 데이터 로드: {len(df_p)}개 노드")

    def is_stub(self) -> bool:
        return self._is_stub


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 노드 피처 빌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NodeFeatureBuilder:
    """
    노드 피처 텐서 [N, F] 구성.

    ┌─────────────────────────────────────────────────────────────────┐
    │  정적 피처 (Static)  — 시간 불변, GeoDataFrame에서 직접 추출  │
    │  idx  컬럼명              설명                                  │
    │   0   port_dist_km        가장 가까운 어항까지의 거리 (km)      │
    │   1   river_dist_km       가장 가까운 하천 방출점까지 거리(km)  │
    │   2   expo_flag           여수 박람회 관할 여부 (0/1)           │
    │   3   is_yeosu            여수 관할 섬 여부 (0/1)               │
    ├─────────────────────────────────────────────────────────────────┤
    │  동적 피처 (Dynamic) — MOHID 결과, 시간 스텝마다 갱신          │
    │  idx  컬럼명              설명                          [STUB?] │
    │   4   particle_count_n    노드 도달 입자 수 (정규화)   [STUB]  │
    │   5   residence_time_n    체류 시간 hr (정규화)         [STUB]  │
    │   6   u_mean              평균 동서 유속 m/s                    │
    │   7   v_mean              평균 남북 유속 m/s                    │
    │   8   flow_speed          유속 크기 m/s                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  관측 마스크                                                     │
    │   9   label_mask          관측값 보유 여부 (0/1)                │
    │  10   label_count_n       정규화된 수거량 (미관측=0)            │
    └─────────────────────────────────────────────────────────────────┘
    총 F = 11 차원
    """

    N_STATIC  = 4
    N_DYNAMIC = 5
    N_LABEL   = 2
    N_TOTAL   = 11

    FEATURE_NAMES = [
        "port_dist_km",      # 0 static
        "river_dist_km",     # 1 static
        "expo_flag",         # 2 static
        "is_yeosu",          # 3 static
        "particle_count_n",  # 4 dynamic [STUB]
        "residence_time_n",  # 5 dynamic [STUB]
        "u_mean",            # 6 dynamic
        "v_mean",            # 7 dynamic
        "flow_speed",        # 8 dynamic
        "label_mask",        # 9  label
        "label_count_n",     # 10 label
    ]

    def __init__(
        self,
        gdf_nodes:   gpd.GeoDataFrame,
        gdf_ports:   Optional[gpd.GeoDataFrame],
        gdf_rivers:  Optional[gpd.GeoDataFrame],
        dynamic_df:  pd.DataFrame,
    ):
        self.gdf    = gdf_nodes.reset_index(drop=True)
        self.ports  = gdf_ports
        self.rivers = gdf_rivers
        self.dyn    = dynamic_df

    def _dist_to_nearest_km(
        self,
        gdf_target: Optional[gpd.GeoDataFrame],
        default_km: float = 999.0,
    ) -> np.ndarray:
        """각 노드에서 target GDF 내 최근접 피처까지의 거리(km) 배열 반환."""
        if gdf_target is None or gdf_target.empty:
            return np.full(len(self.gdf), default_km)

        nodes_proj  = self.gdf.to_crs("EPSG:5179")
        target_proj = gdf_target.to_crs("EPSG:5179")
        target_union = target_proj.geometry.union_all()

        dists = nodes_proj.geometry.apply(
            lambda p: p.distance(target_union) / 1000.0
        ).values
        return dists

    def _minmax(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    def build(self) -> np.ndarray:
        """
        노드 피처 행렬 [N, 11] (numpy float32) 반환.
        PyTorch 사용 시 torch.tensor(x, dtype=torch.float) 로 변환.
        """
        N = len(self.gdf)
        X = np.zeros((N, self.N_TOTAL), dtype=np.float32)

        # ── 정적 피처 ──────────────────────────────────────────────────────
        port_dist  = self._dist_to_nearest_km(self.ports)
        river_dist = self._dist_to_nearest_km(self.rivers)

        X[:, 0] = self._minmax(port_dist)
        X[:, 1] = self._minmax(river_dist)
        X[:, 2] = self.gdf["is_yeosu"].astype(float).values  # expo_flag (여수 = 박람회 구역)
        X[:, 3] = self.gdf["is_yeosu"].astype(float).values

        # ── 동적 피처 (MOHID) ──────────────────────────────────────────────
        dyn = self.dyn.reindex(self.gdf.index)  # 인덱스 정렬

        X[:, 4] = self._minmax(dyn["particle_count"].fillna(0).values)
        X[:, 5] = self._minmax(dyn["residence_time"].fillna(0).values)
        X[:, 6] = dyn["u_mean"].fillna(0).values
        X[:, 7] = dyn["v_mean"].fillna(0).values
        X[:, 8] = dyn["flow_speed"].fillna(0).values

        # ── 레이블 마스크 ──────────────────────────────────────────────────
        is_obs = (self.gdf["node_type"] == "observed").astype(float).values
        X[:, 9]  = is_obs
        X[:, 10] = self._minmax(self.gdf["id_count"].values) * is_obs

        print(f"✅ 노드 피처 텐서 구성: shape={X.shape}  "
              f"(정적 {self.N_STATIC}d + 동적 {self.N_DYNAMIC}d + 레이블 {self.N_LABEL}d)")
        return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 그래프 빌더 — 해수유동 기반 방향성 그래프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GraphBuilder:
    """
    해수유동 벡터 기반 방향성 그래프 (Directed Weighted Graph) 생성.

    Edge 생성 원칙:
      1. 거리 필터  : 직선거리 max_dist_km 이내 노드 쌍만 후보
      2. 유동 정렬  : 노드 i의 평균 유동 벡터와 i→j 방향 벡터의 cos 유사도 계산
      3. 임계값 필터: cos_threshold 이상인 Edge만 생성 (유동이 j 방향을 향할 때만)
      4. 가중치     : w = cos_sim × exp(-dist / decay_km)
                        → 유동 정렬도 ↑, 거리 ↓ 일수록 강한 연결

    Parameters
    ----------
    max_dist_km    : Edge 후보 최대 거리 (기본 200km — 남해안 전체 커버)
    cos_threshold  : 유동 정렬 cos 임계값 (기본 0.0 — 같은 방향 반구)
    decay_km       : 거리 감쇠 스케일 (기본 100km)
    """

    def __init__(
        self,
        gdf_nodes:     gpd.GeoDataFrame,
        mohid_loader:  MOHIDDataLoader,
        gdf_ports:     Optional[gpd.GeoDataFrame] = None,
        gdf_rivers:    Optional[gpd.GeoDataFrame] = None,
        max_dist_km:   float = 200.0,
        cos_threshold: float = 0.0,
        decay_km:      float = 100.0,
    ):
        self.gdf      = gdf_nodes.reset_index(drop=True)
        self.loader   = mohid_loader
        self.ports    = gdf_ports
        self.rivers   = gdf_rivers
        self.max_dist = max_dist_km
        self.cos_thr  = cos_threshold
        self.decay    = decay_km

        self.X:          Optional[np.ndarray] = None
        self.edge_index: Optional[np.ndarray] = None   # [2, E]
        self.edge_attr:  Optional[np.ndarray] = None   # [E, 3]: cos, dist, weight
        self.node_meta:  Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # 핵심 로직: 유동 기반 Edge 생성
    # ─────────────────────────────────────────────────────────────────────────

    def _build_edges(self, dynamic_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        해수유동 정렬 기반 방향성 Edge 생성.

        Returns
        -------
        edge_index : [2, E] int64 — (src_i, dst_j) 인덱스 쌍
        edge_attr  : [E, 3] float32 — (cos_alignment, dist_km, weight)
        """
        N   = len(self.gdf)
        xs  = self.gdf["x_5179"].values   # UTM-K X (m)
        ys  = self.gdf["y_5179"].values   # UTM-K Y (m)

        dyn = dynamic_df.reindex(self.gdf.index)
        us  = dyn["u_mean"].fillna(0).values
        vs  = dyn["v_mean"].fillna(0).values

        src_list, dst_list, attr_list = [], [], []

        for i in range(N):
            u_i = float(us[i])
            v_i = float(vs[i])
            flow_mag = float(np.hypot(u_i, v_i)) + 1e-9

            for j in range(N):
                if i == j:
                    continue

                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_m  = float(np.hypot(dx, dy))
                dist_km = dist_m / 1000.0

                if dist_km > self.max_dist:
                    continue

                # 유동 벡터와 i→j 방향 벡터의 cos 유사도
                dir_mag = dist_m + 1e-9
                cos_sim = (u_i * dx + v_i * dy) / (flow_mag * dir_mag)

                if cos_sim < self.cos_thr:
                    continue  # 유동이 j 방향을 향하지 않으면 Edge 생략

                weight = float(cos_sim * np.exp(-dist_km / self.decay))

                src_list.append(i)
                dst_list.append(j)
                attr_list.append([cos_sim, dist_km, weight])

        if not src_list:
            raise RuntimeError(
                "Edge가 하나도 생성되지 않았습니다. "
                "cos_threshold 또는 max_dist_km 값을 조정하세요."
            )

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr  = np.array(attr_list, dtype=np.float32)
        return edge_index, edge_attr

    # ─────────────────────────────────────────────────────────────────────────
    # 전체 빌드
    # ─────────────────────────────────────────────────────────────────────────

    def build(self) -> dict:
        """
        그래프 전체 구성.

        Returns
        -------
        dict with keys:
            X          : [N, 11] float32 노드 피처
            edge_index : [2, E] int64
            edge_attr  : [E, 3] float32
            y          : [N] float32 레이블 (미관측 노드 = 0)
            train_mask : [N] bool — 관측 노드 (학습용)
            test_mask  : [N] bool — 미관측 노드 (추론 대상)
            node_meta  : DataFrame — 노드 메타 정보
        """
        # 1. MOHID 보간
        dynamic_df = self.loader.interpolate_to_nodes(self.gdf)

        # 2. 노드 피처
        feat_builder = NodeFeatureBuilder(
            self.gdf, self.ports, self.rivers, dynamic_df
        )
        self.X = feat_builder.build()

        # 3. Edge 생성
        print("⚙️  해수유동 기반 방향성 Edge 생성 중... (N²={}회 탐색)".format(len(self.gdf)**2))
        self.edge_index, self.edge_attr = self._build_edges(dynamic_df)
        E = self.edge_index.shape[1]
        print(f"✅ Edge 생성 완료: {E}개  (평균 out-degree: {E/len(self.gdf):.1f})")

        # 4. 레이블 및 마스크
        y          = self.gdf["id_count"].values.astype(np.float32)
        train_mask = (self.gdf["node_type"] == "observed").values
        test_mask  = ~train_mask

        # 5. 노드 메타
        self.node_meta = self.gdf[["지역명", "grid_code", "node_type", "is_yeosu"]].copy()
        self.node_meta["u_mean"]         = dynamic_df["u_mean"].values
        self.node_meta["v_mean"]         = dynamic_df["v_mean"].values
        self.node_meta["particle_count"] = dynamic_df["particle_count"].values
        self.node_meta["is_stub"]        = self.loader.is_stub()

        result = dict(
            X=self.X,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            node_meta=self.node_meta,
        )

        self._print_summary(result)
        return result

    def _print_summary(self, g: dict) -> None:
        N, F = g["X"].shape
        E    = g["edge_index"].shape[1]
        print("\n" + "=" * 60)
        print("  📐 그래프 구조 요약")
        print("=" * 60)
        print(f"  노드 수 N      : {N}")
        print(f"  피처 차원 F    : {F}  {NodeFeatureBuilder.FEATURE_NAMES}")
        print(f"  엣지 수 E      : {E}")
        print(f"  평균 in-degree : {E/N:.1f}")
        print(f"  학습 노드      : {g['train_mask'].sum()} (관측)")
        print(f"  추론 노드      : {g['test_mask'].sum()} (미관측)")
        w = g["edge_attr"][:, 2]
        print(f"  엣지 가중치    : min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")
        stub_flag = "⚠️  STUB 데이터" if self.loader.is_stub() else "✅ REAL MOHID 데이터"
        print(f"  동적 피처 소스 : {stub_flag}")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch Geometric Data 객체 조립
    # ─────────────────────────────────────────────────────────────────────────

    def build_pyg_data(self):
        """
        PyTorch Geometric Data 객체 반환.
        torch_geometric 미설치 시 build()의 dict 반환으로 fallback.
        """
        g = self.build()

        if not _HAS_PYG:
            print("⚠️  PyG 미설치 — dict 반환. pip install torch-geometric 권장.")
            return g

        data = PygData(
            x          = torch.tensor(g["X"],          dtype=torch.float),
            edge_index = torch.tensor(g["edge_index"],  dtype=torch.long),
            edge_attr  = torch.tensor(g["edge_attr"],   dtype=torch.float),
            y          = torch.tensor(g["y"],           dtype=torch.float),
        )
        data.train_mask = torch.tensor(g["train_mask"], dtype=torch.bool)
        data.test_mask  = torch.tensor(g["test_mask"],  dtype=torch.bool)
        data.node_meta  = g["node_meta"]

        print(f"\n✅ PyG Data 객체 생성 완료: {data}")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # 시각화: 그래프 구조
    # ─────────────────────────────────────────────────────────────────────────

    def plot_graph(
        self,
        save_path:  Optional[str] = None,
        top_k_edge: int = 80,
    ) -> None:
        """
        노드-엣지 그래프 지도 시각화.

        - 엣지 색상: 가중치 크기 (밝을수록 강한 연결)
        - 엣지 화살표: 유동 방향 (i → j)
        - 노드 색상: 관측(빨강) / 미관측(파랑)
        - 노드 크기: 수거량 비례
        """
        assert self.edge_index is not None, "build() 또는 build_pyg_data()를 먼저 호출하세요."

        gdf = self.gdf
        xs  = gdf.geometry.x.values
        ys  = gdf.geometry.y.values
        is_obs = (gdf["node_type"] == "observed").values

        fig, ax = plt.subplots(figsize=(14, 9), facecolor="#0d1b2a")
        ax.set_facecolor("#0d1b2a")
        ax.tick_params(colors="#8892b0", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3f5f")

        # ── 상위 top_k_edge 엣지 렌더링 (가중치 순) ──────────────────────
        weights = self.edge_attr[:, 2]
        top_idx = np.argsort(weights)[-top_k_edge:]

        w_min, w_max = weights[top_idx].min(), weights[top_idx].max()
        cmap = plt.cm.plasma

        for ei in top_idx:
            i, j = self.edge_index[0, ei], self.edge_index[1, ei]
            w_norm = (weights[ei] - w_min) / (w_max - w_min + 1e-9)
            color  = cmap(w_norm)
            ax.annotate(
                "", xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=0.6 + w_norm * 1.2,
                    alpha=0.4 + w_norm * 0.5,
                    mutation_scale=8,
                ),
            )

        # ── 노드 렌더링 ───────────────────────────────────────────────────
        counts = gdf["id_count"].values
        max_c  = counts.max() if counts.max() > 0 else 1.0
        sizes  = (counts / max_c * 300).clip(min=40)

        for ntype, color, label in [
            ("observed",   "#E63946", "관측 노드"),
            ("unobserved", "#457B9D", "미관측 노드"),
        ]:
            mask = gdf["node_type"] == ntype
            ax.scatter(
                xs[mask], ys[mask],
                c=color, s=sizes[mask],
                edgecolors="white", linewidths=0.5,
                zorder=5, label=label, alpha=0.92,
            )

        for i, row in gdf.iterrows():
            ax.annotate(
                row["지역명"],
                (xs[i], ys[i]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6, color="#ccd6f6", zorder=6,
            )

        # ── 유동 벡터 퀴버 (격자 전체) ───────────────────────────────────
        if self.loader.flow_df is not None:
            fd = self.loader.flow_df
            # 범위 필터
            lat_range = (ys.min() - 0.5, ys.max() + 0.5)
            lon_range = (xs.min() - 0.5, xs.max() + 0.5)
            fd_sub = fd[
                fd["lat"].between(*lat_range) &
                fd["lon"].between(*lon_range)
            ]
            # 격자 간소화 (과밀 방지)
            step = max(1, len(fd_sub) // 300)
            fd_sub = fd_sub.iloc[::step]
            ax.quiver(
                fd_sub["lon"], fd_sub["lat"],
                fd_sub["u_mean"], fd_sub["v_mean"],
                color="#1a6b8a", alpha=0.25, scale=3,
                width=0.0012, headwidth=3,
                zorder=2, label="해수유동 벡터 (STUB)",
            )

        # 컬러바 (엣지 가중치)
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=w_min, vmax=w_max),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("Edge Weight (유동정렬 × 거리감쇠)", color="#8892b0", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#8892b0")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892b0", fontsize=7)

        ax.set_title(
            f"해수유동 기반 방향성 그래프  (상위 {top_k_edge}개 Edge)\n"
            f"{'⚠️ STUB 데이터' if self.loader.is_stub() else '✅ REAL MOHID 데이터'}",
            color="white", fontsize=12, pad=10,
        )
        ax.set_xlabel("Longitude (°E)", color="#8892b0", fontsize=8)
        ax.set_ylabel("Latitude (°N)",  color="#8892b0", fontsize=8)
        ax.legend(loc="lower left", fontsize=8,
                  facecolor="#1e2d3d", edgecolor="#2a3f5f", labelcolor="white")
        ax.grid(True, color="#1e3a5f", alpha=0.3, linestyle="--", linewidth=0.4)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#0d1b2a", edgecolor="none")
            print(f"💾 그래프 시각화 저장: {save_path}")
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

    # Step 1
    pipeline = SpatioTemporalDataPipeline(str(_ROOT / "final_data"))
    pipeline.load_nodes()
    pipeline.load_auxiliary()

    # Step 2
    loader = MOHIDDataLoader(str(_ROOT / "final_data"))
    loader.load_stub()

    builder = GraphBuilder(
        gdf_nodes   = pipeline.gdf_nodes,
        mohid_loader= loader,
        gdf_ports   = pipeline.gdf_ports,
        gdf_rivers  = pipeline.gdf_rivers,
        max_dist_km = 200.0,
        cos_threshold = 0.0,
        decay_km    = 100.0,
    )

    graph_data = builder.build()
    builder.plot_graph(save_path=str(_ROOT / "output" / "step2_graph.png"))

    # 피처 이름 출력
    print("\n[노드 피처 인덱스 맵]")
    for i, name in enumerate(NodeFeatureBuilder.FEATURE_NAMES):
        stub = " ← [STUB: MOHID 실데이터로 교체 필요]" if i in (4, 5) else ""
        print(f"  x[:, {i:2d}]  {name}{stub}")
