"""
해양쓰레기 예측 프로젝트 - PyTorch Geometric 그래프 구성 + Folium 시각화
"""
import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import folium

ROOT = "/Users/jeongwon/yeosu"
FINAL = os.path.join(ROOT, "final_data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────
# 유틸: haversine 거리 (km)
# ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────
rp = pd.read_csv(os.path.join(FINAL, "release_points_encoded.csv"))
islands = pd.read_csv(os.path.join(FINAL, "island_merged.csv"))
env = pd.read_csv(os.path.join(FINAL, "env_timeseries_merged.csv"), parse_dates=["datetime"])

print(f"방출점: {len(rp)}개 | 섬: {len(islands)}개")
print(f"환경변수: {len(env)}행, {env['datetime'].min()} ~ {env['datetime'].max()}")

# 환경변수 전체 평균 (그래프 엣지에 사용)
mean_u = env["u_current"].mean()
mean_v = env["v_current"].mean()
mean_speed = np.sqrt(mean_u**2 + mean_v**2)
print(f"평균 해류: u={mean_u:.4f}, v={mean_v:.4f}, speed={mean_speed:.4f} m/s")


# ─────────────────────────────────────────────────────────
# 1. 노드 피처 구성 [150, 6]
# ─────────────────────────────────────────────────────────
# 방출점 노드 (0~126): [lat, lon, is_beach, is_port, is_river, is_fishery]
rp_feats = rp[["lat", "lon", "type_beach", "type_port", "type_river", "type_fishery"]].values.astype(float)

# 섬 노드 (127~149): [lat, lon, 0, 0, 0, 0]
island_feats = np.zeros((len(islands), 6))
island_feats[:, 0] = islands["Latitude"].values
island_feats[:, 1] = islands["Longitude"].values

node_feats = np.vstack([rp_feats, island_feats])  # [150, 6]
print(f"\n노드 피처 shape: {node_feats.shape}")

# 타깃 y: 방출점=0, 섬=수량(개)
y = np.zeros(len(rp) + len(islands))
y[len(rp):] = islands["수량(개)"].values

# train_mask: 섬 노드만 True
train_mask = np.zeros(len(rp) + len(islands), dtype=bool)
train_mask[len(rp):] = True


# ─────────────────────────────────────────────────────────
# 2. 엣지 구성 (방출점 → 섬, 거리 < 80km)
# ─────────────────────────────────────────────────────────
DIST_THRESH = 80.0  # km
N_RP = len(rp)
N_IS = len(islands)

# source_count 총합 (비율 계산용)
total_sc = islands["source_count"].sum()

edges_src, edges_dst = [], []
edge_feats = []

for i in range(N_RP):
    lat_r, lon_r = rp_feats[i, 0], rp_feats[i, 1]
    for j in range(N_IS):
        lat_s = islands.iloc[j]["Latitude"]
        lon_s = islands.iloc[j]["Longitude"]

        dist = haversine(lat_r, lon_r, lat_s, lon_s)
        if dist >= DIST_THRESH:
            continue

        # 방출점→섬 방향 벡터 (경도차, 위도차) 근사
        dlon = lon_s - lon_r
        dlat = lat_s - lat_r

        # 해류가 방출점→섬 방향이면 가중치 × 1.5 (내적 > 0)
        dot = mean_u * dlon + mean_v * dlat
        weight = 1.5 if dot > 0 else 1.0

        # source_count_비율: 섬의 source_count / 전체 합
        sc_ratio = islands.iloc[j]["source_count"] / total_sc

        # 엣지 피처: [거리, u_current, v_current, speed, sc_ratio]  (weight 반영: dist에 inverse)
        edge_feats.append([
            dist * (1.0 / weight),   # weight가 높을수록 효과적 거리 줄임
            mean_u,
            mean_v,
            mean_speed,
            sc_ratio,
        ])

        edges_src.append(i)
        edges_dst.append(N_RP + j)

print(f"\n엣지 수 (방출점→섬, 거리<{DIST_THRESH}km): {len(edges_src)}")

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
edge_attr = torch.tensor(edge_feats, dtype=torch.float)
x = torch.tensor(node_feats, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)
mask_tensor = torch.tensor(train_mask, dtype=torch.bool)


# ─────────────────────────────────────────────────────────
# 3. PyG Data 객체 저장
# ─────────────────────────────────────────────────────────
data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y_tensor,
    train_mask=mask_tensor,
)

print(f"\n[Data 객체]")
print(f"  data.x:          {data.x.shape}")
print(f"  data.edge_index: {data.edge_index.shape}")
print(f"  data.edge_attr:  {data.edge_attr.shape}")
print(f"  data.y:          {data.y.shape}")
print(f"  data.train_mask: {data.train_mask.shape}  (True: {data.train_mask.sum().item()}개)")

pt_path = os.path.join(FINAL, "graph_data.pt")
torch.save(data, pt_path)
print(f"\n  저장: {pt_path}")


# ─────────────────────────────────────────────────────────
# 4. Folium 시각화
# ─────────────────────────────────────────────────────────
center_lat = node_feats[:, 0].mean()
center_lon = node_feats[:, 1].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")

# 엣지: 거리 가까울수록 진하게 (투명도 조절)
dist_vals = edge_attr[:, 0].numpy()
dist_min, dist_max = dist_vals.min(), dist_vals.max()

for k in range(len(edges_src)):
    i, j_idx = edges_src[k], edges_dst[k] - N_RP
    lat_r, lon_r = rp_feats[i, 0], rp_feats[i, 1]
    lat_s = islands.iloc[j_idx]["Latitude"]
    lon_s = islands.iloc[j_idx]["Longitude"]

    raw_dist = float(dist_vals[k])
    # 거리 가까울수록 opacity 높게 (0.15 ~ 0.55)
    norm = (raw_dist - float(dist_min)) / (float(dist_max) - float(dist_min) + 1e-9)
    opacity = round(0.55 - norm * 0.40, 4)

    folium.PolyLine(
        [(float(lat_r), float(lon_r)), (float(lat_s), float(lon_s))],
        color="#888888",
        weight=0.8,
        opacity=opacity,
    ).add_to(m)

# 방출점 마커
TYPE_COLOR = {
    "beach": "orange",
    "port": "blue",
    "river": "green",
    "fishery": "red",
}
for _, row in rp.iterrows():
    if row["type_beach"]:
        t = "beach"
    elif row["type_port"]:
        t = "port"
    elif row["type_river"]:
        t = "river"
    else:
        t = "fishery"
    folium.CircleMarker(
        location=[float(row["lat"]), float(row["lon"])],
        radius=4,
        color=TYPE_COLOR[t],
        fill=True,
        fill_color=TYPE_COLOR[t],
        fill_opacity=0.7,
        popup=f"방출점 {int(row['id'])} ({t})",
    ).add_to(m)

# 섬 마커
for _, row in islands.iterrows():
    folium.CircleMarker(
        location=[float(row["Latitude"]), float(row["Longitude"])],
        radius=7,
        color="black",
        fill=True,
        fill_color="black",
        fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['지역명']}</b><br>수량: {int(row['수량(개)'])}개<br>무게: {row['무게(kg)']}kg",
            max_width=180,
        ),
    ).add_to(m)

# 범례 (HTML 삽입)
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:12px 16px; border-radius:8px;
     border:1px solid #ccc; font-size:13px; line-height:1.8;">
  <b>범례</b><br>
  <span style="color:orange;">●</span> 방출점 – 해수욕장 (beach)<br>
  <span style="color:blue;">●</span> 방출점 – 항구 (port)<br>
  <span style="color:green;">●</span> 방출점 – 하천 (river)<br>
  <span style="color:red;">●</span> 방출점 – 어업 (fishery)<br>
  <span style="color:black;">●</span> 섬 노드 (수량 팝업)<br>
  <span style="color:#888;">─</span> 엣지 (거리<80km, 진할수록 가까움)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

html_path = os.path.join(OUT, "graph_visualization.html")
m.save(html_path)
print(f"\nFolium 시각화 저장: {html_path}")
print("\n완료!")
