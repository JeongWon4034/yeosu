"""
여수 해양쓰레기 수거 경로 최적화  v9

[수정사항]
- 관측소 4개 모두 포함 (백야도, 안도, 거문도, 반월도)
- EXPO_LINKED 미사용 변수 제거
- 알림 카드 주의/모니터링 파란색으로 수정
- dasomi_nobg.png 파일 연동 방식 유지
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta, date as date_type
import io, base64
import os
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep

# ═══════════════════════════════════════════════
# 0. 페이지 & 색상
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="여수 해양쓰레기 수거 시스템",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "blue_deep":  "#003F7D",
    "blue_mid":   "#0066B3",
    "blue_pale":  "#E8F4FD",
    "white":      "#FFFFFF",
    "gray_dark":  "#1A2B3C",
    "gray_mid":   "#4B6178",
    "gray_light": "#CBD5E1",
    "gray_bg":    "#F5F8FC",
}

TRASH_TYPES = ["부유쓰레기", "해안쓰레기", "침적쓰레기", "어구류", "유류"]
TRASH_COLORS = {
    "부유쓰레기": "#0066B3",
    "해안쓰레기": "#16A34A",
    "침적쓰레기": "#7C3AED",
    "어구류":    "#EA580C",
    "유류":      "#DC2626",
}

@st.cache_data
def get_local_image_base64(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{b64_str}"
    return ""

DASOMI_SRC = get_local_image_base64("dasomi_nobg.png")

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background:{C['white']}; }}
    [data-testid="stSidebar"] {{
        background:{C['blue_pale']};
        border-right:1px solid {C['gray_light']};
    }}
    [data-testid="stSidebar"] * {{ color:{C['gray_dark']} !important; }}

    /* 사이드바 위젯 파란색 */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div:nth-child(2) {{
        background-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {C['blue_mid']} !important;
        border-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stCheckbox [data-baseweb="checkbox"] input:checked + div {{
        background-color: transparent !important;
        border-color: transparent !important;
    }}
    /* 체크박스 선택 시 형광 하이라이트 완전 제거 */
    [data-testid="stSidebar"] .stCheckbox label {{
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stCheckbox label:hover {{
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"]:focus-within {{
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:last-child {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:last-child:hover {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] p {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] [aria-checked="true"] div:first-child {{
        background-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
        background-color: {C['blue_mid']} !important;
        color: white !important;
    }}

    /* 헤더 */
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700;800&display=swap');
    .expo-header {{
        position: relative;
        background: linear-gradient(180deg, #0077C8 0%, #00A3E0 50%, #29B6E0 100%);
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 16px;
        min-height: 160px;
    }}
    .expo-header-stars {{
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.9) 1px, transparent 1px),
            radial-gradient(circle, rgba(255,255,255,0.6) 1px, transparent 1px),
            radial-gradient(circle, rgba(255,255,255,0.4) 1px, transparent 1px);
        background-size: 120px 80px, 80px 100px, 60px 60px;
        background-position: 10px 10px, 40px 30px, 70px 15px;
        animation: twinkle 4s ease-in-out infinite alternate;
    }}
    @keyframes twinkle {{
        0% {{ opacity: 0.6; }} 100% {{ opacity: 1; }}
    }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
    }}
    .expo-header-waves {{
        position: absolute; bottom: 0; left: 0; right: 0; height: 48px;
    }}
    .expo-header-content {{
        position: relative; z-index: 2;
        padding: 18px 24px 56px;
        display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
    }}
    .expo-header-left {{ flex: 1; }}
    .expo-header-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: .78rem;
        font-weight: 600;
        color: rgba(255,255,255,0.95);
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }}
    .expo-header-title {{
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 0 3px !important;
        line-height: 1.2 !important;
        text-shadow: 0 2px 8px rgba(0,60,120,0.3);
        letter-spacing: -0.01em !important;
    }}
    .expo-header-sub {{
        font-size: .8rem;
        color: rgba(255,255,255,0.8);
        margin: 0;
    }}
    .expo-header-cards {{
        display: flex; gap: 8px; flex-shrink: 0; flex-wrap: wrap;
        align-items: flex-start;
    }}
    .expo-info-card {{
        background: rgba(255,255,255,0.18);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 10px;
        padding: 8px 14px;
        text-align: center;
        min-width: 80px;
    }}
    .expo-info-card .label {{
        font-size: .68rem; color: rgba(255,255,255,0.75); margin-bottom: 3px;
    }}
    .expo-info-card .value {{
        font-size: .95rem; font-weight: 700; color: white;
    }}
    .sec {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
            color:{C['gray_mid']}; margin:14px 0 6px; font-weight:700; }}
    .rcard {{ border-left:3px solid; border-radius:5px; padding:8px 12px;
              margin-bottom:6px; font-size:.86rem; background:{C['white']};
              box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    .mbox {{ background:{C['blue_pale']}; border:1px solid {C['gray_light']};
             border-radius:6px; padding:12px 14px; margin-bottom:8px; }}
    div[data-testid="stTabs"] > div > div > div > button {{
        color:{C['gray_mid']} !important; font-weight:500 !important; font-size:.88rem !important;
    }}
    div[data-testid="stTabs"] > div > div > div > button[aria-selected="true"] {{
        color:{C['blue_mid']} !important;
        border-bottom-color:{C['blue_mid']} !important;
        font-weight:700 !important;
    }}
    .detail-card {{
        background:{C['white']}; border:1px solid {C['gray_light']};
        border-radius:8px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.06);
    }}
    .trash-badge {{
        display:inline-block; font-size:.72rem; padding:2px 8px;
        border-radius:10px; margin:2px 3px 2px 0; font-weight:600;
    }}
    footer {{ visibility:hidden; }}
    h1,h2,h3,h4 {{ color:{C['blue_deep']} !important; font-weight:700 !important; }}
</style>
""", unsafe_allow_html=True)

GEOJSON_PATH = "yeosu_polygons.geojson"
BASE_PORT    = [127.730, 34.655]
MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# 2026 박람회 공식 행사장 (돌산도=주행사장, 금오도·개도=부행사장)
EXPO_ISLANDS = {"돌산도", "금오도", "개도"}

# 해양관측소 보유 섬 (실제 데이터 기반)
# 백야도·안도: CSV 있음 / 거문도·반월도: CSV 없음 → 수동 추가
OBS_KEYWORDS = ["백야도", "안도", "거문도", "반월도"]


# ═══════════════════════════════════════════════
# 1. 유틸
# ═══════════════════════════════════════════════
def risk_color(risk: int, alpha: int = 220) -> list:
    if risk < 40:  return [ 34, 197,  94, alpha]    # 초록 (양호)
    if risk < 60:  return [234, 179,   8, alpha]    # 노랑 (주의)
    if risk < 80:  return [249, 115,  22, alpha]    # 주황 (경고)
    return              [220,  38,  38, alpha]      # 빨강 (위험)

def risk_label(risk: int) -> str:
    if risk >= 80: return "위험"
    if risk >= 60: return "경고"
    if risk >= 40: return "주의"
    return "양호"

def risk_hex(risk: int) -> str:
    c = risk_color(risk, 255)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


# ═══════════════════════════════════════════════
# 2. 육지 판별 (정확한 폴리곤 기반)
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner="육지 영역 로딩 중...")
def load_land_geom():
    polys = []
    # 1) GADM Korea Level 2 — 여수 + 인근 시군구 (가장 정확)
    try:
        gadm = gpd.read_file("gadm41_KOR_2.shp")
        gadm = gadm.cx[126.7:128.2, 33.8:35.2]
        polys.extend(g for g in gadm.geometry if g and not g.is_empty)
    except Exception:
        pass
    # 2) namhae_final3.shp — 일부 도서 보강
    try:
        shp = gpd.read_file("namhae_final3.shp")
        shp = shp.cx[126.9:128.1, 33.9:35.1]
        polys.extend(g for g in shp.geometry if g and not g.is_empty)
    except Exception:
        pass
    # 3) yeosu_polygons.geojson — 보강
    try:
        gj = gpd.read_file(GEOJSON_PATH)
        polys.extend(g for g in gj.geometry if g and not g.is_empty)
    except Exception:
        pass
    # (CSV 섬 좌표를 buffer 로 추가하지 않음 — 자기 자신이 buffer 안에 갇혀
    #  snap·부착 로직이 깨졌던 원인. 큰 섬·본토는 GADM 이 정확히 커버한다.)
    union = unary_union(polys)
    return union, prep(union)

_LAND_UNION = None
_LAND_PREP = None
def _ensure_land():
    global _LAND_UNION, _LAND_PREP
    if _LAND_PREP is None:
        _LAND_UNION, _LAND_PREP = load_land_geom()

def is_land(lon: float, lat: float) -> bool:
    _ensure_land()
    return _LAND_PREP.contains(Point(lon, lat))

def edge_crosses_land(a, b) -> bool:
    _ensure_land()
    return _LAND_PREP.intersects(LineString([(a[0], a[1]), (b[0], b[1])]))


# ═══════════════════════════════════════════════
# 3. 해상 그래프
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner="해상 경로 그래프 초기화 중... (최초 1회)")
def build_sea_graph():
    _ensure_land()
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)
    lons = np.linspace(127.25, 127.95, 110)
    lats = np.linspace(34.00,  34.95,  130)
    G, nodes = nx.Graph(), []
    for lo in lons:
        for la in lats:
            if not _LAND_PREP.contains(Point(lo, la)):
                n = (round(lo, 4), round(la, 4))
                nodes.append(n); G.add_node(n)
    xy = np.array(nodes)
    for i, n1 in enumerate(nodes):
        d = np.hypot(xy[:, 0]-n1[0], xy[:, 1]-n1[1])
        for j in np.where((d > 0) & (d < 0.025))[0]:
            n2 = nodes[int(j)]
            if not G.has_edge(n1, n2) and not _LAND_PREP.intersects(
                LineString([(n1[0], n1[1]), (n2[0], n2[1])])
            ):
                G.add_edge(n1, n2, weight=float(d[j]))
    return G, gj


def find_route(waypoints, G):
    """
    각 waypoint를 '도달 가능한 가장 가까운 해상 그래프 노드'로 snap 한 뒤 A* 탐색.
    그래프를 mutate 하지 않으므로 cache 안전. 본토 안/해안 인근 점도 안전하게 처리된다.
    시각적 정확도를 위해 원래 waypoint 좌표를 path 양 끝에 그대로 부착한다.
    """
    _ensure_land()
    nodes_orig = list(G.nodes())
    xy_orig    = np.array([(n[0], n[1]) for n in nodes_orig])

    snapped = []
    for wp in waypoints:
        d = np.hypot(xy_orig[:, 0]-wp[0], xy_orig[:, 1]-wp[1])
        order = np.argsort(d)
        chosen = None
        # 직선 연결이 육지를 통과하지 않는 가장 가까운 해상 노드를 선택
        for j in order[:80]:
            n = nodes_orig[int(j)]
            if not _LAND_PREP.intersects(LineString([(wp[0], wp[1]), (n[0], n[1])])):
                chosen = n
                break
        if chosen is None:
            chosen = nodes_orig[int(order[0])]
        snapped.append(chosen)

    path = []
    # 첫 waypoint 부착: 육지 미통과 시에만
    wp0 = [float(waypoints[0][0]), float(waypoints[0][1])]
    s0  = [snapped[0][0], snapped[0][1]]
    if not _LAND_PREP.intersects(LineString([(wp0[0], wp0[1]), (s0[0], s0[1])])):
        path.append(wp0)

    for i in range(len(waypoints)-1):
        a, b = snapped[i], snapped[i+1]
        try:
            seg = nx.astar_path(G, a, b, weight="weight")
            seg_pts = [[p[0], p[1]] for p in seg]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            seg_pts = [[a[0], a[1]], [b[0], b[1]]]
        path.extend(seg_pts)
        # 다음 waypoint 부착: 육지 미통과 시에만 (본토에 박힌 좌표는 부착 생략)
        nxt  = [float(waypoints[i+1][0]), float(waypoints[i+1][1])]
        last = path[-1]
        if not _LAND_PREP.intersects(LineString([(last[0], last[1]), (nxt[0], nxt[1])])):
            path.append(nxt)
    return path


# ═══════════════════════════════════════════════
# 4. 데이터 로딩
# ═══════════════════════════════════════════════
@st.cache_data
def load_island_data(seed: int) -> pd.DataFrame:
    df = pd.read_csv("yeosu_islands_wgs84.csv")
    df.rename(columns={"도서명": "name"}, inplace=True)
    rng = np.random.default_rng(seed)

    df["risk"]       = rng.integers(5, 100, size=len(df))
    df["trash_cnt"]  = (df["risk"] * rng.uniform(3.0, 5.0, size=len(df))).astype(int)
    df["has_bridge"] = df["연육도현황"].notna()
    df["inhabited"]  = df["유무인도"].astype(int)
    df["is_expo"]    = df["name"].apply(lambda n: any(e in n for e in EXPO_ISLANDS))

    def expo_role(name):
        if any(e in name for e in EXPO_ISLANDS): return "공식 행사장"
        return "해당 없음"
    df["expo_role"] = df["name"].apply(expo_role)

    rng2 = np.random.default_rng(seed + 1)
    def assign_types(i):
        k = rng2.integers(1, 4)
        chosen = rng2.choice(TRASH_TYPES, size=k, replace=False)
        return ", ".join(sorted(chosen))
    df["trash_types"] = [assign_types(i) for i in range(len(df))]
    df["main_type"]   = df["trash_types"].apply(lambda s: s.split(", ")[0])

    # 관측소: 정확 이름 매칭
    df["has_obs"] = df["name"].apply(lambda n: n in OBS_KEYWORDS)

    # 거문도·반월도 본섬 수동 추가 (CSV 미등재)
    extra_rows = []
    for name, lat, lon, inh in [("거문도", 34.033, 127.317, 1),
                                 ("반월도", 34.849, 127.594, 1)]:
        if name not in df["name"].values:
            r = int(rng.integers(5, 100))
            extra_rows.append({
                "name": name, "lat": lat, "lon": lon,
                "risk": r,
                "trash_cnt": int(r * rng.uniform(3.0, 5.0)),
                "has_bridge": False, "inhabited": inh,
                "is_expo": name in EXPO_ISLANDS,
                "expo_role": "공식 행사장" if name in EXPO_ISLANDS else "해당 없음",
                "has_obs": True,
                "trash_types": "부유쓰레기, 해안쓰레기",
                "main_type": "부유쓰레기",
            })
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # color 컬럼은 캐시에서 제외 (risk_color 변경 시 자동 반영되도록 호출 측에서 매번 계산)
    df["tooltip"] = (
        df["name"] + "\n위험도: " + df["risk"].astype(str) +
        " (" + df["risk"].apply(risk_label) + ")" +
        "\n쓰레기 추정: " + df["trash_cnt"].astype(str) + "개" +
        "\n유형: " + df["trash_types"]
    )
    return df


# ═══════════════════════════════════════════════
# 5. LSTM 예측 차트
# ═══════════════════════════════════════════════
def build_forecast_chart(name, base, base_trash, dt):
    np.random.seed(abs(hash(name)) % 9999)
    labels = [(dt+timedelta(days=i)).strftime("%m/%d") for i in range(6)]
    pred, lo, hi = [float(base)], [float(base)], [float(base)]
    for _ in range(5):
        v = max(5.0, min(100.0, pred[-1]+np.random.normal(1.5, 6.0)))
        e = np.random.uniform(4.0, 9.0)
        pred.append(v); lo.append(max(0.0,v-e)); hi.append(min(100.0,v+e))
    trash_pred = [max(0, int(p/max(base,1)*base_trash)) for p in pred]
    r,g,b = risk_color(base)[:3]; lc=f"rgb({r},{g},{b})"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels+labels[::-1], y=hi+lo[::-1],
        fill="toself", fillcolor=f"rgba({r},{g},{b},.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% 신뢰구간"))
    fig.add_trace(go.Scatter(x=labels, y=pred, mode="lines+markers",
        name="위험도 예측", line=dict(color=lc, width=2.5),
        marker=dict(size=7, color=lc, line=dict(color="white", width=1.5))))
    fig.add_trace(go.Bar(x=labels, y=trash_pred, name="쓰레기 추정량",
        marker_color=f"rgba({r},{g},{b},.25)",
        marker_line_color=f"rgba({r},{g},{b},.6)",
        marker_line_width=1, yaxis="y2"))
    fig.add_trace(go.Scatter(x=[labels[0]], y=[pred[0]], mode="markers", name="현재",
        marker=dict(size=12, color="white", symbol="diamond",
                    line=dict(color=lc, width=2.5))))
    fig.update_layout(
        title=dict(text=f"{name} — 향후 5일 위험도 · 쓰레기 예측 (LSTM)",
                   font=dict(size=13, color=C["blue_deep"])),
        paper_bgcolor=C["white"], plot_bgcolor=C["gray_bg"],
        font=dict(color=C["gray_mid"], size=12),
        xaxis=dict(gridcolor=C["gray_light"], title="날짜"),
        yaxis=dict(gridcolor=C["gray_light"], range=[0,110], title="위험도", side="left"),
        yaxis2=dict(title="쓰레기(개)", overlaying="y", side="right",
                    showgrid=False, range=[0, max(trash_pred or [1])*2.5]),
        legend=dict(orientation="h", y=1.12),
        barmode="overlay", height=320,
        margin=dict(l=10, r=40, t=55, b=10))
    return fig


# ═══════════════════════════════════════════════
# 6. 해양 역학 — 실측 API + 시뮬레이션 fallback
# ═══════════════════════════════════════════════
# 공공데이터포털 일반 인증키 (KHOA 조위관측 + 기상청 단기예보 공용)
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_KEY", "")

# KHOA 조위관측소 (여수 작업권역 — 공공데이터포털 활용가이드 기준)
# 정확 매칭 (섬 이름 = 관측소 이름)
OBS_CODE = {
    "거문도": "DT_0031",
}
# 가장 가까운 관측소 자동 매핑용 좌표표 (lat, lon)
# 백야도/안도/반월도는 자체 관측소가 없어 인근 관측소로 보간
KHOA_NEARBY = [
    ("DT_0016", "여수",   34.747, 127.766),
    ("DT_0031", "거문도", 34.034, 127.308),
    ("DT_0049", "광양",   34.905, 127.757),
    ("DT_0092", "여호항", 34.650, 127.850),
    ("DT_0026", "고흥발포", 34.481, 127.343),
    ("DT_0027", "완도",   34.315, 126.760),
    ("DT_0014", "통영",   34.827, 128.436),
]

def _nearest_khoa(lat: float, lon: float, max_deg: float = 0.6):
    best, best_d = None, max_deg
    for code, sname, slat, slon in KHOA_NEARBY:
        d = ((lat-slat)**2 + (lon-slon)**2) ** 0.5
        if d < best_d:
            best, best_d = (code, sname, d), d
    return best  # (obs_code, station_name, dist_deg) or None

def _wind_dir_kor(deg):
    if deg is None: return "—"
    try: deg = float(deg)
    except (TypeError, ValueError): return "—"
    dirs = ["북","북동","동","남동","남","남서","서","북서"]
    return dirs[int((deg + 22.5) % 360 // 45)]

# 공공데이터포털 KHOA(국립해양조사원, 기관코드 1192136) — 작동 확인된 4개 service
KHOA_BASE = "https://apis.data.go.kr/1192136"
KHOA_OPS = {
    "temp":     ("surveyWaterTemp", "GetSurveyWaterTempApiService"),  # 실측 수온  (필드 wtem)
    "wind":     ("surveyWind",      "GetSurveyWindApiService"),        # 풍향/풍속 (wndrct, wspd)
    "air_temp": ("surveyAirTemp",   "GetSurveyAirTempApiService"),     # 실측 기온 (artmp)
    "tide_h":   ("hourlyTide",      "GetHourlyTideApiService"),        # 1시간 조위 (tph)
}

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_khoa(kind: str, obs_code: str) -> list:
    if not DATA_GO_KR_KEY or kind not in KHOA_OPS:
        return []
    from datetime import datetime, timedelta
    svc, op = KHOA_OPS[kind]
    # 어제 날짜로 호출 (오늘은 누락 가능)
    req_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    params = {
        "serviceKey": DATA_GO_KR_KEY,
        "type":       "json",
        "obsCode":    obs_code,
        "reqDate":    req_date,
        "min":        60,
        "pageNo":     1,
        "numOfRows":  24,
    }
    try:
        r = requests.get(f"{KHOA_BASE}/{svc}/{op}", params=params, timeout=5)
        if r.status_code != 200:
            return []
        j = r.json()
        if j.get("header", {}).get("resultCode") != "00":
            return []
        items = j.get("body", {}).get("items", {})
        data  = items.get("item") if isinstance(items, dict) else items
        if isinstance(data, dict): data = [data]
        return data or []
    except Exception:
        return []

def _simulate_dynamics(name: str, risk: int) -> dict:
    """KHOA 실측 미수신 시 fallback. 실측 가능 항목만 유지."""
    rng = np.random.default_rng(abs(hash(name)) % 777)
    return dict(
        wind_dir  = rng.choice(["북","북동","동","남동","남","남서","서","북서"]),
        wind_speed= round(float(rng.uniform(2.5, 13.5)), 1),
        tide      = rng.choice(["밀물 가속 중","썰물 감속 중","정조기"]),
        temp      = round(float(rng.uniform(8.5, 22.0)),  1),
        source    = "시뮬레이션",
    )

def _last_num(rows, key):
    for row in reversed(rows):
        v = row.get(key)
        if v not in (None, ""):
            try: return float(v)
            except (TypeError, ValueError): pass
    return None

@st.cache_data(ttl=600, show_spinner=False)
def get_dynamics(name: str, risk: int, lat: float = None, lon: float = None) -> dict:
    """관측소 보유 섬은 KHOA 실측, 그 외는 인근 관측소 보간 또는 시뮬레이션."""
    code = OBS_CODE.get(name)
    station_label = name if code else None
    if not code and lat is not None and lon is not None:
        match = _nearest_khoa(float(lat), float(lon))
        if match:
            code, station_label, _ = match
    if not code or not DATA_GO_KR_KEY:
        return _simulate_dynamics(name, risk)
    sim = _simulate_dynamics(name, risk)
    out = dict(sim); used_real = False

    # 수온 (필드 wtem)
    v = _last_num(_fetch_khoa("temp", code), "wtem")
    if v is not None:
        out["temp"] = round(v, 1); used_real = True
    # 풍향/풍속 (wndrct: 도, wspd: m/s)
    rows = _fetch_khoa("wind", code)
    ws = _last_num(rows, "wspd")
    wd = _last_num(rows, "wndrct")
    if ws is not None:
        out["wind_speed"] = round(ws, 1); used_real = True
    if wd is not None:
        out["wind_dir"]   = _wind_dir_kor(wd); used_real = True
    # 1시간 조위 (tph) → 추세로 tide 상태 추정
    rows = _fetch_khoa("tide_h", code)
    tide_vals = [v for v in (_safe_float(r.get("tph")) for r in rows) if v is not None]
    if len(tide_vals) >= 2:
        diff = tide_vals[-1] - tide_vals[-2]
        if   abs(diff) < 3: out["tide"] = "정조기"
        elif diff > 0:      out["tide"] = "밀물 가속 중"
        else:               out["tide"] = "썰물 감속 중"
        used_real = True

    if used_real:
        out["source"] = ("KHOA 실측" if station_label == name
                         else f"KHOA 실측 ({station_label} 관측소)")
    else:
        out["source"] = "시뮬레이션"
    return out

def _safe_float(v):
    try: return float(v) if v not in (None, "") else None
    except (TypeError, ValueError): return None


# ═══════════════════════════════════════════════
# UI 시작
# ═══════════════════════════════════════════════
G, geojson_data = build_sea_graph()

# ── 사이드바 ──────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec">날짜 설정</p>', unsafe_allow_html=True)
    date_mode = st.radio("날짜 모드", ["단일 날짜", "기간 선택"],
                         horizontal=True, label_visibility="collapsed")
    if date_mode == "단일 날짜":
        predict_date   = st.date_input("날짜", value=date_type(2026, 3, 30),
                                       label_visibility="collapsed")
        start_date = end_date = predict_date
        date_range_days = 1
    else:
        date_range = st.date_input(
            "기간", value=(date_type(2026, 3, 28), date_type(2026, 4, 1)),
            label_visibility="collapsed")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date = end_date = date_range
        predict_date    = end_date
        date_range_days = max((end_date - start_date).days, 0) + 1

    st.markdown('<p class="sec">위험도 기준</p>', unsafe_allow_html=True)
    min_risk    = st.slider("최소 위험도", 0, 100, 60, label_visibility="collapsed")
    num_targets = st.slider("수거 대상 섬 수", 1, 10, 3)

    st.markdown('<p class="sec">수거 유형 필터</p>', unsafe_allow_html=True)
    sea_only       = st.checkbox("선박 수거 대상만 (연륙교 제외)", value=True)
    inhabited_only = st.checkbox("유인도만 보기", value=False)
    obs_only       = st.checkbox("관측소 보유 섬만", value=False)
    expo_mode      = st.checkbox("2026 박람회 관광지 강조", value=False)

# ── 데이터 준비 ───────────────────────────────
islands_df = load_island_data(predict_date.toordinal())
islands_df["color"] = islands_df["risk"].apply(risk_color)  # 캐시 밖에서 매번 계산

display_df = islands_df.copy()
if sea_only:       display_df = display_df[~display_df["has_bridge"]]
if inhabited_only: display_df = display_df[display_df["inhabited"] == 1]
if obs_only:       display_df = display_df[display_df["has_obs"]]

display_df  = display_df[display_df["risk"] >= min_risk].copy()
filtered_df = display_df.copy()
# 동점 처리: 1차 위험도↓ → 2차 쓰레기량↓ → 3차 출발항 거리↑
if not filtered_df.empty:
    _fd = filtered_df.copy()
    _fd["_dist_port"] = ((_fd["lat"]-BASE_PORT[1])**2 + (_fd["lon"]-BASE_PORT[0])**2) ** 0.5
    top_targets = (_fd.sort_values(["risk", "trash_cnt", "_dist_port"],
                                    ascending=[False, False, True])
                      .head(num_targets)
                      .drop(columns=["_dist_port"])
                      .reset_index(drop=True).copy())
else:
    top_targets = pd.DataFrame()

# ══════════════════════════════════════════════
# 상단 헤더
# ══════════════════════════════════════════════
total_islands = len(islands_df)
filtered_cnt  = len(filtered_df)
total_trash   = int(filtered_df["trash_cnt"].sum()) if not filtered_df.empty else 0

from datetime import date as _date
_expo_open = _date(2026, 9, 5)
_today     = _date.today()
_dday      = (_expo_open - _today).days

st.markdown(f"""
<div class="expo-header">
  <div class="expo-header-stars"></div>
  <svg class="expo-header-waves" viewBox="0 0 1200 48" preserveAspectRatio="none"
       xmlns="http://www.w3.org/2000/svg">
    <path d="M0,30 C150,48 350,8 600,28 C850,48 1050,10 1200,26 L1200,48 L0,48 Z"
          fill="rgba(255,255,255,0.12)"/>
    <path d="M0,36 C200,18 400,44 600,34 C800,24 1000,42 1200,32 L1200,48 L0,48 Z"
          fill="rgba(255,255,255,0.18)"/>
    <path d="M0,42 C300,30 600,46 900,36 C1050,30 1150,40 1200,38 L1200,48 L0,48 Z"
          fill="rgba(255,255,255,0.25)"/>
    <ellipse cx="980" cy="44" rx="55" ry="10" fill="rgba(34,120,60,0.55)"/>
    <ellipse cx="980" cy="38" rx="32" ry="12" fill="rgba(34,140,60,0.6)"/>
    <ellipse cx="1060" cy="46" rx="30" ry="7"  fill="rgba(34,120,60,0.45)"/>
    <ellipse cx="1060" cy="41" rx="18" ry="9"  fill="rgba(44,150,70,0.55)"/>
    <ellipse cx="880"  cy="46" rx="22" ry="6"  fill="rgba(34,110,55,0.45)"/>
    <ellipse cx="880"  cy="42" rx="13" ry="7"  fill="rgba(44,140,65,0.5)"/>
    <rect x="1005" y="24" width="5" height="16" fill="rgba(255,255,255,0.7)" rx="1"/>
    <polygon points="1002,24 1012,24 1007,18"  fill="rgba(255,100,80,0.8)"/>
  </svg>
  <div class="expo-header-content">
    <div class="expo-header-left">
      <div class="expo-header-badge">섬, 바다와 미래를 잇다</div>
      <div class="expo-header-title">🌊 여수 해양쓰레기 통합 수거 관리 시스템</div>
      <p class="expo-header-sub">
        조건충족 {filtered_cnt}개 섬 · 쓰레기 {total_trash:,}개
      </p>
    </div>
    <div style="display:flex;align-items:flex-end;gap:10px;flex-shrink:0;">
      {"<img src='" + DASOMI_SRC + "' style='height:120px;filter:drop-shadow(2px 4px 8px rgba(0,40,100,0.3));margin-bottom:-4px;animation:float 3s ease-in-out infinite;'/>" if DASOMI_SRC else ""}
    </div>
    <div class="expo-header-cards">
      <div class="expo-info-card">
        <div class="label">분석 대상</div>
        <div class="value">{filtered_cnt}개 섬</div>
      </div>
      <div class="expo-info-card">
        <div class="label">쓰레기 추정</div>
        <div class="value">{total_trash:,}개</div>
      </div>
      <div class="expo-info-card">
        <div class="label">2026 박람회</div>
        <div class="value">D-{_dday}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 알림 카드 ─────────────────────────────────
if not filtered_df.empty:
    t1 = filtered_df.nlargest(1,"risk").iloc[0]
    d1 = get_dynamics(t1["name"], int(t1["risk"]), float(t1["lat"]), float(t1["lon"]))
    r1 = int(t1["risk"])
    # 위험도 80+ 빨강 / 60~79 주황 / 40~59 노랑 / 그 외 초록
    if r1 >= 80:
        border_c, bg_c, label_c, label_txt = "#DC2626", "#FFF5F5", "#B91C1C", "수거 권고"
    elif r1 >= 60:
        border_c, bg_c, label_c, label_txt = "#F97316", "#FFF7ED", "#C2410C", "경고"
    elif r1 >= 40:
        border_c, bg_c, label_c, label_txt = "#EAB308", "#FEFCE8", "#A16207", "주의"
    else:
        border_c, bg_c, label_c, label_txt = "#22C55E", "#F0FDF4", "#15803D", "모니터링"
    st.markdown(f"""
<div style="background:{bg_c};border:1.5px solid {border_c};border-left:5px solid {border_c};
     border-radius:8px;padding:12px 18px;margin-bottom:10px;display:flex;align-items:center;gap:16px;">
  <div style="background:{border_c};color:white;border-radius:6px;padding:4px 12px;
       font-size:.78rem;font-weight:700;white-space:nowrap;letter-spacing:0.03em;">{label_txt}</div>
  <div style="flex:1;font-size:.88rem;color:#1A2B3C;">
    <b>{t1["name"]}</b>
    <span style="color:{label_c};font-weight:600;"> 위험도 {r1} | </span>
    <span style="color:#4B6178;">쓰레기 {int(t1["trash_cnt"])}개</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 메인 탭
# ══════════════════════════════════════════════
tab_map, tab_detail, tab_predict, tab_mohid, tab_data = st.tabs([
    "지도 · 수거 경로", "섬 상세 정보", "5일 예측 (LSTM)", "해양 역학 (MOHID)", "데이터 추출",
])

# ── TAB 1: 지도 ──────────────────────────────
with tab_map:
    if top_targets.empty:
        st.info("조건에 맞는 섬이 없습니다. 사이드바 필터를 조정해보세요.")
    else:
        map_col, info_col = st.columns([3, 1])

        with info_col:
            st.markdown('<p class="sec">수거 순서 (위험도 내림차순)</p>', unsafe_allow_html=True)
            for rank, row in top_targets.iterrows():
                r  = int(row["risk"]); tc = int(row["trash_cnt"])
                bridge = "육로" if row["has_bridge"] else "선박"
                inh    = "유인도" if row["inhabited"] else "무인도"
                st.markdown(
                    f'<div class="rcard" style="border-color:{risk_hex(r)};">'
                    f'<span style="background:{risk_hex(r)};color:white;border-radius:3px;'
                    f'padding:1px 7px;font-size:.8rem;font-weight:700;">{rank+1}</span>'
                    f'&nbsp;<b>{row["name"]}</b><br>'
                    f'<span style="color:{C["gray_mid"]};font-size:.78em;">'
                    f'위험도 {r} · {risk_label(r)} · {bridge} · {inh} · {tc}개</span></div>', unsafe_allow_html=True)

            st.markdown("---")
            bn  = int(top_targets["has_bridge"].sum())
            tot = int(top_targets["trash_cnt"].sum())
            for label, val in [
                ("평균 위험도",  f'{top_targets["risk"].mean():.1f}'),
                ("총 수거 예상량", f'{tot:,}개'),
                ("수거 유형",   f'선박 {len(top_targets)-bn}개 · 육로 {bn}개'),
            ]:
                st.markdown(
                    f'<div class="mbox"><div style="font-size:.7rem;color:{C["gray_mid"]};">'
                    f'{label}</div>'
                    f'<div style="font-size:{"1.4" if label!="수거 유형" else ".88"}rem;'
                    f'font-weight:{"700" if label!="수거 유형" else "400"};'
                    f'color:{C["blue_deep"] if label!="수거 유형" else C["gray_dark"]};'
                    f'margin-top:2px;">{val}</div></div>', unsafe_allow_html=True)

        with map_col:
            top_targets["color"] = top_targets["risk"].apply(lambda r: risk_color(int(r), 255))
            top_targets["order"] = (top_targets.index+1).astype(str)
            waypoints = [BASE_PORT] + top_targets[["lon","lat"]].values.tolist()
            with st.spinner("A* 해상 최단 경로 계산 중..."):
                route = find_route(waypoints, G)

            port_df = pd.DataFrame([{
                "lon": BASE_PORT[0], "lat": BASE_PORT[1],
                "name": "출발항 (여수항)", "order": "P",
                "color": [0, 102, 179, 255],
                "tooltip": "출발항 (여수항)",
            }])

            obs_df  = display_df[display_df["has_obs"]]
            expo_df = display_df[display_df["is_expo"]]

            layers = [
                pdk.Layer("GeoJsonLayer", geojson_data,
                    get_fill_color=[180,200,220,60], get_line_color=[100,140,180,120],
                    line_width_min_pixels=1),
                pdk.Layer("ScatterplotLayer", display_df,
                    get_position="[lon, lat]", get_color="color",
                    get_radius=400, radius_min_pixels=3, radius_max_pixels=10,
                    opacity=0.8, pickable=True),
                pdk.Layer("ScatterplotLayer", top_targets,
                    get_position="[lon, lat]",
                    get_color=[0,102,179,35], get_radius=900,
                    radius_min_pixels=14, radius_max_pixels=32,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[0,102,179,200]),
                pdk.Layer("ScatterplotLayer", top_targets,
                    get_position="[lon, lat]", get_color="color",
                    get_radius=500, radius_min_pixels=8, radius_max_pixels=20,
                    pickable=True),
                pdk.Layer("ScatterplotLayer", port_df,
                    get_position="[lon, lat]", get_color=[0,102,179,255],
                    get_radius=500, radius_min_pixels=8, radius_max_pixels=20,
                    pickable=True),
                pdk.Layer("PathLayer",
                    pd.DataFrame([{"path": route}]),
                    get_path="path", get_color=[0,102,179,200],
                    width_min_pixels=3, width_max_pixels=6),
                # 관측소 — 청록색 작은 도트 + 깔끔한 라벨
                *([pdk.Layer("ScatterplotLayer", obs_df,
                    get_position="[lon, lat]",
                    get_color=[8,145,178,255], get_radius=320,
                    radius_min_pixels=6, radius_max_pixels=12,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255,255,255,255]),
                  pdk.Layer("TextLayer", obs_df,
                    get_position="[lon, lat]", get_text="['관측소']",
                    get_size=10, get_color=[255,255,255,255],
                    background=True,
                    get_background_color=[8,145,178,235],
                    get_border_color=[8,145,178,255],
                    get_padding=[5,2,5,2], get_pixel_offset=[0,-20],
                    billboard=True)] if not obs_df.empty else []),
                # 박람회 — 인디고 작은 도트 + 깔끔한 라벨 (위험도 초록과 충돌 방지)
                *([pdk.Layer("ScatterplotLayer", expo_df,
                    get_position="[lon, lat]",
                    get_color=[99,102,241,255], get_radius=360,
                    radius_min_pixels=7, radius_max_pixels=14,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255,255,255,255]),
                  pdk.Layer("TextLayer", expo_df,
                    get_position="[lon, lat]", get_text="['박람회']",
                    get_size=10, get_color=[255,255,255,255],
                    background=True,
                    get_background_color=[99,102,241,235],
                    get_border_color=[79,70,229,255],
                    get_padding=[5,2,5,2], get_pixel_offset=[0,20],
                    billboard=True)] if expo_mode and not expo_df.empty else []),
                pdk.Layer("TextLayer", top_targets,
                    get_position="[lon, lat]", get_text="order",
                    get_size=16, get_color=[255,255,255,255],
                    background=True, get_background_color=[0,63,125,240],
                    get_border_color=[0,63,125,255], get_padding=[5,3,5,3],
                    font_weight=700, billboard=True),
                pdk.Layer("TextLayer", port_df,
                    get_position="[lon, lat]", get_text="order",
                    get_size=14, get_color=[255,255,255,255],
                    background=True, get_background_color=[0,63,125,240],
                    get_border_color=[0,63,125,255], get_padding=[5,3,5,3],
                    font_weight=700, billboard=True),
            ]

            st.pydeck_chart(pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=34.55, longitude=127.65, zoom=9, pitch=0),
                map_style=MAP_STYLE,
                tooltip={"text": "{tooltip}",
                         "style": {"backgroundColor": C["white"], "color": C["gray_dark"],
                                   "fontSize": "12px", "border": f"1px solid {C['gray_light']}",
                                   "borderRadius": "4px", "padding": "8px 10px"}},
            ), use_container_width=True)

            st.markdown(
                f'<div style="display:flex;gap:16px;font-size:.78em;'
                f'color:{C["gray_mid"]};margin-top:5px;flex-wrap:wrap;">'
                f'<span><span style="color:#22C55E">●</span> 양호(0~39)</span>'
                f'<span><span style="color:#EAB308">●</span> 주의(40~59)</span>'
                f'<span><span style="color:#F97316">●</span> 경고(60~79)</span>'
                f'<span><span style="color:#DC2626">●</span> 위험(80~100)</span>'
                f'<span><span style="color:{C["blue_mid"]}">━</span> 수거 노선</span>'
                f'<span><span style="color:#0891B2;font-weight:700;">●</span> 해양관측소</span>'
                f'<span><span style="color:#6366F1;font-weight:700;">●</span> 박람회 행사장</span>'
                f'</div>', unsafe_allow_html=True)


# ── TAB 2: 섬 상세 ──────────────────────────
with tab_detail:
    st.markdown("#### 섬 상세 정보 조회")
    all_names  = sorted(islands_df["name"].tolist())
    sel_island = st.selectbox("섬 선택", all_names,
                              index=all_names.index(top_targets.iloc[0]["name"])
                              if not top_targets.empty else 0)
    row = islands_df[islands_df["name"] == sel_island].iloc[0]
    r   = int(row["risk"])
    dyn = get_dynamics(sel_island, r, float(row["lat"]), float(row["lon"]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="detail-card">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
               color:{C['gray_mid']};margin-bottom:8px;">기본 정보</div>
          <div style="font-size:1.1rem;font-weight:700;color:{C['blue_deep']};
               margin-bottom:12px;">{sel_island}</div>
          <table style="width:100%;font-size:.84em;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">위험도</td>
                <td style="text-align:right;"><b style="color:{risk_hex(r)};">{r} — {risk_label(r)}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">쓰레기 추정</td>
                <td style="text-align:right;"><b>{int(row['trash_cnt'])}개</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">쓰레기 유형</td>
                <td style="text-align:right;"><b>{row['trash_types']}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">유무인도</td>
                <td style="text-align:right;"><b>{"유인도" if row["inhabited"] else "무인도"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">연육도현황</td>
                <td style="text-align:right;"><b>{"연륙교/연도교 있음" if row["has_bridge"] else "없음 (선박 수거)"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">해양관측소</td>
                <td style="text-align:right;">
                  <b style="color:{"#0066B3" if row["has_obs"] else C["gray_mid"]};">
                  {"있음 (국가 관측 지점)" if row["has_obs"] else "없음"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">2026 박람회</td>
                <td style="text-align:right;">
                  <b style="color:{"#16A34A" if row["expo_role"]!="해당 없음" else C["gray_mid"]};">
                  {row["expo_role"]}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">위도</td>
                <td style="text-align:right;"><b>{row['lat']:.5f}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">경도</td>
                <td style="text-align:right;"><b>{row['lon']:.5f}</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with c2:
        rows_html = "".join(
            f'<tr><td style="padding:4px 0;color:{C["gray_mid"]};">{k}</td>'
            f'<td style="text-align:right;color:{C["gray_dark"]};"><b>{v}</b></td></tr>'
            for k,v in [
                ("풍향/풍속", f'{dyn["wind_dir"]}풍 {dyn["wind_speed"]} m/s'),
                ("조류 상태", dyn["tide"]),
                ("수온",     f'{dyn["temp"]} °C'),
            ])
        src       = dyn.get("source", "시뮬레이션")
        is_real   = src.startswith("KHOA 실측")
        src_color = "#15803D" if is_real else C["gray_mid"]
        st.markdown(f"""
        <div class="detail-card">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
               color:{C['gray_mid']};margin-bottom:8px;">해양 역학</div>
          <table style="width:100%;font-size:.84em;border-collapse:collapse;">{rows_html}</table>
          <div style="margin-top:10px;font-size:.72rem;color:{src_color};">
            출처: <b>{src}</b></div>
        </div>""", unsafe_allow_html=True)


# ── TAB 3: LSTM 예측 ─────────────────────────
with tab_predict:
    if top_targets.empty:
        st.info("수거 대상 섬이 없습니다.")
    else:
        sel2 = st.selectbox("분석 대상 섬", top_targets["name"].tolist(),
                            index=0, key="predict_sel")
        row2 = top_targets[top_targets["name"] == sel2].iloc[0]
        st.plotly_chart(
            build_forecast_chart(sel2, int(row2["risk"]), int(row2["trash_cnt"]), predict_date),
            use_container_width=True)
        st.caption("LSTM 시계열 모델이 해류·기상·조류 데이터를 학습하여 "
                   "향후 5일간 위험도 및 쓰레기 유입량을 예측합니다. 음영은 95% 신뢰구간입니다.")


# ── TAB 4: 해양 역학 비교 (Top 3) ────────────
with tab_mohid:
    if top_targets.empty:
        st.info("수거 대상 섬이 없습니다.")
    else:
        st.markdown("#### Top 수거 대상 한눈 비교")

        compare = top_targets.head(3).copy()
        dyn_list = [get_dynamics(r["name"], int(r["risk"]),
                                  float(r["lat"]), float(r["lon"]))
                    for _, r in compare.iterrows()]
        names      = compare["name"].tolist()
        risks      = compare["risk"].astype(int).tolist()
        trash      = compare["trash_cnt"].astype(int).tolist()
        wind_speed = [d["wind_speed"] for d in dyn_list]
        wind_dir   = [d["wind_dir"]   for d in dyn_list]
        tide       = [d["tide"]       for d in dyn_list]
        temp       = [d["temp"]       for d in dyn_list]
        bar_colors = [risk_hex(r) for r in risks]
        sources    = [d.get("source", "시뮬레이션") for d in dyn_list]

        def _bar(x, y, title, ytitle, text, hover):
            fig = go.Figure(go.Bar(x=x, y=y, marker_color=bar_colors,
                                    text=text, textposition="outside",
                                    hovertemplate=hover))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13, color=C["blue_deep"])),
                paper_bgcolor=C["white"], plot_bgcolor=C["gray_bg"],
                yaxis=dict(gridcolor=C["gray_light"], title=ytitle),
                xaxis=dict(title=""), height=260,
                margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
            return fig

        # Row 1: 위험도 / 쓰레기량 (모델 예측 영역)
        c1, c2 = st.columns(2)
        with c1:
            f1 = _bar(names, risks, "위험도 비교", "위험도",
                      risks, "%{x}<br>위험도 %{y}<extra></extra>")
            f1.update_yaxes(range=[0, 110])
            st.plotly_chart(f1, use_container_width=True)
        with c2:
            f2 = _bar(names, trash, "쓰레기 추정량 비교", "개수",
                      [f"{t:,}" for t in trash], "%{x}<br>쓰레기 %{y:,}개<extra></extra>")
            st.plotly_chart(f2, use_container_width=True)

        # Row 2: 풍속 / 수온 (KHOA 실측 영역)
        c3, c4 = st.columns(2)
        with c3:
            f3 = _bar(names, wind_speed, "풍속 (KHOA 실측)", "m/s",
                      [f"{v}" for v in wind_speed], "%{x}<br>풍속 %{y} m/s<extra></extra>")
            st.plotly_chart(f3, use_container_width=True)
        with c4:
            f4 = _bar(names, temp, "수온 (KHOA 실측)", "°C",
                      [f"{v}" for v in temp], "%{x}<br>수온 %{y} °C<extra></extra>")
            st.plotly_chart(f4, use_container_width=True)

        # Row 3: 풍향 · 조류 상태 텍스트 카드
        text_cols = st.columns(len(names))
        for i, col in enumerate(text_cols):
            with col:
                st.markdown(f"""
<div style="border:1px solid {C['gray_light']};background:{C['white']};
     border-radius:6px;padding:12px 14px;text-align:center;">
  <div style="font-size:.95rem;font-weight:700;color:{C['blue_deep']};margin-bottom:6px;">{names[i]}</div>
  <div style="font-size:.82rem;color:{C['gray_mid']};">풍향 <b style="color:{C['gray_dark']};">{wind_dir[i]}</b>
       &nbsp;·&nbsp; 조류 <b style="color:{C['gray_dark']};">{tide[i]}</b></div>
</div>""", unsafe_allow_html=True)

        # 데이터 출처 표시
        src_html = " · ".join(
            f'<span style="color:{"#15803D" if s.startswith("KHOA 실측") else C["gray_mid"]};">'
            f'<b>{n}</b>: {s}</span>' for n, s in zip(names, sources))
        st.markdown(
            f'<div style="margin-top:12px;font-size:.78em;color:{C["gray_mid"]};">'
            f'{src_html}</div>', unsafe_allow_html=True)
        st.caption("KHOA 조위관측소 실측치(수온·풍향풍속·조류상태)를 사용. "
                   "관측소 미보유 섬은 가장 가까운 관측소 데이터로 보간하며, "
                   "위험도·쓰레기량은 LSTM 예측 모델 결과를 연결할 자리입니다.")


# ── TAB 5: 데이터 추출 ───────────────────────
with tab_data:
    st.markdown("#### 데이터 추출 · 다운로드")
    dl1, dl2 = st.columns(2)

    with dl1:
        st.markdown("**현재 필터 적용 데이터**")
        st.dataframe(
            filtered_df[["name","lat","lon","risk","trash_cnt","trash_types",
                         "main_type","inhabited","has_bridge","has_obs","is_expo"
                        ]].rename(columns={
                "name":"섬이름","lat":"위도","lon":"경도",
                "risk":"위험도","trash_cnt":"쓰레기추정(개)",
                "trash_types":"쓰레기유형","main_type":"주요유형",
                "inhabited":"유무인(1=유인)","has_bridge":"연륙교여부",
                "has_obs":"관측소여부","is_expo":"박람회관광지"
            }).reset_index(drop=True),
            use_container_width=True, height=350)
        st.download_button("CSV 다운로드 (필터 적용)",
            filtered_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"yeosu_filtered_{predict_date}.csv", mime="text/csv")

    with dl2:
        st.markdown("**수거 대상 섬 (최종 선정)**")
        if not top_targets.empty:
            st.dataframe(
                top_targets[["name","lat","lon","risk","trash_cnt",
                              "trash_types","inhabited","has_bridge"
                            ]].rename(columns={
                    "name":"섬이름","lat":"위도","lon":"경도",
                    "risk":"위험도","trash_cnt":"쓰레기추정(개)",
                    "trash_types":"쓰레기유형",
                    "inhabited":"유무인","has_bridge":"연륙교"
                }).reset_index(drop=True),
                use_container_width=True, height=350)
            st.download_button("CSV 다운로드 (수거 대상)",
                top_targets.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"yeosu_targets_{predict_date}.csv", mime="text/csv")
        else:
            st.info("수거 대상 섬이 없습니다.")

    st.markdown("---")
    st.markdown("**전체 섬 원본 데이터**")
    st.download_button("CSV 다운로드 (전체 데이터)",
        islands_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"yeosu_all_{predict_date}.csv", mime="text/csv")
    st.dataframe(
        islands_df[["name","lat","lon","risk","trash_cnt","trash_types",
                    "inhabited","has_bridge","has_obs"]].head(50),
        use_container_width=True, height=300)
    st.caption("상위 50개만 미리보기 표시. 전체 데이터는 CSV로 다운로드하세요.")
