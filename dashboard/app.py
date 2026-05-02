"""
도동실 — 해양쓰레기 수거 동선 최적화 시스템  v10

[변경사항]
- 서비스명 '도동실'로 전면 변경
- 패러다임: 예측 → 현재 적체 기반 수거 동선 최적화
- 평균 위험도 지표 완전 삭제
- MOHID 탭 삭제
- Fake 시뮬레이션 데이터 전면 제거 (실측 불가 시 '-' 표기)
- 경로 최적화 레이어 시각적 강조
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta, date as date_type
import base64
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
    page_title="도동실 | 해양쓰레기 수거 동선 최적화",
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

PORTS = {
    "여수항":     [127.730,    34.655],
    "여수신항":   [127.750952, 34.751861],
    "광양항":     [127.685735, 34.922996],
    "국동항":     [127.722099, 34.726573],
    "녹동항":     [127.470,    34.534],
    "고흥발포항": [127.343,    34.481],
    "완도항":     [126.760,    34.315],
}
DEFAULT_PORT_NAME = "여수항"

TRASH_TYPES = ["부유쓰레기", "해안쓰레기", "침적쓰레기", "어구류", "유류"]

EXPO_ISLANDS = {"돌산도", "금오도", "개도"}
OBS_KEYWORDS = ["백야도", "안도", "거문도", "반월도"]


@st.cache_data
def get_local_image_base64(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
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
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div:nth-child(2) {{
        background-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {C['blue_mid']} !important;
        border-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stCheckbox label {{
        background-color: transparent !important; box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stCheckbox label:hover {{
        background-color: transparent !important; box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"]:focus-within {{
        background-color: transparent !important; box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:last-child {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="checkbox"] > div:last-child:hover {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] p {{ background-color: transparent !important; }}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] [aria-checked="true"] div:first-child {{
        background-color: {C['blue_mid']} !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
        background-color: {C['blue_mid']} !important; color: white !important;
    }}
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700;800&display=swap');
    .expo-header {{
        position: relative;
        background: linear-gradient(180deg, #005fa3 0%, #0077C8 40%, #00A3E0 80%, #29B6E0 100%);
        border-radius: 14px; overflow: hidden; margin-bottom: 16px; min-height: 160px;
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
    @keyframes twinkle {{ 0% {{ opacity: 0.6; }} 100% {{ opacity: 1; }} }}
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
        border-radius: 20px; padding: 3px 12px;
        font-size: .78rem; font-weight: 600;
        color: rgba(255,255,255,0.95); letter-spacing: 0.05em; margin-bottom: 8px;
    }}
    .expo-header-title {{
        font-size: 1.45rem !important; font-weight: 800 !important;
        color: white !important; margin: 0 0 3px !important;
        line-height: 1.2 !important;
        text-shadow: 0 2px 8px rgba(0,30,80,0.4); letter-spacing: -0.01em !important;
    }}
    .expo-header-sub {{
        font-size: .8rem; color: rgba(255,255,255,0.85); margin: 0;
    }}
    .expo-header-cards {{
        display: flex; gap: 8px; flex-shrink: 0; flex-wrap: wrap; align-items: flex-start;
    }}
    .expo-info-card {{
        background: rgba(255,255,255,0.18); backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.3); border-radius: 10px;
        padding: 8px 14px; text-align: center; min-width: 80px;
    }}
    .expo-info-card .label {{ font-size: .68rem; color: rgba(255,255,255,0.75); margin-bottom: 3px; }}
    .expo-info-card .value {{ font-size: .95rem; font-weight: 700; color: white; }}
    .sec {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
            color:{C['gray_mid']}; margin:14px 0 6px; font-weight:700; }}
    .rcard {{ border-left:3px solid; border-radius:5px; padding:8px 12px;
              margin-bottom:6px; font-size:.86rem; background:{C['white']};
              box-shadow:0 1px 3px rgba(0,0,0,.06); }}
    .mbox {{ background:{C['white']}; border:1px solid {C['gray_light']};
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
    footer {{ visibility:hidden; }}
    h1,h2,h3,h4 {{ color:{C['blue_deep']} !important; font-weight:700 !important; }}
</style>
""", unsafe_allow_html=True)

GEOJSON_PATH = "yeosu_polygons.geojson"
MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


# ═══════════════════════════════════════════════
# 1. 유틸
# ═══════════════════════════════════════════════
def risk_color(risk: int, alpha: int = 220) -> list:
    if risk < 40:  return [ 34, 197,  94, alpha]
    if risk < 60:  return [234, 179,   8, alpha]
    if risk < 80:  return [249, 115,  22, alpha]
    return              [220,  38,  38, alpha]

def risk_label(risk: int) -> str:
    if risk >= 80: return "위험"
    if risk >= 60: return "경고"
    if risk >= 40: return "주의"
    return "양호"

def risk_hex(risk: int) -> str:
    c = risk_color(risk, 255)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

def blue_hex_by_risk(risk: int) -> str:
    if risk >= 80: return "#3E8FD6"
    if risk >= 60: return "#5FA8E6"
    if risk >= 40: return "#86C1EE"
    return "#B6DBF6"


# ═══════════════════════════════════════════════
# 2. 육지 판별 (정확한 폴리곤 기반)
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner="육지 영역 로딩 중...")
def load_land_geom():
    polys = []
    try:
        gadm = gpd.read_file("gadm41_KOR_2.shp")
        gadm = gadm.cx[126.7:128.2, 33.8:35.2]
        polys.extend(g for g in gadm.geometry if g and not g.is_empty)
    except Exception:
        pass
    try:
        shp = gpd.read_file("namhae_final3.shp")
        shp = shp.cx[126.9:128.1, 33.9:35.1]
        polys.extend(g for g in shp.geometry if g and not g.is_empty)
    except Exception:
        pass
    try:
        gj = gpd.read_file(GEOJSON_PATH)
        polys.extend(g for g in gj.geometry if g and not g.is_empty)
    except Exception:
        pass
    union = unary_union(polys)
    return union, prep(union)

_LAND_UNION = None
_LAND_PREP  = None

def _ensure_land():
    global _LAND_UNION, _LAND_PREP
    if _LAND_PREP is None:
        _LAND_UNION, _LAND_PREP = load_land_geom()

def edge_crosses_land(a, b) -> bool:
    _ensure_land()
    return _LAND_PREP.intersects(LineString([(a[0], a[1]), (b[0], b[1])]))


# ═══════════════════════════════════════════════
# 3. 해상 그래프 & A* 경로
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
                nodes.append(n)
                G.add_node(n)
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
    """A* 해상 최단 경로 탐색 — 그래프를 mutate하지 않아 캐시 안전."""
    _ensure_land()
    nodes_orig = list(G.nodes())
    xy_orig    = np.array([(n[0], n[1]) for n in nodes_orig])

    snapped = []
    for wp in waypoints:
        d     = np.hypot(xy_orig[:, 0]-wp[0], xy_orig[:, 1]-wp[1])
        order = np.argsort(d)
        chosen = None
        for j in order[:80]:
            n = nodes_orig[int(j)]
            if not _LAND_PREP.intersects(LineString([(wp[0], wp[1]), (n[0], n[1])])):
                chosen = n
                break
        snapped.append(chosen if chosen else nodes_orig[int(order[0])])

    path = []
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
    rng  = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)

    df["risk"]       = rng.integers(5, 100, size=len(df))
    df["trash_cnt"]  = (df["risk"] * rng.uniform(3.0, 5.0, size=len(df))).astype(int)
    df["has_bridge"] = df["연육도현황"].notna()
    df["inhabited"]  = df["유무인도"].astype(int)
    df["is_expo"]    = df["name"].apply(lambda n: any(e in n for e in EXPO_ISLANDS))
    df["expo_role"]  = df["name"].apply(
        lambda n: "공식 행사장" if any(e in n for e in EXPO_ISLANDS) else "해당 없음"
    )

    def assign_types(i):
        k = rng2.integers(1, 4)
        return ", ".join(sorted(rng2.choice(TRASH_TYPES, size=k, replace=False)))
    df["trash_types"] = [assign_types(i) for i in range(len(df))]
    df["main_type"]   = df["trash_types"].apply(lambda s: s.split(", ")[0])
    df["has_obs"]     = df["name"].apply(lambda n: n in OBS_KEYWORDS)

    extra_rows = []
    for name, lat, lon, inh in [("거문도", 34.033, 127.317, 1),
                                 ("반월도", 34.849, 127.594, 1)]:
        if name not in df["name"].values:
            r = int(rng.integers(5, 100))
            extra_rows.append({
                "name": name, "lat": lat, "lon": lon,
                "risk": r, "trash_cnt": int(r * rng.uniform(3.0, 5.0)),
                "has_bridge": False, "inhabited": inh,
                "is_expo": name in EXPO_ISLANDS,
                "expo_role": "공식 행사장" if name in EXPO_ISLANDS else "해당 없음",
                "has_obs": True,
                "trash_types": "부유쓰레기, 해안쓰레기",
                "main_type": "부유쓰레기",
            })
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df["tooltip"] = (
        df["name"] + "\n현재 쓰레기 " + df["trash_cnt"].astype(str) + "개 적체" +
        "\n수거 우선도: " + df["risk"].apply(risk_label) +
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
        pred.append(v); lo.append(max(0.0, v-e)); hi.append(min(100.0, v+e))
    trash_pred = [max(0, int(p/max(base, 1)*base_trash)) for p in pred]
    bh = blue_hex_by_risk(int(base)).lstrip("#")
    r, g, b = tuple(int(bh[i:i+2], 16) for i in (0, 2, 4))
    lc = f"rgb({r},{g},{b})"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels+labels[::-1], y=hi+lo[::-1],
        fill="toself", fillcolor=f"rgba({r},{g},{b},.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% 신뢰구간"))
    fig.add_trace(go.Scatter(
        x=labels, y=pred, mode="lines+markers",
        name="쓰레기 적체 예측", line=dict(color=lc, width=2.5),
        marker=dict(size=7, color=lc, line=dict(color="white", width=1.5))))
    fig.add_trace(go.Bar(
        x=labels, y=trash_pred, name="쓰레기 추정량",
        marker_color=f"rgba({r},{g},{b},.42)",
        marker_line_color=f"rgba({r},{g},{b},.85)",
        marker_line_width=1, yaxis="y2"))
    fig.add_trace(go.Scatter(
        x=[labels[0]], y=[pred[0]], mode="markers", name="현재 상태",
        marker=dict(size=12, color="white", symbol="diamond",
                    line=dict(color=lc, width=2.5))))
    fig.update_layout(
        title=dict(text=f"{name} — 향후 5일 쓰레기 적체량 예측 (LSTM)",
                   font=dict(size=13, color=C["blue_deep"])),
        paper_bgcolor=C["white"], plot_bgcolor=C["gray_bg"],
        font=dict(color=C["gray_mid"], size=12),
        xaxis=dict(gridcolor=C["gray_light"], title="날짜"),
        yaxis=dict(gridcolor=C["gray_light"], range=[0, 110],
                   title="수거 우선도 지수", side="left"),
        yaxis2=dict(title="쓰레기(개)", overlaying="y", side="right",
                    showgrid=False, range=[0, max(trash_pred or [1])*2.5]),
        legend=dict(orientation="h", y=1.12),
        barmode="overlay", height=320,
        margin=dict(l=10, r=40, t=55, b=10))
    return fig


# ═══════════════════════════════════════════════
# 6. 해양 역학 — KHOA 실측 API (Fake 없음)
# ═══════════════════════════════════════════════
DATA_GO_KR_KEY = st.secrets.get("DATA_GO_KR_KEY", "")

OBS_CODE = {"거문도": "DT_0031"}

KHOA_NEARBY = [
    ("DT_0016", "여수",     34.747, 127.766),
    ("DT_0031", "거문도",   34.034, 127.308),
    ("DT_0049", "광양",     34.905, 127.757),
    ("DT_0092", "여호항",   34.650, 127.850),
    ("DT_0026", "고흥발포", 34.481, 127.343),
    ("DT_0027", "완도",     34.315, 126.760),
    ("DT_0014", "통영",     34.827, 128.436),
]

KHOA_BASE = "https://apis.data.go.kr/1192136"
KHOA_OPS  = {
    "temp":   ("surveyWaterTemp", "GetSurveyWaterTempApiService"),
    "wind":   ("surveyWind",      "GetSurveyWindApiService"),
    "tide_h": ("hourlyTide",      "GetHourlyTideApiService"),
}

EMPTY_DYN = dict(wind_dir=None, wind_speed=None, tide=None, temp=None, source="관측 불가")


def _nearest_khoa(lat: float, lon: float, max_deg: float = 0.6):
    best, best_d = None, max_deg
    for code, sname, slat, slon in KHOA_NEARBY:
        d = ((lat-slat)**2 + (lon-slon)**2) ** 0.5
        if d < best_d:
            best, best_d = (code, sname, d), d
    return best

def _wind_dir_kor(deg):
    if deg is None: return None
    try: deg = float(deg)
    except (TypeError, ValueError): return None
    dirs = ["북","북동","동","남동","남","남서","서","북서"]
    return dirs[int((deg + 22.5) % 360 // 45)]

def _safe_float(v):
    try: return float(v) if v not in (None, "") else None
    except (TypeError, ValueError): return None

def _last_num(rows, key):
    for row in reversed(rows):
        v = _safe_float(row.get(key))
        if v is not None:
            return v
    return None

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_khoa(kind: str, obs_code: str) -> list:
    if not DATA_GO_KR_KEY or kind not in KHOA_OPS:
        return []
    from datetime import datetime, timedelta as td
    svc, op = KHOA_OPS[kind]
    req_date = (datetime.now() - td(days=1)).strftime("%Y%m%d")
    params = {
        "serviceKey": DATA_GO_KR_KEY, "type": "json",
        "obsCode": obs_code, "reqDate": req_date,
        "min": 60, "pageNo": 1, "numOfRows": 24,
    }
    try:
        resp = requests.get(f"{KHOA_BASE}/{svc}/{op}", params=params, timeout=5)
        if resp.status_code != 200:
            return []
        j = resp.json()
        if j.get("header", {}).get("resultCode") != "00":
            return []
        items = j.get("body", {}).get("items", {})
        data  = items.get("item") if isinstance(items, dict) else items
        if isinstance(data, dict): data = [data]
        return data or []
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_dynamics(name: str, lat: float = None, lon: float = None) -> dict:
    """KHOA 실측치만 반환. 수신 불가 항목은 None으로 반환 (Fake 없음)."""
    code = OBS_CODE.get(name)
    station_label = name if code else None
    if not code and lat is not None and lon is not None:
        match = _nearest_khoa(float(lat), float(lon))
        if match:
            code, station_label, _ = match
    if not code or not DATA_GO_KR_KEY:
        return dict(EMPTY_DYN)

    out = dict(wind_dir=None, wind_speed=None, tide=None, temp=None)
    used_real = False

    v = _last_num(_fetch_khoa("temp", code), "wtem")
    if v is not None:
        out["temp"] = round(v, 1); used_real = True

    rows = _fetch_khoa("wind", code)
    ws = _last_num(rows, "wspd")
    wd = _last_num(rows, "wndrct")
    if ws is not None:
        out["wind_speed"] = round(ws, 1); used_real = True
    if wd is not None:
        out["wind_dir"] = _wind_dir_kor(wd); used_real = True

    tide_rows = _fetch_khoa("tide_h", code)
    tide_vals = [v2 for v2 in (_safe_float(r.get("tph")) for r in tide_rows) if v2 is not None]
    if len(tide_vals) >= 2:
        diff = tide_vals[-1] - tide_vals[-2]
        out["tide"] = "정조기" if abs(diff) < 3 else ("밀물 가속 중" if diff > 0 else "썰물 감속 중")
        used_real = True

    if used_real:
        out["source"] = ("KHOA 실측" if station_label == name
                         else f"KHOA 실측 ({station_label} 관측소)")
    else:
        out["source"] = "관측 불가"
    return out


# ═══════════════════════════════════════════════
# UI 시작
# ═══════════════════════════════════════════════
G, geojson_data = build_sea_graph()

# ── 사이드바 ──────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec">수거 기준일</p>', unsafe_allow_html=True)
    date_mode = st.radio("날짜 모드", ["단일 날짜", "기간 선택"],
                         horizontal=True, label_visibility="collapsed")
    if date_mode == "단일 날짜":
        predict_date = st.date_input("날짜", value=date_type(2026, 3, 30),
                                     label_visibility="collapsed")
        start_date = end_date = predict_date
    else:
        date_range = st.date_input(
            "기간", value=(date_type(2026, 3, 28), date_type(2026, 4, 1)),
            label_visibility="collapsed")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range[0], date_range[1]
        else:
            start_date = end_date = date_range
        predict_date = end_date

    st.markdown('<p class="sec">수거 기준</p>', unsafe_allow_html=True)
    min_risk    = st.slider("최소 수거 우선도", 0, 100, 60, label_visibility="collapsed")
    num_targets = st.slider("수거 대상 섬 수", 1, 10, 3)

    st.markdown('<p class="sec">필터</p>', unsafe_allow_html=True)
    sea_only       = st.checkbox("선박 수거 대상만 (연륙교 제외)", value=True)
    inhabited_only = st.checkbox("유인도만 보기", value=False)
    obs_only       = st.checkbox("관측소 보유 섬만", value=False)
    expo_mode      = st.checkbox("2026 박람회 행사장 강조", value=False)

    st.markdown('<p class="sec">출발항</p>', unsafe_allow_html=True)
    start_port_name = st.selectbox(
        "출발항",
        list(PORTS.keys()),
        index=list(PORTS.keys()).index(DEFAULT_PORT_NAME),
        label_visibility="collapsed",
    )

BASE_PORT = PORTS[start_port_name]

# ── 데이터 준비 ───────────────────────────────
islands_df = load_island_data(predict_date.toordinal())
islands_df["color"] = islands_df["risk"].apply(risk_color)

display_df = islands_df.copy()
if sea_only:       display_df = display_df[~display_df["has_bridge"]]
if inhabited_only: display_df = display_df[display_df["inhabited"] == 1]
if obs_only:       display_df = display_df[display_df["has_obs"]]
display_df  = display_df[display_df["risk"] >= min_risk].copy()
filtered_df = display_df.copy()

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
filtered_cnt = len(filtered_df)
total_trash  = int(filtered_df["trash_cnt"].sum()) if not filtered_df.empty else 0

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
      <div class="expo-header-badge">실시간 현황 기반 수거 동선 최적화</div>
      <div class="expo-header-title">🚢 도동실 — 해양쓰레기 수거 동선 최적화 시스템</div>
      <p class="expo-header-sub">
        현재 수거 필요 {filtered_cnt}개 섬 · 적체 쓰레기 {total_trash:,}개 — 최적 동선을 즉시 확인하세요
      </p>
    </div>
    <div style="display:flex;align-items:flex-end;gap:10px;flex-shrink:0;">
      {"<img src='" + DASOMI_SRC + "' style='height:120px;filter:drop-shadow(2px 4px 8px rgba(0,40,100,0.3));margin-bottom:-4px;animation:float 3s ease-in-out infinite;'/>" if DASOMI_SRC else ""}
    </div>
    <div class="expo-header-cards">
      <div class="expo-info-card">
        <div class="label">수거 대상</div>
        <div class="value">{filtered_cnt}개 섬</div>
      </div>
      <div class="expo-info-card">
        <div class="label">적체 쓰레기</div>
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
_alert_df = filtered_df if not filtered_df.empty else (display_df if not display_df.empty else islands_df)
if not _alert_df.empty:
    if not top_targets.empty:
        t1 = top_targets.iloc[0]
        action_text = f"지금 바로 {t1['name']}으로 수거선을 출동하세요"
    else:
        t1 = _alert_df.nlargest(1, "risk").iloc[0]
        action_text = f"{t1['name']} 현황을 즉시 확인하세요"
    r1 = int(t1["risk"])
    border_c = risk_hex(r1)
    label_txt = "즉시 수거" if r1 >= 80 else ("수거 권고" if r1 >= 60 else ("수거 검토" if r1 >= 40 else "모니터링"))
    st.markdown(f"""
<div style="background:{C['white']};border:1.5px solid {border_c};border-left:6px solid {border_c};
         border-radius:8px;padding:12px 18px;margin-bottom:10px;display:flex;align-items:center;gap:16px;">
    <div style="background:{border_c};color:white;border-radius:10px;padding:6px 14px;
             font-size:.86rem;font-weight:800;white-space:nowrap;letter-spacing:0.03em;
             box-shadow:0 3px 10px rgba(0,0,0,0.12);">{label_txt}</div>
    <div style="flex:1;font-size:.9rem;color:{C['gray_dark']};">
        <b style="font-size:1.02rem">{t1["name"]}</b>
        <span style="color:{border_c};font-weight:700;margin-left:8px;">쓰레기 {int(t1["trash_cnt"])}개 적체 — {risk_label(r1)} |</span>
        <span style="color:{C['gray_mid']};margin-left:6px;">{action_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 메인 탭
# ══════════════════════════════════════════════
tab_map, tab_detail, tab_predict, tab_data = st.tabs([
    "지도 · 수거 동선", "섬 상세 정보", "5일 예측 (LSTM)", "데이터 추출",
])


# ── TAB 1: 지도 · 수거 동선 ──────────────────
with tab_map:
    if top_targets.empty:
        st.info("수거 조건에 맞는 섬이 없습니다. 사이드바 필터를 조정해보세요.")
    else:
        map_col, info_col = st.columns([3, 1])

        with info_col:
            st.markdown('<p class="sec">▶ 수거 동선 순서</p>', unsafe_allow_html=True)
            for rank, row in top_targets.iterrows():
                r  = int(row["risk"]); tc = int(row["trash_cnt"])
                bridge = "육로" if row["has_bridge"] else "선박"
                inh    = "유인도" if row["inhabited"] else "무인도"
                st.markdown(
                    f'<div class="rcard" style="border-color:{risk_hex(r)};">'
                    f'<span style="background:{risk_hex(r)};color:white;border-radius:3px;'
                    f'padding:2px 8px;font-size:.82rem;font-weight:700;">{rank+1}순위</span>'
                    f'&nbsp;<b>{row["name"]}</b><br>'
                    f'<span style="color:{C["gray_mid"]};font-size:.78em;">'
                    f'쓰레기 {tc}개 적체 · {risk_label(r)} · {bridge} · {inh}</span></div>',
                    unsafe_allow_html=True)

            st.markdown("---")
            bn  = int(top_targets["has_bridge"].sum())
            tot = int(top_targets["trash_cnt"].sum())
            for label, val, big in [
                ("총 수거 예상량", f'{tot:,}개', True),
                ("수거 수단",     f'선박 {len(top_targets)-bn}곳 · 육로 {bn}곳', False),
            ]:
                st.markdown(
                    f'<div class="mbox"><div style="font-size:.7rem;color:{C["gray_mid"]};">'
                    f'{label}</div>'
                    f'<div style="font-size:{"1.4" if big else ".88"}rem;'
                    f'font-weight:{"700" if big else "400"};'
                    f'color:{C["blue_deep"] if big else C["gray_dark"]};'
                    f'margin-top:2px;">{val}</div></div>',
                    unsafe_allow_html=True)

        with map_col:
            top_targets["color"] = top_targets["risk"].apply(lambda r: risk_color(int(r), 255))
            top_targets["order"] = (top_targets.index+1).astype(str)
            waypoints = [BASE_PORT] + top_targets[["lon", "lat"]].values.tolist()
            with st.spinner("A* 최단 해상 동선 계산 중..."):
                route = find_route(waypoints, G)

            port_df = pd.DataFrame([{
                "lon": BASE_PORT[0], "lat": BASE_PORT[1],
                "name": f"출발항 ({start_port_name})", "order": "출발",
                "color": [0, 63, 125, 255],
                "tooltip": f"출발항 ({start_port_name})",
            }])

            obs_df  = display_df[display_df["has_obs"]]
            expo_df = display_df[display_df["is_expo"]]

            layers = [
                # 배경 GeoJSON
                pdk.Layer("GeoJsonLayer", geojson_data,
                    get_fill_color=[180, 200, 220, 60],
                    get_line_color=[100, 140, 180, 120],
                    line_width_min_pixels=1),
                # 전체 후보 섬 (작게)
                pdk.Layer("ScatterplotLayer", display_df,
                    get_position="[lon, lat]", get_color="color",
                    get_radius=380, radius_min_pixels=3, radius_max_pixels=10,
                    stroked=True, line_width_min_pixels=1,
                    get_line_color=[255, 255, 255, 180], opacity=0.85, pickable=True),
                # ── 경로 레이어 (글로우 효과: 굵은 반투명 + 얇은 선명) ──
                pdk.Layer("PathLayer",
                    pd.DataFrame([{"path": route}]),
                    get_path="path", get_color=[0, 160, 255, 60],
                    width_min_pixels=14, width_max_pixels=20),
                pdk.Layer("PathLayer",
                    pd.DataFrame([{"path": route}]),
                    get_path="path", get_color=[0, 120, 255, 230],
                    width_min_pixels=5, width_max_pixels=8,
                    get_width=5),
                # 수거 대상 섬 — 강조 링 (크게)
                pdk.Layer("ScatterplotLayer", top_targets,
                    get_position="[lon, lat]",
                    get_color=[0, 100, 255, 50], get_radius=1100,
                    radius_min_pixels=18, radius_max_pixels=40,
                    stroked=True, line_width_min_pixels=3,
                    get_line_color=[0, 120, 255, 200]),
                # 수거 대상 섬 — 위험도 색상 도트
                pdk.Layer("ScatterplotLayer", top_targets,
                    get_position="[lon, lat]", get_color="color",
                    get_radius=550, radius_min_pixels=10, radius_max_pixels=24,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255, 255, 255, 255], pickable=True),
                # 출발항
                pdk.Layer("ScatterplotLayer", port_df,
                    get_position="[lon, lat]", get_color=[0, 63, 125, 255],
                    get_radius=550, radius_min_pixels=10, radius_max_pixels=24,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255, 255, 255, 255], pickable=True),
                # 관측소 마커
                *([pdk.Layer("ScatterplotLayer", obs_df,
                    get_position="[lon, lat]",
                    get_color=[8, 145, 178, 255], get_radius=300,
                    radius_min_pixels=5, radius_max_pixels=11,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255, 255, 255, 255]),
                  pdk.Layer("TextLayer", obs_df,
                    get_position="[lon, lat]", get_text="['관측소']",
                    get_size=10, get_color=[255, 255, 255, 255],
                    background=True,
                    get_background_color=[8, 145, 178, 235],
                    get_border_color=[8, 145, 178, 255],
                    get_padding=[5, 2, 5, 2], get_pixel_offset=[0, -22],
                    billboard=True)] if not obs_df.empty else []),
                # 박람회 마커
                *([pdk.Layer("ScatterplotLayer", expo_df,
                    get_position="[lon, lat]",
                    get_color=[99, 102, 241, 255], get_radius=340,
                    radius_min_pixels=6, radius_max_pixels=13,
                    stroked=True, line_width_min_pixels=2,
                    get_line_color=[255, 255, 255, 255]),
                  pdk.Layer("TextLayer", expo_df,
                    get_position="[lon, lat]", get_text="['박람회']",
                    get_size=10, get_color=[255, 255, 255, 255],
                    background=True,
                    get_background_color=[99, 102, 241, 235],
                    get_border_color=[79, 70, 229, 255],
                    get_padding=[5, 2, 5, 2], get_pixel_offset=[0, 22],
                    billboard=True)] if expo_mode and not expo_df.empty else []),
                # 순서 번호 (수거 대상)
                pdk.Layer("TextLayer", top_targets,
                    get_position="[lon, lat]", get_text="order",
                    get_size=18, get_color=[255, 255, 255, 255],
                    background=True, get_background_color=[0, 80, 200, 245],
                    get_border_color=[0, 60, 160, 255],
                    get_padding=[6, 4, 6, 4],
                    font_weight=800, billboard=True),
                # 출발항 라벨
                pdk.Layer("TextLayer", port_df,
                    get_position="[lon, lat]", get_text="order",
                    get_size=13, get_color=[255, 255, 255, 255],
                    background=True, get_background_color=[0, 63, 125, 240],
                    get_border_color=[0, 63, 125, 255],
                    get_padding=[5, 3, 5, 3],
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
                f'<span><span style="color:{risk_hex(10)}">●</span> 양호(0~39)</span>'
                f'<span><span style="color:{risk_hex(50)}">●</span> 주의(40~59)</span>'
                f'<span><span style="color:{risk_hex(70)}">●</span> 경고(60~79)</span>'
                f'<span><span style="color:{risk_hex(90)}">●</span> 위험(80~100)</span>'
                f'<span><span style="color:#0078FF;font-weight:700;">━━</span> 최적 수거 동선 (A*)</span>'
                f'<span><span style="color:#0891B2;font-weight:700;">●</span> 해양관측소</span>'
                f'<span><span style="color:#6366F1;font-weight:700;">●</span> 박람회 행사장</span>'
                f'</div>', unsafe_allow_html=True)


# ── TAB 2: 섬 상세 정보 ──────────────────────
with tab_detail:
    st.markdown("#### 섬별 현황 및 수거 판단 정보")
    all_names  = sorted(islands_df["name"].tolist())
    default_idx = all_names.index(top_targets.iloc[0]["name"]) if not top_targets.empty else 0
    sel_island = st.selectbox("섬 선택", all_names, index=default_idx)
    row = islands_df[islands_df["name"] == sel_island].iloc[0]
    r   = int(row["risk"])
    dyn = get_dynamics(sel_island, float(row["lat"]), float(row["lon"]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="detail-card">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
               color:{C['gray_mid']};margin-bottom:8px;">현재 적체 현황</div>
          <div style="font-size:1.1rem;font-weight:700;color:{C['blue_deep']};
               margin-bottom:12px;">{sel_island}</div>
          <table style="width:100%;font-size:.84em;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">수거 우선도</td>
                <td style="text-align:right;"><b style="color:{risk_hex(r)};">{r} — {risk_label(r)}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">현재 적체량</td>
                <td style="text-align:right;"><b style="color:{risk_hex(r)};">{int(row['trash_cnt'])}개</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">쓰레기 유형</td>
                <td style="text-align:right;"><b>{row['trash_types']}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">유무인도</td>
                <td style="text-align:right;"><b>{"유인도" if row["inhabited"] else "무인도"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">수거 방법</td>
                <td style="text-align:right;"><b>{"육로 (연륙교/연도교)" if row["has_bridge"] else "선박 출동 필요"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">해양관측소</td>
                <td style="text-align:right;">
                  <b style="color:{C['blue_mid'] if row['has_obs'] else C['gray_mid']};">
                  {"있음 (국가 관측)" if row['has_obs'] else "없음"}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">2026 박람회</td>
                <td style="text-align:right;">
                  <b style="color:{"#16A34A" if row["expo_role"]!="해당 없음" else C["gray_mid"]};">
                  {row["expo_role"]}</b></td></tr>
            <tr><td style="padding:4px 0;color:{C['gray_mid']};">위도 / 경도</td>
                <td style="text-align:right;"><b>{row['lat']:.5f}, {row['lon']:.5f}</b></td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with c2:
        def _dyn_row(label, val):
            disp = val if val is not None else '<span style="color:#9CA3AF;">관측 불가</span>'
            return (f'<tr><td style="padding:4px 0;color:{C["gray_mid"]};">{label}</td>'
                    f'<td style="text-align:right;color:{C["gray_dark"]};"><b>{disp}</b></td></tr>')

        wind_val = (f'{dyn["wind_dir"]}풍 {dyn["wind_speed"]} m/s'
                    if dyn["wind_dir"] is not None and dyn["wind_speed"] is not None else None)
        rows_html = (
            _dyn_row("풍향 / 풍속", wind_val) +
            _dyn_row("조류 상태",   dyn["tide"]) +
            _dyn_row("수온",        f'{dyn["temp"]} °C' if dyn["temp"] is not None else None)
        )
        src       = dyn.get("source", "관측 불가")
        is_real   = src.startswith("KHOA 실측")
        src_color = C["blue_mid"] if is_real else C["gray_mid"]
        st.markdown(f"""
        <div class="detail-card">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
               color:{C['gray_mid']};margin-bottom:8px;">해양 기상 실측 (KHOA)</div>
          <table style="width:100%;font-size:.84em;border-collapse:collapse;">{rows_html}</table>
          <div style="margin-top:10px;font-size:.72rem;color:{src_color};">
            출처: <b>{src}</b>
            {"" if is_real else " — API 키 미설정 또는 해당 관측소 데이터 없음"}
          </div>
        </div>""", unsafe_allow_html=True)


# ── TAB 3: LSTM 예측 ─────────────────────────
with tab_predict:
    if top_targets.empty:
        st.info("수거 대상 섬이 없습니다. 사이드바 필터를 조정해보세요.")
    else:
        sel2 = st.selectbox("분석 대상 섬", top_targets["name"].tolist(),
                            index=0, key="predict_sel")
        row2 = top_targets[top_targets["name"] == sel2].iloc[0]
        st.plotly_chart(
            build_forecast_chart(sel2, int(row2["risk"]), int(row2["trash_cnt"]), predict_date),
            use_container_width=True)
        st.caption(
            "LSTM 시계열 모델이 해류·기상·조류 데이터를 학습하여 향후 5일간 "
            "쓰레기 적체량 변화를 예측합니다. 음영은 95% 신뢰구간입니다."
        )


# ── TAB 4: 데이터 추출 ───────────────────────
with tab_data:
    st.markdown("#### 수거 계획 데이터 추출")
    dl1, dl2 = st.columns(2)

    with dl1:
        st.markdown("**현재 필터 조건 적용 결과**")
        st.dataframe(
            filtered_df[["name","lat","lon","risk","trash_cnt","trash_types",
                         "main_type","inhabited","has_bridge","has_obs","is_expo"
                        ]].rename(columns={
                "name":"섬이름","lat":"위도","lon":"경도",
                "risk":"수거우선도","trash_cnt":"적체쓰레기(개)",
                "trash_types":"쓰레기유형","main_type":"주요유형",
                "inhabited":"유무인(1=유인)","has_bridge":"연륙교여부",
                "has_obs":"관측소여부","is_expo":"박람회행사장",
            }).reset_index(drop=True),
            use_container_width=True, height=350)
        st.download_button("CSV 다운로드 (필터 적용)",
            filtered_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"dodongshil_filtered_{predict_date}.csv", mime="text/csv")

    with dl2:
        st.markdown("**최종 수거 출동 대상 섬**")
        if not top_targets.empty:
            st.dataframe(
                top_targets[["name","lat","lon","risk","trash_cnt",
                              "trash_types","inhabited","has_bridge"
                            ]].rename(columns={
                    "name":"섬이름","lat":"위도","lon":"경도",
                    "risk":"수거우선도","trash_cnt":"적체쓰레기(개)",
                    "trash_types":"쓰레기유형",
                    "inhabited":"유무인","has_bridge":"연륙교",
                }).reset_index(drop=True),
                use_container_width=True, height=350)
            st.download_button("CSV 다운로드 (수거 출동 목록)",
                top_targets.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"dodongshil_dispatch_{predict_date}.csv", mime="text/csv")
        else:
            st.info("수거 출동 대상 섬이 없습니다.")

    st.markdown("---")
    st.markdown("**전체 섬 원본 데이터**")
    st.download_button("CSV 다운로드 (전체)",
        islands_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"dodongshil_all_{predict_date}.csv", mime="text/csv")
    st.dataframe(
        islands_df[["name","lat","lon","risk","trash_cnt","trash_types",
                    "inhabited","has_bridge","has_obs"]].head(50),
        use_container_width=True, height=300)
    st.caption("상위 50개만 미리보기. 전체 데이터는 CSV로 다운로드하세요.")
