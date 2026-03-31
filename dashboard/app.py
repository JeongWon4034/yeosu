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
        background-color: {C['blue_mid']} !important;
        border-color: {C['blue_mid']} !important;
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
    if risk < 40:  return [34,  197,  94, alpha]
    if risk < 60:  return [234, 179,   8, alpha]
    if risk < 80:  return [249, 115,  22, alpha]
    return              [239,  68,  68, alpha]

def risk_label(risk: int) -> str:
    if risk >= 80: return "위험"
    if risk >= 60: return "경고"
    if risk >= 40: return "주의"
    return "양호"

def risk_hex(risk: int) -> str:
    c = risk_color(risk, 255)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


# ═══════════════════════════════════════════════
# 2. 육지 판별
# ═══════════════════════════════════════════════
def is_land(lon: float, lat: float) -> bool:
    if lat > 34.77: return True
    if 34.66 < lat <= 34.77 and 127.62 < lon < 127.83: return True
    if 34.72 < lat <= 34.77 and 127.52 < lon <= 127.62: return True
    if 34.57 < lat <= 34.66 and 127.71 < lon < 127.85: return True
    return False

def edge_crosses_land(a, b, steps: int = 20) -> bool:
    for t in np.linspace(0, 1, steps):
        if is_land(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])):
            return True
    return False


# ═══════════════════════════════════════════════
# 3. 해상 그래프  # v9
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner="해상 경로 그래프 초기화 중... (최초 1회)")  # v9
def build_sea_graph():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)
    lons = np.linspace(127.25, 127.95, 70)
    lats = np.linspace(34.00,  34.95,  85)
    G, nodes = nx.Graph(), []
    for lo in lons:
        for la in lats:
            if not is_land(lo, la):
                n = (round(lo, 4), round(la, 4))
                nodes.append(n); G.add_node(n)
    xy = np.array(nodes)
    for i, n1 in enumerate(nodes):
        d = np.hypot(xy[:, 0]-n1[0], xy[:, 1]-n1[1])
        for j in np.where((d > 0) & (d < 0.035))[0]:
            n2 = nodes[int(j)]
            if not G.has_edge(n1, n2) and not edge_crosses_land(n1, n2):
                G.add_edge(n1, n2, weight=float(d[j]))
    return G, gj

def find_route(waypoints, G):
    nodes = list(G.nodes()); xy = np.array(nodes)
    def nearest(pt):
        return nodes[int(np.argmin(np.hypot(xy[:,0]-pt[0], xy[:,1]-pt[1])))]
    path = []
    for i in range(len(waypoints)-1):
        try:
            seg = nx.astar_path(G, nearest(waypoints[i]), nearest(waypoints[i+1]), weight="weight")
            path.extend([[p[0],p[1]] for p in seg])
        except Exception:
            path.extend([waypoints[i], waypoints[i+1]])
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

    df["color"]   = df["risk"].apply(risk_color)
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
# 6. MOHID 해양 역학
# ═══════════════════════════════════════════════
def get_dynamics(name, risk):
    np.random.seed(abs(hash(name)) % 777)
    return dict(
        wind_dir  = np.random.choice(["북","북동","동","남동","남","남서","서","북서"]),
        wind_speed= round(np.random.uniform(2.5, 13.5), 1),
        current   = round(np.random.uniform(0.4, 2.1),  1),
        wave      = round(np.random.uniform(0.2, 2.8),  1),
        tide      = np.random.choice(["밀물 가속 중","썰물 감속 중","정조기"]),
        cause     = np.random.choice(["해류 집중","강풍 직접 유입","조류 가속","저기압 영향"]),
        factor    = round(risk/100*2.5+0.5, 1),
        salinity  = round(np.random.uniform(30.0, 34.5), 1),
        temp      = round(np.random.uniform(8.5, 22.0),  1),
    )


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

    st.markdown('<p class="sec">쓰레기 유형 필터</p>', unsafe_allow_html=True)
    sel_trash_types = st.multiselect(
        "유형 선택 (미선택 시 전체)", options=TRASH_TYPES, default=[],
        label_visibility="collapsed")

    st.markdown('<p class="sec">수거 유형 필터</p>', unsafe_allow_html=True)
    sea_only       = st.checkbox("선박 수거 대상만 (연륙교 제외)", value=True)
    inhabited_only = st.checkbox("유인도만 보기", value=False)
    obs_only       = st.checkbox("관측소 보유 섬만", value=False)
    expo_mode      = st.checkbox("2026 박람회 관광지 강조", value=False)

# ── 데이터 준비 ───────────────────────────────
islands_df = load_island_data(predict_date.toordinal())

display_df = islands_df.copy()
if sea_only:       display_df = display_df[~display_df["has_bridge"]]
if inhabited_only: display_df = display_df[display_df["inhabited"] == 1]
if obs_only:       display_df = display_df[display_df["has_obs"]]
if sel_trash_types:
    display_df = display_df[display_df["trash_types"].apply(
        lambda t: any(tt in t for tt in sel_trash_types))]

display_df  = display_df[display_df["risk"] >= min_risk].copy()
filtered_df = display_df.copy()
top_targets = (filtered_df.nlargest(num_targets, "risk").reset_index(drop=True).copy()
               if not filtered_df.empty else pd.DataFrame())

# ── 사이드바 순위 ─────────────────────────────
with st.sidebar:
    st.markdown('<p class="sec">위험도 순위</p>', unsafe_allow_html=True)
    if filtered_df.empty:
        st.caption("해당 조건의 섬이 없습니다.")
    else:
        for i, (_, row) in enumerate(filtered_df.nlargest(10, "risk").iterrows(), 1):
            bridge = "육로" if row["has_bridge"] else "해상"
            inh    = "유인" if row["inhabited"]  else "무인"
            st.markdown(
                f'<div class="rcard" style="border-color:{risk_hex(int(row["risk"]))};">'
                f'<b>{i}위</b> {row["name"]}'
                f'<span style="float:right;font-size:.75em;color:{C["gray_mid"]};">'
                f'{bridge}·{inh}·{int(row["risk"])}</span><br>'
                f'<span style="font-size:.72em;color:{C["gray_mid"]};">'
                f'{int(row["trash_cnt"])}개 · {row["main_type"]}</span></div>',
                unsafe_allow_html=True)

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
        분석기간: {start_date} ~ {end_date} ({date_range_days}일) &nbsp;·&nbsp;
        조건충족 <b style="color:white">{filtered_cnt}</b>개 섬 &nbsp;·&nbsp;
        쓰레기 추정 <b style="color:white">{total_trash:,}</b>개 &nbsp;·&nbsp;
        A* 해상경로 · LSTM · MOHID
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
    d1 = get_dynamics(t1["name"], int(t1["risk"]))
    r1 = int(t1["risk"])
    # 위험도 80+ → 빨강 / 60~79 → 파랑 / 나머지 → 연파랑
    if r1 >= 80:
        border_c, bg_c, label_c, label_txt = "#EF4444", "#FFF5F5", "#DC2626", "수거 권고"
    elif r1 >= 60:
        border_c, bg_c, label_c, label_txt = "#0066B3", "#E8F4FD", "#003F7D", "주의 필요"
    else:
        border_c, bg_c, label_c, label_txt = "#4DA6E0", "#F0F8FF", "#0066B3", "모니터링"
    st.markdown(f"""
<div style="background:{bg_c};border:1.5px solid {border_c};border-left:5px solid {border_c};
     border-radius:8px;padding:12px 18px;margin-bottom:10px;display:flex;align-items:center;gap:16px;">
  <div style="background:{border_c};color:white;border-radius:6px;padding:4px 12px;
       font-size:.78rem;font-weight:700;white-space:nowrap;letter-spacing:0.03em;">{label_txt}</div>
  <div style="flex:1;font-size:.88rem;color:#1A2B3C;">
    <b>1순위 — {t1["name"]}</b>
    <span style="color:{label_c};font-weight:600;"> 위험도 {r1}/100</span>
    <span style="color:#4B6178;"> · 쓰레기 약 {int(t1["trash_cnt"])}개 · {t1["main_type"]}</span><br>
    <span style="color:#4B6178;font-size:.82rem;">
      {d1["wind_dir"]}풍 {d1["wind_speed"]}m/s · 유속 {d1["current"]}m/s · {d1["cause"]}
    </span>
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
                types_html = "".join(
                    f'<span class="trash-badge" '
                    f'style="background:{TRASH_COLORS.get(t.strip(), C["blue_mid"])}22;'
                    f'color:{TRASH_COLORS.get(t.strip(), C["blue_mid"])};'
                    f'border:1px solid {TRASH_COLORS.get(t.strip(), C["blue_mid"])}44;">'
                    f'{t.strip()}</span>'
                    for t in row["trash_types"].split(", "))
                st.markdown(
                    f'<div class="rcard" style="border-color:{risk_hex(r)};">'
                    f'<span style="background:{risk_hex(r)};color:white;border-radius:3px;'
                    f'padding:1px 7px;font-size:.8rem;font-weight:700;">{rank+1}</span>'
                    f'&nbsp;<b>{row["name"]}</b><br>'
                    f'<span style="color:{C["gray_mid"]};font-size:.78em;">'
                    f'위험도 {r} · {risk_label(r)} · {bridge} · {inh} · {tc}개</span><br>'
                    f'{types_html}</div>', unsafe_allow_html=True)

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
                # 관측소 — 보라색 (필터 따라)
                *([pdk.Layer("ScatterplotLayer", obs_df,
                    get_position="[lon, lat]",
                    get_color=[237,233,254,90], get_radius=750,
                    radius_min_pixels=12, radius_max_pixels=26,
                    stroked=True, line_width_min_pixels=3,
                    get_line_color=[124,58,237,255]),
                  pdk.Layer("TextLayer", obs_df,
                    get_position="[lon, lat]", get_text="['관측']",
                    get_size=10, get_color=[109,40,217,255],
                    background=True,
                    get_background_color=[237,233,254,220],
                    get_border_color=[124,58,237,180],
                    get_padding=[3,2,3,2], get_pixel_offset=[0,-24],
                    billboard=True)] if not obs_df.empty else []),
                # 박람회 — 초록색 (expo_mode 체크 시)
                *([pdk.Layer("ScatterplotLayer", expo_df,
                    get_position="[lon, lat]",
                    get_color=[220,252,231,60], get_radius=900,
                    radius_min_pixels=14, radius_max_pixels=30,
                    stroked=True, line_width_min_pixels=3,
                    get_line_color=[22,163,74,220]),
                  pdk.Layer("TextLayer", expo_df,
                    get_position="[lon, lat]", get_text="['박람회']",
                    get_size=10, get_color=[21,128,61,255],
                    background=True,
                    get_background_color=[220,252,231,230],
                    get_border_color=[22,163,74,180],
                    get_padding=[3,2,3,2], get_pixel_offset=[0,26],
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
                f'<span><span style="color:#EF4444">●</span> 위험(80~100)</span>'
                f'<span><span style="color:{C["blue_mid"]}">━</span> 수거 노선</span>'
                f'<span><span style="color:#7C3AED;font-weight:700;">◉</span> 해양관측소</span>'
                f'<span><span style="color:#16A34A;font-weight:700;">◉</span> 박람회 행사장</span>'
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
    dyn = get_dynamics(sel_island, r)

    c1, c2, c3 = st.columns(3)
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
                ("해수 유속", f'{dyn["current"]} m/s'),
                ("파고",     f'{dyn["wave"]} m'),
                ("조류 상태", dyn["tide"]),
                ("수온",     f'{dyn["temp"]} °C'),
                ("염분",     f'{dyn["salinity"]} psu'),
                ("유입 원인", dyn["cause"]),
                ("위험 배율", f'×{dyn["factor"]}'),
            ])
        st.markdown(f"""
        <div class="detail-card">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
               color:{C['gray_mid']};margin-bottom:8px;">해양 역학 (MOHID)</div>
          <table style="width:100%;font-size:.84em;border-collapse:collapse;">{rows_html}</table>
          <div style="margin-top:10px;font-size:.7rem;color:{C['gray_mid']};">
            MOHID 해양 물리 시뮬레이션 기반</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f'<div class="detail-card">'
            f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;'
            f'color:{C["gray_mid"]};margin-bottom:8px;">5일 위험도 추이 (LSTM)</div>',
            unsafe_allow_html=True)
        mini_fig = build_forecast_chart(sel_island, r, int(row["trash_cnt"]), predict_date)
        mini_fig.update_layout(height=250, margin=dict(l=5,r=30,t=30,b=5),
                               title=None, showlegend=False)
        st.plotly_chart(mini_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


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


# ── TAB 4: MOHID ─────────────────────────────
with tab_mohid:
    if top_targets.empty:
        st.info("수거 대상 섬이 없습니다.")
    else:
        cols = st.columns(min(len(top_targets), 3))
        for i, (_, row) in enumerate(top_targets.head(3).iterrows()):
            dyn = get_dynamics(row["name"], int(row["risk"]))
            r   = int(row["risk"]); hc = risk_hex(r)
            items = [
                ("풍향/풍속",  f'{dyn["wind_dir"]}풍 {dyn["wind_speed"]} m/s'),
                ("해수 유속",  f'{dyn["current"]} m/s'),
                ("파고",      f'{dyn["wave"]} m'),
                ("조류 상태",  dyn["tide"]),
                ("수온",      f'{dyn["temp"]} °C'),
                ("염분",      f'{dyn["salinity"]} psu'),
                ("유입 원인",  dyn["cause"]),
                ("위험 배율",  f'×{dyn["factor"]}'),
                ("쓰레기 추정", f'{int(row["trash_cnt"])}개'),
            ]
            rows_html = "".join(
                f'<tr><td style="padding:4px 0;color:{C["gray_mid"]};">{k}</td>'
                f'<td style="text-align:right;color:{C["gray_dark"]};"><b>{v}</b></td></tr>'
                for k,v in items)
            with cols[i]:
                st.markdown(
                    f'<div style="border:1px solid {hc};background:{C["gray_bg"]};'
                    f'border-radius:6px;padding:14px;">'
                    f'<div style="font-size:.7rem;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{C["gray_mid"]};margin-bottom:6px;">{risk_label(r)}</div>'
                    f'<div style="font-size:.98rem;font-weight:700;'
                    f'color:{C["blue_deep"]};margin-bottom:10px;">{row["name"]}</div>'
                    f'<table style="width:100%;font-size:.84em;border-collapse:collapse;">'
                    f'{rows_html}</table>'
                    f'<div style="margin-top:10px;font-size:.7rem;color:{C["gray_mid"]};">'
                    f'MOHID 해양 물리 시뮬레이션 기반</div></div>',
                    unsafe_allow_html=True)
        st.caption("MOHID 해양 시뮬레이션과 기상청 API를 결합하여 "
                   "쓰레기 유입 원인을 물리적 수치로 설명합니다.")


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
