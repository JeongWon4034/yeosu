"""
해양쓰레기 예측 프로젝트 - 데이터 전처리 스크립트
"""
import pandas as pd
import numpy as np
import glob
import os

ROOT = "/Users/jeongwon/yeosu"
FINAL = os.path.join(ROOT, "final_data")
MOHID = os.path.join(FINAL, "Mohid_prepare_data")


def describe_df(name, df):
    print(f"\n{'='*60}")
    print(f"[{name}]  shape={df.shape}")
    print(f"  columns: {df.columns.tolist()}")
    print(f"  결측치:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"  기본 통계:")
    num = df.select_dtypes(include="number")
    if not num.empty:
        print(num.describe().to_string())


# ─────────────────────────────────────────────────────────
# 1. MOHID 결과 + 정답지 병합
# ─────────────────────────────────────────────────────────
print("\n>>> STEP 1: MOHID + 정답지 병합")

mohid = pd.read_csv(os.path.join(FINAL, "섬_쓰레기_입자개수_3개월_.csv"))
answer = pd.read_csv(os.path.join(FINAL, "남해_정답지데이터.csv"))

describe_df("MOHID 원본", mohid)
describe_df("정답지 원본", answer)

# source_count 결측치 평균값으로 대체
mean_sc = mohid["source_count"].mean()
nan_rows = mohid[mohid["source_count"].isna()]["name"].tolist()
print(f"\n  source_count 결측 행: {nan_rows}")
print(f"  대체값 (평균): {mean_sc:.4f}")
mohid["source_count"] = mohid["source_count"].fillna(mean_sc)

# inner join: MOHID name ↔ 정답지 지역명
merged = pd.merge(
    mohid[["name", "source_count"]],
    answer[["지역명", "수량(개)", "무게(kg)", "Latitude", "Longitude"]],
    left_on="name",
    right_on="지역명",
    how="inner",
)
merged = merged.drop(columns=["name"])
print(f"\n  병합 결과: {len(merged)}개 행")
print(merged.to_string())

out_path = os.path.join(FINAL, "island_merged.csv")
merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  저장: {out_path}")
describe_df("island_merged", merged)


# ─────────────────────────────────────────────────────────
# 2. 환경변수 3종 병합
# ─────────────────────────────────────────────────────────
print("\n>>> STEP 2: 환경변수 시계열 병합")

# 2-A. 조위 (obs_tide, pred_tide) - 관측소별 시간평균
tide_raw = pd.read_csv(os.path.join(MOHID, "namhae_tide_weather_master_1hr.csv"))
tide_raw["datetime"] = pd.to_datetime(tide_raw["datetime"])
describe_df("조위+기상 원본", tide_raw)

tide_agg = (
    tide_raw.groupby("datetime")[["obs_tide", "pred_tide"]]
    .mean()
    .rename(columns={"obs_tide": "tide", "pred_tide": "tide_pred"})
    .reset_index()
)

# 2-B. 기상 (wind) - 관측소별 시간평균
weather_raw = pd.read_csv(os.path.join(MOHID, "namhae_weather_master_1hr_final2.csv"))
weather_raw["datetime"] = pd.to_datetime(weather_raw["time"])
weather_raw = weather_raw.drop(columns=["time"])
describe_df("기상 원본", weather_raw)

weather_agg = (
    weather_raw.groupby("datetime")[["wind_dir_deg", "wind_speed_ms", "u_wind", "v_wind"]]
    .mean()
    .rename(
        columns={
            "wind_dir_deg": "wind_dir",
            "wind_speed_ms": "wind_speed",
        }
    )
    .reset_index()
)

# 2-C. 해류 (u_current, v_current) - 6개 파일 합본 후 남해 영역 평균
flow_files = sorted(glob.glob(os.path.join(MOHID, "namhae_water_flow_*.txt")))
print(f"\n  해류 파일 {len(flow_files)}개: {[os.path.basename(f) for f in flow_files]}")

flow_dfs = []
for fpath in flow_files:
    df = pd.read_csv(fpath)
    flow_dfs.append(df)

flow_all = pd.concat(flow_dfs, ignore_index=True)
flow_all["datetime"] = pd.to_datetime(flow_all["time"])
flow_all = flow_all.drop(columns=["time"])

# 남해 영역 필터 (위도 33~36, 경도 124~130)
mask = (
    (flow_all["lat"] >= 33) & (flow_all["lat"] <= 36) &
    (flow_all["lon"] >= 124) & (flow_all["lon"] <= 130)
)
flow_region = flow_all[mask].copy()
print(f"  남해 영역 필터: {len(flow_all)} → {len(flow_region)} 행")
describe_df("해류 원본 (남해 영역)", flow_region)

flow_agg = (
    flow_region.groupby("datetime")[["u_current_ms", "v_current_ms"]]
    .mean()
    .rename(columns={"u_current_ms": "u_current", "v_current_ms": "v_current"})
    .reset_index()
)

# 2-D. 3종 합치기 (outer join → 선형보간)
env = pd.merge(tide_agg, weather_agg, on="datetime", how="outer")
env = pd.merge(env, flow_agg, on="datetime", how="outer")
env = env.sort_values("datetime").reset_index(drop=True)

# 선형보간
cols_to_interp = ["tide", "tide_pred", "wind_speed", "wind_dir", "u_wind", "v_wind", "u_current", "v_current"]
env[cols_to_interp] = env[cols_to_interp].interpolate(method="linear", limit_direction="both")

env = env[["datetime"] + cols_to_interp]

out_path = os.path.join(FINAL, "env_timeseries_merged.csv")
env.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  저장: {out_path}")
describe_df("env_timeseries_merged", env)


# ─────────────────────────────────────────────────────────
# 3. 방출점 원핫인코딩
# ─────────────────────────────────────────────────────────
print("\n>>> STEP 3: 방출점 원핫인코딩")

rp = pd.read_csv(os.path.join(MOHID, "namhae_release_points_127_최종.csv"))
describe_df("방출점 원본", rp)

rp_enc = pd.get_dummies(rp, columns=["type"], prefix="type")
# bool → int
bool_cols = rp_enc.select_dtypes(include="bool").columns
rp_enc[bool_cols] = rp_enc[bool_cols].astype(int)

out_path = os.path.join(FINAL, "release_points_encoded.csv")
rp_enc.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  저장: {out_path}")
print(f"  컬럼: {rp_enc.columns.tolist()}")
describe_df("release_points_encoded", rp_enc)

print("\n" + "="*60)
print("전처리 완료!")
print(f"  - {FINAL}/island_merged.csv")
print(f"  - {FINAL}/env_timeseries_merged.csv")
print(f"  - {FINAL}/release_points_encoded.csv")
