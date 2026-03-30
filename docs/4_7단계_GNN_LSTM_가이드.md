# 해양쓰레기 예측 모델: 4~7단계 가이드

## 목차
1. [GNN+LSTM이 뭔지 쉽게 이해하기](#1-gnnlstm이-뭔지-쉽게-이해하기)
2. [우리 데이터에 맞는 현실적 모델 설계](#2-우리-데이터에-맞는-현실적-모델-설계)
3. [4단계: 모델 구축 상세 흐름](#3-4단계-모델-구축)
4. [5단계: 검증](#4-5단계-검증)
5. [6단계: 여수 335개 섬 예측](#5-6단계-여수-335개-섬-예측)
6. [7단계: 최종 시각화](#6-7단계-최종-시각화)
7. [단계별 코딩 프롬프트](#7-단계별-코딩-프롬프트)

---

## 1. GNN+LSTM이 뭔지 쉽게 이해하기

### GNN을 비유로 설명하면

**카카오톡 단톡방**을 생각하자.

- 단톡방에 사람(=노드)이 있고, 서로 연결(=엣지)이 되어 있다.
- 어떤 사람이 메시지(=쓰레기)를 보내면, 연결된 사람들에게 전달된다.
- GNN은 "누가 누구에게 얼마나 영향을 주는가"를 학습하는 AI다.

**우리 연구에 적용하면:**

```
[방출점들]              [바다(해류)]            [섬들]
항만 ─────────────┐
해변 ──────────┐  │    해류가 연결해줌      ┌── 여수반월
하천 ────────┐ │  │  ==================>   ├── 여수백야도
양식장 ────┐ │ │  │                        ├── 완도신지도
           ▼ ▼ ▼  ▼                        ▼
        [GNN이 학습하는 것]
        "항만A에서 나온 쓰레기가
         여수반월 섬에 얼마나 갈까?"
```

- **노드(Node)** = 방출점(항만 68개, 해변 133개, 하천 33개, 어업 14개) + 섬(남해 32개)
- **엣지(Edge)** = MOHID가 시뮬레이션한 해류 연결 (방출점→섬으로 입자가 도달하면 연결)
- **GNN의 역할** = 공간적 관계 학습 → "어느 방출점이 어느 섬에 영향을 주는가"

### LSTM을 비유로 설명하면

**일기예보**를 생각하자.

- 오늘 날씨만 보고 내일을 예측하면 정확도가 낮다.
- 지난 3일간의 날씨 변화 패턴을 보면 훨씬 정확하게 예측한다.
- LSTM은 "시간 흐름에 따른 패턴"을 기억하고 학습하는 AI다.

**우리 연구에 적용하면:**

```
[시간 흐름 데이터]
  1월: 북풍 강함 + 조위 낮음 → 쓰레기 남쪽으로 이동
  2월: 서풍 강함 + 조위 높음 → 쓰레기 동쪽으로 이동
  3월: 남풍 + 대조기         → 쓰레기 해안 축적
                    ↓
  [LSTM이 학습하는 것]
  "풍향이 바뀌고 조위가 높아지면
   3일 후에 쓰레기가 이 섬에 몰린다"
```

- **입력** = 조위, 풍향/풍속, 해류(u,v) → 시간순으로 쌓인 데이터
- **LSTM의 역할** = 시간적 패턴 학습 → "지금 환경 조건이면 며칠 후 쓰레기가 얼마나 오는가"

### 둘을 합치면?

```
                    ┌─────────────┐
  조위/풍속/해류 ──→│    LSTM     │──→ "지금 환경이면 쓰레기가 이렇게 움직여"
  (시간 패턴)       │ (시간 학습)  │         │
                    └─────────────┘         │
                                            ▼  합치기(concat)
                    ┌─────────────┐    ┌──────────┐
  방출점→섬 연결 ──→│    GNN      │──→ │ 예측 레이어 │──→ 섬별 쓰레기 도달량
  (공간 관계)       │ (공간 학습)  │    └──────────┘
                    └─────────────┘
```

**한 줄 요약: LSTM은 "언제"를, GNN은 "어디서 어디로"를 학습해서, 합치면 "언제 어디에 쓰레기가 얼마나 오는가"를 예측한다.**

---

## 2. 우리 데이터에 맞는 현실적 모델 설계

### 현재 데이터 현황 (실제 파일 기준)

| 파일 | 내용 | 행수 | 핵심 컬럼 |
|------|------|------|-----------|
| test_data_1.csv | MOHID 결과 (섬별 입자도달수) | 32행 | 지역명, lat, lon, id_count, source_count |
| 남해_정답지데이터.csv | 실제 쓰레기 관측값 (정답) | 32행 | 지역명, 수량(개), 무게(kg), lat, lon |
| 항만 방출점 | jeonnam_coastal_outlets_mapped_final2 | 68행 | 하천명, 위도, 경도 |
| 해수욕장 방출점 | 남해안_해수욕장_위경도 | 133행 | 해수욕장명, 위도, 경도 |
| 하천 방출점 | release_point_river | 33행 | id, type, lon, lat |
| 어업 방출점 | 남해_어업_방출점2 | 14행 | 해구번호, 중심_위, 중심_경, 척수 |
| 조위 데이터 | namhae_tide_weather_master_1hr | 시간별 | datetime, obs_tide, pred_tide, wind 등 |
| 기상 데이터 | namhae_weather_master_1hr_final2 | 시간별 | time, wind_dir, wind_speed, u/v_wind |
| 유동 데이터 | namhae_water_flow_2025_Q1.txt | 시간별 | time, lat, lon, u_current, v_current |
| 여수 섬 | 여수_섬전체_위경도(335개) | 335행 | 번호, lat, lon 등 |

### 핵심 도전 과제

**문제: 정답지(관측값)가 32개 섬밖에 없다.**

32개로 GNN+LSTM을 학습시키면 과적합(overfitting) 위험이 매우 높다. 이걸 해결하기 위한 전략이 필요하다.

### 해결 전략: 2단계 모델

```
[1단계] MOHID 궤적 기반 → GNN으로 "방출점→섬" 연결 강도 학습
        (MOHID가 이미 물리 시뮬레이션을 했으므로, 이 결과를 그래프 구조로 변환)

[2단계] 환경변수 시계열 → LSTM으로 "시간에 따른 쓰레기 도달 패턴" 학습
        (조위/풍속/해류가 바뀌면 도달량이 어떻게 변하는지)

[결합] GNN 출력(공간 임베딩) + LSTM 출력(시간 임베딩) → 최종 예측
```

### 데이터 증강 전략 (32개 → 더 많은 학습 데이터)

1. **시간 윈도우 분할**: 환경변수를 월별/계절별로 나누면 32개 × 6분기 = 192개 학습 샘플
2. **MOHID 다중 시나리오**: MOHID를 여러 조건에서 돌린 결과가 있다면 활용
3. **Leave-One-Out 교차검증**: 32개 중 1개를 빼고 31개로 학습 → 32번 반복
4. **물리 기반 제약 추가**: 순수 데이터 학습이 아닌, 해류 방향/거리를 물리 제약으로 넣어 과적합 방지

---

## 3. 4단계: 모델 구축

### 전체 파이프라인

```
Step 4-1: 데이터 전처리
  ├── MOHID 결과 + 정답지 병합
  ├── 방출점 4종 통합 및 타입 라벨링
  ├── 환경변수(조위/풍속/유동) 시간 정렬 및 보간
  └── 결측치 처리

Step 4-2: 그래프 구성
  ├── 노드 정의 (방출점 248개 + 섬 32개 = 280개)
  ├── 엣지 생성 (거리 기반 + MOHID 유동 방향 기반)
  ├── 노드 피처 구성
  └── 엣지 피처 구성

Step 4-3: 시계열 데이터 준비
  ├── 환경변수 병합 (조위 + 풍속 + 해류)
  ├── 섬 근처 관측소 매핑
  ├── 슬라이딩 윈도우 생성
  └── LSTM 입력 텐서 구성

Step 4-4: GNN+LSTM 모델 정의 및 학습
  ├── LSTM 인코더 정의
  ├── GAT(Graph Attention Network) 레이어 정의
  ├── 통합 모델 정의
  ├── 학습 루프 (Leave-One-Out CV)
  └── 모델 저장
```

### 그래프 구성 상세

```
노드 피처 (각 노드가 가지는 속성값):

방출점 노드:
  - [위도, 경도, 방출유형(원핫: 항만/해변/하천/양식장), 방출 강도]
    * 항만: [lat, lon, 1, 0, 0, 0, estimated_volume]
    * 해변: [lat, lon, 0, 1, 0, 0, estimated_volume]
    * 하천: [lat, lon, 0, 0, 1, 0, estimated_volume]
    * 양식장: [lat, lon, 0, 0, 0, 1, 척수(어업활동량)]

섬 노드:
  - [위도, 경도, 0, 0, 0, 0, 0]  (방출 안 하므로 0)
  - 학습 시: 정답값 = 수량(개)

엣지 구성 규칙:
  1. 거리 기반: 방출점↔섬 거리 < 50km면 엣지 생성
  2. 유동 기반: MOHID 유동장에서 방출점→섬 방향으로 해류가 흐르면 가중치 UP
  3. 엣지 피처: [거리(km), 해류u성분, 해류v성분, 유속크기]
```

### LSTM 입력 구성

```
환경변수 시계열 (1시간 간격):
  - 조위: obs_tide, pred_tide, tide_anomaly
  - 풍향/풍속: wind_dir_deg, wind_speed_ms, u_wind, v_wind
  - 해류: u_current_ms, v_current_ms

  ↓ 병합 후 총 9개 변수

슬라이딩 윈도우:
  - 윈도우 크기: 168시간 (7일) → 과거 7일 환경이 쓰레기 도달에 영향
  - 스텝: 24시간 (1일 단위로 이동)
  - 입력 텐서 shape: [배치, 168, 9]
  - 출력: 시간 임베딩 벡터 [배치, 64]
```

---

## 4. 5단계: 검증

```
검증 방법: Leave-One-Out Cross Validation (LOOCV)

  32개 섬 중 1개를 테스트용으로 빼고 31개로 학습
  → 이걸 32번 반복
  → 32개 섬 모두에 대해 "학습에 안 쓴 상태에서의 예측값" 확보

평가 지표:
  - MAE (평균절대오차): 예측이 평균적으로 몇 개 빗나갔는지
  - RMSE (제곱근평균오차): 큰 오차에 민감한 지표
  - R² (결정계수): 전체 변동 중 모델이 설명하는 비율 (1에 가까울수록 좋음)
  - Pearson 상관계수: 예측 순서가 실제와 얼마나 일치하는지

시각화:
  - 예측 vs 실제 scatter plot (대각선에 가까울수록 좋음)
  - 섬별 오차 지도 (Folium)
  - 방출유형별 기여도 분석 (어떤 유형이 가장 많이 기여하는지)
```

---

## 5. 6단계: 여수 335개 섬 예측

```
핵심: 학습된 모델을 그대로 사용하되, 섬 노드만 교체

[기존 모델]
  방출점 248개 → 남해 섬 32개 (학습 완료)

[예측 모델]
  방출점 248개 → 여수 섬 335개 (새 노드)

절차:
  1. 여수_섬전체_위경도(335개).csv 로드
  2. 335개 섬을 새 노드로 추가
  3. 기존 방출점↔여수섬 사이 엣지를 같은 규칙(거리+해류)으로 생성
  4. 저장된 모델(model_namhae.pt)로 예측 실행
  5. 정답지 4개(여수반월, 여수백야도, 여수거문도, 여수안도)와 비교

비교 가능한 정답지 4개:
  | 섬 이름 | 실제 수량(개) |
  |---------|-------------|
  | 여수반월 | 246 |
  | 여수백야도 | 525 |
  | 여수거문도 | 4,207 |
  | 여수안도 | 11,420 |
```

---

## 6. 7단계: 최종 시각화

```
산출 지표:
  1. 예측 쓰레기 도달량 (섬별, 월별)
  2. 위험도 지수 = 예측도달량 정규화 × 접근성 가중치
  3. 주요 기여 방출점 Top3 (GAT attention weight 기반)
  4. 수거 우선순위 랭킹
  5. 계절별 위험도 변화

시각화 산출물:
  - yeosu_marine_debris_map.html (Folium 인터랙티브 지도)
  - risk_ranking_top20.png (위험도 상위 20개 섬 바 차트)
  - seasonal_risk_heatmap.png (계절×섬 히트맵)
  - source_contribution.png (방출유형별 기여도 파이차트)
  - flow_animation.html (해류 + 쓰레기 이동 애니메이션)
```

---

## 7. 단계별 코딩 프롬프트

> 아래 프롬프트를 Claude에게 순서대로 넘기면 됩니다.
> 각 프롬프트 실행 전에 이전 단계 결과물이 있는지 확인하세요.

---

### 프롬프트 1: 데이터 전처리 및 통합

```
해양쓰레기 예측 프로젝트의 데이터 전처리를 해줘.

[프로젝트 위치]
/sessions/laughing-wizardly-euler/mnt/yeosu/

[입력 파일들 - 모두 final_data/Mohid_prepare_data/ 안에 있음]
1. 방출점 데이터 4종:
   - jeonnam_coastal_outlets_mapped_final2.csv (항만, 68행)
     컬럼: 하천명, 본류, ..., 위도, 경도, ...
   - 남해안_해수욕장_위경도.csv (해수욕장, 133행)
     컬럼: 연도, 시도, 시군구, 해수욕장명, 위도, 경도
   - release_point_river.csv (하천, 33행)
     컬럼: id, type, lon, lat
   - 남해_어업_방출점2.csv (어업, 14행)
     컬럼: 해구번호, 중심_위, 중심_경, 척수

2. 환경변수 시계열:
   - namhae_tide_weather_master_1hr.csv (조위+기상)
     컬럼: datetime, obs_name, latitude, longitude, obs_tide, pred_tide,
            tide_anomaly, anomaly_flag, wind_dir_deg, wind_speed_ms, u_wind, v_wind
   - namhae_weather_master_1hr_final2.csv (풍향풍속)
     컬럼: time, station_id, lat, lon, wind_dir_deg, wind_speed_ms, u_wind, v_wind, anomaly_flag
   - namhae_water_flow_2025_Q1.txt~namhae_water_flow_2026_Q2 (해류유동)
     컬럼: time, lat, lon, u_current_ms, v_current_ms

3. 정답 데이터:
   - final_data/남해_정답지데이터.csv (32행)
     컬럼: 지역명, 수량(개), 무게(kg), Latitude, Longitude
   - final_data/test_data_1.csv (MOHID 결과, 32행)
     컬럼: 지역명, Latitude, Longitude, id_count, source_count

[해야 할 것]
1. 방출점 4종을 하나의 DataFrame으로 통합
   - 공통 컬럼: source_id, source_type(harbor/beach/river/fishery), lat, lon, intensity
   - intensity: 어업은 '척수' 값 사용, 나머지는 1.0
   - 저장: final_data/all_release_points.csv

2. 환경변수 3개 파일을 datetime 기준으로 병합
   - 조위 데이터의 관측소별 데이터를 평균 또는 대표값으로 집계
   - 해류 유동 데이터는 격자별이므로 남해 영역 평균으로 집계
   - 최종: datetime, tide, wind_speed, wind_dir, u_wind, v_wind, u_current, v_current
   - 결측치는 선형보간
   - 저장: final_data/env_timeseries_merged.csv

3. MOHID 결과(test_data_1)와 정답지 병합
   - 지역명 기준 inner join
   - 저장: final_data/island_truth_mohid.csv

4. 각 파일의 기본 통계와 결측치 현황을 출력해줘
```

---

### 프롬프트 2: GNN 그래프 구성

```
전처리된 데이터로 PyTorch Geometric 그래프를 구성해줘.

[입력]
- final_data/all_release_points.csv (방출점 통합)
- final_data/island_truth_mohid.csv (섬 + 정답 + MOHID)
- final_data/env_timeseries_merged.csv (환경변수)
- final_data/Mohid_prepare_data/namhae_water_flow_2025_Q1.txt (해류장)

[그래프 설계]
1. 노드 구성:
   - 방출점 노드 (~248개): 피처 = [lat, lon, type_harbor, type_beach, type_river, type_fishery, intensity]
   - 섬 노드 (32개): 피처 = [lat, lon, 0, 0, 0, 0, 0]
   - 총 ~280개 노드

2. 엣지 구성 (방출점 → 섬 방향):
   - 조건1: 두 노드 간 거리 < 80km (haversine 거리)
   - 조건2: 해류 유동장에서 방출점→섬 방향의 성분이 양수 (해류가 그 방향으로 흐름)
   - 엣지 피처: [거리(km), 평균_u_current, 평균_v_current, 유속크기]
   - 조건1만 충족해도 엣지 생성하되, 조건2 충족 시 가중치 부스트

3. 타깃(Y):
   - 섬 노드의 수량(개) 값 (정답지)
   - 방출점 노드는 타깃 없음 (mask 처리)

4. PyTorch Geometric Data 객체로 저장:
   - data.x (노드 피처), data.edge_index, data.edge_attr, data.y, data.train_mask
   - 저장: final_data/graph_data.pt

5. Folium으로 그래프 시각화:
   - 방출점은 유형별 색상 (항만=파랑, 해변=노랑, 하천=초록, 어업=빨강)
   - 섬은 검정 마커
   - 엣지는 반투명 선으로 표시
   - 저장: output/graph_visualization.html

pip install torch torch-geometric 이 필요하면 먼저 설치해줘.
```

---

### 프롬프트 3: LSTM 시계열 인코더

```
환경변수 시계열을 인코딩하는 LSTM 모듈을 만들어줘.

[입력]
- final_data/env_timeseries_merged.csv
  컬럼: datetime, tide, wind_speed, wind_dir, u_wind, v_wind, u_current, v_current

[해야 할 것]
1. 데이터를 MinMaxScaler로 정규화 (0~1)

2. 슬라이딩 윈도우 데이터셋 생성:
   - 윈도우 크기: 168시간 (7일)
   - 스텝: 24시간
   - 입력 shape: [batch, 168, 7] (7개 환경변수)
   - 각 윈도우의 라벨: 해당 기간의 중심 시점

3. LSTM 인코더 클래스 (PyTorch):
   - input_dim=7, hidden_dim=64, num_layers=2, dropout=0.2
   - 출력: 마지막 hidden state → Linear(64, 32)
   - 출력 shape: [batch, 32] (시간 임베딩 벡터)

4. 테스트: 더미 데이터로 forward pass 확인

저장:
- src/model/lstm_encoder.py
- final_data/env_sequences.pt (전처리된 시퀀스 텐서)
```

---

### 프롬프트 4: GNN+LSTM 통합 모델 학습

```
LSTM 인코더와 GAT를 결합한 해양쓰레기 예측 모델을 만들고 학습시켜줘.

[입력]
- final_data/graph_data.pt (그래프 데이터)
- final_data/env_sequences.pt (LSTM 입력 시퀀스)
- src/model/lstm_encoder.py (LSTM 인코더)

[모델 구조]
class MarineDebrisGNN(nn.Module):
    1. LSTM 인코더: env_sequences → time_embedding [batch, 32]
    2. time_embedding을 모든 노드 피처에 concat → [N, 7+32=39]
    3. GAT Layer 1: GATConv(39, 64, heads=4) → [N, 256]
    4. GAT Layer 2: GATConv(256, 64, heads=1) → [N, 64]
    5. 출력 레이어: Linear(64, 1) → 섬별 쓰레기 도달량

[학습 전략]
- 데이터가 32개뿐이므로 Leave-One-Out Cross Validation (LOOCV):
  * 32번 반복, 매번 1개 섬을 빼고 31개로 학습
  * 빠진 1개 섬의 예측값을 기록
  * 32개 예측값 모아서 최종 성능 평가

- 손실함수: MSE Loss
- 옵티마이저: Adam (lr=1e-3, weight_decay=1e-4)
- 에폭: 300 (Early Stopping patience=30)
- 물리 제약 정규화: loss += lambda * distance_penalty
  (멀리 있는 방출점의 영향이 과도하게 높으면 페널티)

[출력]
- 학습된 모델: output/model_namhae.pt
- 학습 곡선: output/training_loss.png
- LOOCV 결과: output/loocv_results.csv
- 예측 vs 실제 scatter plot: output/pred_vs_actual.png
- 평가지표 출력: MAE, RMSE, R², Pearson r
- GAT attention weights 저장 (6단계에서 기여도 분석에 사용)
```

---

### 프롬프트 5: 여수 335개 섬 예측

```
학습된 모델로 여수 335개 섬의 해양쓰레기 도달량을 예측해줘.

[입력]
- output/model_namhae.pt (학습된 모델)
- final_data/Mohid_prepare_data/여수_섬전체_위경도(335개).csv
- final_data/all_release_points.csv (방출점)
- final_data/env_sequences.pt (환경변수)

[절차]
1. 여수 335개 섬을 새 노드로 추가 (방출점은 동일하게 유지)
2. 방출점↔여수섬 엣지를 같은 규칙으로 생성 (거리<80km + 해류방향)
3. 새 그래프에 학습된 모델 적용하여 예측
4. 정답지 4개 섬과 비교:
   - 여수반월(246개), 여수백야도(525개), 여수거문도(4207개), 여수안도(11420개)

[산출 지표]
1. predicted_count: 예측 쓰레기 도달량
2. risk_index: 정규화된 위험도 (0~100)
3. top3_sources: GAT attention 기반 주요 기여 방출점 3개
4. risk_grade: 상(>70) / 중(30~70) / 하(<30)

[출력]
- final_data/yeosu_335_prediction.csv
  컬럼: island_id, island_name, lat, lon, predicted_count, risk_index,
        risk_grade, top3_source_1, top3_source_2, top3_source_3
- output/yeosu_validation_4islands.png (정답지 4개 비교)
```

---

### 프롬프트 6: 최종 시각화 및 대시보드

```
여수 335개 섬 예측 결과를 시각화하고 인터랙티브 대시보드를 만들어줘.

[입력]
- final_data/yeosu_335_prediction.csv

[시각화 목록]

1. Folium 인터랙티브 지도 (yeosu_risk_map.html):
   - 섬을 원형 마커로 표시
   - 마커 크기 = 예측 쓰레기량에 비례
   - 마커 색상 = 위험도 등급 (상=빨강, 중=주황, 하=초록)
   - 팝업: 섬 이름, 예측량, 위험도, 주요 기여 방출점 Top3
   - 방출점도 작은 마커로 함께 표시 (유형별 색상)
   - 상위 10개 섬의 방출점→섬 연결선 표시

2. 위험도 상위 20개 섬 (yeosu_top20_risk.png):
   - 가로 막대그래프
   - 색상: 위험도 등급별

3. 방출유형별 기여도 (source_contribution.png):
   - 전체 기여도 파이차트
   - 위험도 상위 10개 섬의 방출유형별 기여도 stacked bar

4. 계절별 위험도 변화 (seasonal_risk.png):
   - 분기별 × 섬 히트맵 (상위 20개 섬)

5. 정답지 4개 검증 결과 (validation_summary.png):
   - 실제 vs 예측 비교 바 차트
   - 오차율 표시

모든 이미지는 output/ 폴더에 저장해줘.
한글 폰트는 나눔고딕 또는 시스템 한글 폰트를 사용해줘.
```

---

## 부록: 기존 코드와의 관계

현재 프로젝트에 이미 있는 코드:

| 파일 | 내용 | 활용 방안 |
|------|------|-----------|
| src/pipeline/step1_data_pipeline.py | 격자 코드 생성, 지도 시각화 | GridCodeGenerator 클래스 재사용 가능 |
| src/pipeline/step2_graph_builder.py | MOHID 로더, 그래프 빌더 | MOHIDDataLoader 구조 참고, load_real() 확장 |
| src/analysis/train_ai.py | 단순 NN 학습 | GNN+LSTM으로 대체 (기본 구조 참고) |

기존 step2_graph_builder.py의 MOHIDDataLoader.load_real() 메서드가
실제 MOHID 데이터를 받을 수 있도록 설계되어 있으므로,
팀원에게서 MOHID 결과를 받으면 이 인터페이스에 맞춰 넣으면 됨.

---

## 요약 타임라인

```
[4단계] 모델 구축 ─────────────────────────────── 약 1~2주
  4-1. 데이터 전처리 (프롬프트 1)                    1일
  4-2. 그래프 구성 (프롬프트 2)                      1일
  4-3. LSTM 인코더 (프롬프트 3)                      1일
  4-4. 통합 모델 학습 (프롬프트 4)                   3~5일

[5단계] 검증 ──────────────────────────────────── 약 3일
  LOOCV 결과 분석 + 대시보드                        (프롬프트 4에 포함)

[6단계] 여수 예측 ─────────────────────────────── 약 2일
  335개 섬 예측 (프롬프트 5)                         1일

[7단계] 시각화 ────────────────────────────────── 약 3일
  대시보드 + 보고서 (프롬프트 6)                     2일
```
