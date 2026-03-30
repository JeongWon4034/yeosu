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
[방출점 127개]              [바다(해류)]            [섬들]
beach  78개 ──────────┐
port   27개 ────────┐ │    해류가 연결해줌      ┌── 여수반월
river  13개 ──────┐ │ │  ==================>   ├── 여수백야도
fishery 9개 ────┐ │ │ │                        ├── 완도신지도
                ▼ ▼ ▼ ▼                         ▼
          [GNN이 학습하는 것]
          "port 23번 방출점에서 나온 쓰레기가
           여수안도 섬에 얼마나 갈까?"
```

- **노드(Node)** = 방출점 127개 + 섬(남해 23개)
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
  방출점→섬 연결 ──→│    GNN      │──→ │ 예측 레이어│──→ 섬별 쓰레기 도달량
  (공간 관계)       │ (공간 학습)  │    └──────────┘
                    └─────────────┘
```

**한 줄 요약: LSTM은 "언제"를, GNN은 "어디서 어디로"를 학습해서, 합치면 "언제 어디에 쓰레기가 얼마나 오는가"를 예측한다.**

---

## 2. 우리 데이터에 맞는 현실적 모델 설계

### 현재 데이터 현황 (실제 파일 기준)

| 파일 | 내용 | 행수 | 핵심 컬럼 |
|------|------|------|-----------|
| **namhae_release_points_127_최종.csv** | 방출점 통합 (4종 합본) | 127행 | id, type(beach/port/river/fishery), lon, lat |
| **섬_쓰레기_입자개수_3개월_.csv** | MOHID 결과 (섬별 입자 도달수) | 23행 | ADM_CD, ADM_NM, name, source_count |
| 남해_정답지데이터.csv | 실제 쓰레기 관측값 (정답) | 32행 | 지역명, 수량(개), 무게(kg), Latitude, Longitude |
| namhae_tide_weather_master_1hr.csv | 조위+기상 시계열 | 시간별 | datetime, obs_name, lat, lon, obs_tide, pred_tide, tide_anomaly, wind_dir_deg, wind_speed_ms, u_wind, v_wind |
| namhae_weather_master_1hr_final2.csv | 풍향/풍속 시계열 | 시간별 | time, station_id, lat, lon, wind_dir_deg, wind_speed_ms, u_wind, v_wind, anomaly_flag |
| namhae_water_flow_2025_Q1.txt ~ 2026_Q2.txt | 해류유동 (6개 파일) | 시간별 | time, lat, lon, u_current_ms, v_current_ms |
| 여수_섬전체_위경도(335개).csv | 여수 섬 예측 대상 | 335행 | 번호, 섬이름, lat, lon |

### 방출점 타입별 구성 (namhae_release_points_127_최종.csv)

| 타입 | 개수 | 설명 |
|------|------|------|
| beach (해수욕장) | 78개 | 해수욕장 쓰레기 방출 |
| port (항만) | 27개 | 항만/어항 방출 |
| river (하천) | 13개 | 하천→해안 유입 |
| fishery (어업) | 9개 | 어업활동구역 방출 |
| **합계** | **127개** | |

### MOHID 결과와 정답지 매칭 현황

- **MOHID 결과 섬**: 23개 (`섬_쓰레기_입자개수_3개월_.csv`)
- **정답지 섬**: 32개 (`남해_정답지데이터.csv`)
- **공통 섬(학습 사용)**: 23개 (MOHID 결과 섬 전부가 정답지에 포함됨)
- **주의**: 해남묵동리, 여수반월, 마산봉암은 source_count 결측 → 처리 필요
- **정답지에만 있는 섬(9개)**: 신안우이도, 신안흑산도, 신안고장, 진도가사도, 사천아두도, 울산대왕암, 울주나사리, 울산주전, 부산일광 (이 섬들은 MOHID를 돌리지 않았으므로 검증에서 제외)

### 핵심 도전 과제

**문제: MOHID+정답 공통 섬이 23개뿐이다 (결측 3개 제외하면 사실상 20개).**

20~23개로 GNN+LSTM을 학습시키면 과적합(overfitting) 위험이 매우 높다. 이걸 해결하기 위한 전략이 필요하다.

### 해결 전략: 2단계 모델 + 데이터 증강

```
[1단계] MOHID 입자 수 기반 → GNN으로 "방출점→섬" 공간 연결 강도 학습
        (source_count가 높은 섬 = 입자가 많이 도달한 섬 = 쓰레기가 많이 올 가능성 높음)

[2단계] 환경변수 시계열 → LSTM으로 "시간에 따른 쓰레기 도달 패턴" 학습
        (조류·풍향이 바뀌면 도달량이 어떻게 변하는지)

[결합] GNN 출력(공간 임베딩) + LSTM 출력(시간 임베딩) → 최종 예측값(수량)
```

### 데이터 증강 전략 (20개 → 더 많은 학습 데이터)

1. **시간 윈도우 분할**: 환경변수를 월별/계절별로 나누면 20개 × 여러 기간 = 학습 샘플 증가
2. **MOHID 다중 시나리오**: 현재 3개월치 → 추후 계절별로 추가 시뮬레이션
3. **Leave-One-Out 교차검증(LOOCV)**: 23개 중 1개를 빼고 22개로 학습 → 23번 반복
4. **물리 기반 제약 추가**: 해류 방향/거리를 물리 제약으로 넣어 과적합 방지

---

## 3. 4단계: 모델 구축

### 전체 파이프라인

```
Step 4-1: 데이터 전처리
  ├── MOHID 결과(섬_쓰레기_입자개수_3개월_) + 정답지 병합
  ├── 방출점 통합 파일(namhae_release_points_127_최종) 로드 및 타입 인코딩
  ├── 환경변수(조위/풍속/유동 6개 파일) 시간 정렬 및 병합
  └── source_count 결측치 처리 (해남묵동리, 여수반월, 마산봉암)

Step 4-2: 그래프 구성
  ├── 노드 정의 (방출점 127개 + 섬 23개 = 150개)
  ├── 엣지 생성 (거리 기반 + 해류 방향 기반)
  ├── 노드 피처 구성
  └── 엣지 피처 구성

Step 4-3: 시계열 데이터 준비
  ├── 환경변수 3종 병합 (조위 + 풍속 + 해류 6개 파일)
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

방출점 노드 (127개):
  - [lat, lon, is_beach, is_port, is_river, is_fishery]
    * beach:   [lat, lon, 1, 0, 0, 0]
    * port:    [lat, lon, 0, 1, 0, 0]
    * river:   [lat, lon, 0, 0, 1, 0]
    * fishery: [lat, lon, 0, 0, 0, 1]

섬 노드 (23개):
  - [lat, lon, 0, 0, 0, 0]  (방출 안 하므로 0)
  - 학습 시 타깃값 = 수량(개)  ← 정답지에서 가져옴

엣지 구성 규칙:
  1. 거리 기반: 방출점↔섬 haversine 거리 < 80km면 엣지 생성
  2. 유동 기반: 해류장에서 방출점→섬 방향의 해류 성분이 양수면 가중치 UP
  3. 엣지 피처: [거리(km), 평균_u_current, 평균_v_current, 유속크기]
  4. source_count도 엣지 가중치 초기값으로 활용 가능
```

### LSTM 입력 구성

```
환경변수 시계열 병합 (1시간 간격):
  조위: obs_tide, pred_tide, tide_anomaly  ← namhae_tide_weather_master_1hr.csv
  기상: wind_dir_deg, wind_speed_ms, u_wind, v_wind  ← namhae_weather_master_1hr_final2.csv
  해류: u_current_ms, v_current_ms  ← namhae_water_flow_2025_Q1~2026_Q2 (6개 파일 합본)

  ↓ 병합 후 총 9개 변수 → 관측소별 평균 또는 대표 지점 사용

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

  23개 섬 중 1개를 테스트용으로 빼고 22개로 학습
  → 이걸 23번 반복 (결측 3개 제외하면 20번)
  → 모든 섬에 대해 "학습에 안 쓴 상태에서의 예측값" 확보

평가 지표:
  - MAE (평균절대오차): 예측이 평균적으로 몇 개 빗나갔는지
  - RMSE (제곱근평균오차): 큰 오차에 민감한 지표
  - R² (결정계수): 1에 가까울수록 좋음
  - Pearson 상관계수: 예측 순서가 실제와 얼마나 일치하는지

추가 분석:
  - source_count(MOHID 입자수)와 실제 수량(개)의 상관관계 먼저 확인
    → 상관계수가 낮으면 MOHID 자체 정확도 문제일 수 있음
  - 방출유형별(beach/port/river/fishery) 기여도 분석
    (GAT의 attention weight로 각 방출점 타입의 영향력 정량화)

시각화:
  - 예측 vs 실제 scatter plot (대각선에 가까울수록 좋음)
  - 섬별 오차 지도 (Folium)
  - 방출유형별 기여도 바 차트
```

---

## 5. 6단계: 여수 335개 섬 예측

```
핵심: 학습된 모델을 그대로 사용하되, 섬 노드만 교체

[기존 모델]
  방출점 127개 → 남해 섬 23개 (학습 완료)

[예측 모델]
  방출점 127개 → 여수 섬 335개 (새 노드)

절차:
  1. 여수_섬전체_위경도(335개).csv 로드
  2. 335개 섬을 새 노드로 추가
  3. 기존 방출점↔여수섬 사이 엣지를 같은 규칙(거리+해류)으로 생성
  4. 저장된 모델(model_namhae.pt)로 예측 실행

정답지 4개와의 비교 (여수섬 중 정답지에 있는 섬):
  | 섬 이름    | 실제 수량(개) | MOHID source_count |
  |-----------|------------|-------------------|
  | 여수반월   | 246        | 결측               |
  | 여수백야도  | 525        | 11,645            |
  | 여수거문도  | 4,207      | 1,568             |
  | 여수안도   | 11,420     | 34,385            |

  → 여수거문도가 흥미로운 케이스: 실제 쓰레기(4207개)는 많은데
    MOHID 입자(1568개)는 상대적으로 적음 (입자가 도달은 하지만 다른 변수 영향일 수 있음)
```

---

## 6. 7단계: 최종 시각화

```
산출 지표:
  1. 예측 쓰레기 도달량 (섬별)
  2. 위험도 지수 = 예측도달량 정규화 (0~100)
  3. 주요 기여 방출점 Top3 (GAT attention weight 기반)
     → "이 섬 쓰레기의 60%는 beach 타입 방출점에서 온다" 같은 해석
  4. 수거 우선순위 랭킹
  5. 방출유형별 기여 비율

시각화 산출물:
  - yeosu_marine_debris_map.html (Folium 인터랙티브 지도)
  - risk_ranking_top20.png (위험도 상위 20개 섬 바 차트)
  - source_type_contribution.png (방출유형별 기여도)
  - validation_4islands.png (정답지 4개 섬 검증)
```

---

## 7. 단계별 코딩 프롬프트

> 아래 프롬프트를 Claude에게 순서대로 넘기면 됩니다.
> 각 프롬프트 전에 이전 단계의 출력 파일이 있는지 먼저 확인하세요.

---

### 프롬프트 1: 데이터 전처리 및 통합

```
해양쓰레기 예측 프로젝트의 데이터 전처리를 해줘.

[프로젝트 경로]
/sessions/laughing-wizardly-euler/mnt/yeosu/  (또는 실제 프로젝트 루트)

[입력 파일]

1. 방출점 통합 파일 (final_data/Mohid_prepare_data/ 안):
   - namhae_release_points_127_최종.csv (127개 방출점)
     컬럼: id, type(beach/port/river/fishery), lon, lat

2. MOHID 결과:
   - final_data/섬_쓰레기_입자개수_3개월_.csv (23개 섬)
     컬럼: ADM_CD, ADM_NM, name, source_count
     ※ 결측 있음: 해남묵동리, 여수반월, 마산봉암의 source_count가 NaN

3. 정답 데이터:
   - final_data/남해_정답지데이터.csv (32개 섬)
     컬럼: 지역명, 수량(개), 무게(kg), Latitude, Longitude

4. 환경변수 시계열:
   - final_data/Mohid_prepare_data/namhae_tide_weather_master_1hr.csv (조위+기상)
     컬럼: datetime, obs_name, latitude, longitude, obs_tide, pred_tide,
            tide_anomaly, anomaly_flag, wind_dir_deg, wind_speed_ms, u_wind, v_wind
   - final_data/Mohid_prepare_data/namhae_weather_master_1hr_final2.csv (풍향풍속)
     컬럼: time, station_id, lat, lon, wind_dir_deg, wind_speed_ms, u_wind, v_wind, anomaly_flag
   - final_data/Mohid_prepare_data/namhae_water_flow_2025_Q1.txt
     ~ namhae_water_flow_2026_Q2.txt (해류유동, 총 6개 파일)
     컬럼: time, lat, lon, u_current_ms, v_current_ms

[해야 할 것]

1. MOHID 결과 + 정답지 병합:
   - MOHID의 'name' 컬럼 ↔ 정답지의 '지역명' 기준으로 inner join
   - source_count 결측 3개: 해당 행의 평균값으로 대체 (또는 KNN 보간)
   - 병합 결과: 지역명, source_count, 수량(개), 무게(kg), Latitude, Longitude
   - 저장: final_data/island_merged.csv

2. 환경변수 3종 병합:
   - 조위: 관측소별 값을 시간별로 남해 전체 평균으로 집계
   - 해류: 6개 파일을 시간순으로 합본 후 남해 영역(위도 33~36, 경도 124~130) 평균
   - 기상: 관측소별 평균
   - datetime 기준 합치기, 결측치는 선형보간
   - 최종 컬럼: datetime, tide, tide_pred, wind_speed, wind_dir,
                u_wind, v_wind, u_current, v_current
   - 저장: final_data/env_timeseries_merged.csv

3. 방출점 파일은 그대로 사용 (namhae_release_points_127_최종.csv)
   - type 컬럼을 원핫인코딩으로 변환한 버전 저장:
     final_data/release_points_encoded.csv

4. 각 파일의 기본 통계, 결측치, 분포를 출력해줘
```

---

### 프롬프트 2: GNN 그래프 구성

```
전처리된 데이터로 PyTorch Geometric 그래프를 구성해줘.

[입력]
- final_data/release_points_encoded.csv (방출점 127개, 원핫인코딩 포함)
- final_data/island_merged.csv (섬 23개, source_count + 수량(개))
- final_data/env_timeseries_merged.csv (환경변수)

[그래프 설계]

1. 노드 구성 (총 150개):
   - 방출점 노드 (127개):
     피처 = [lat, lon, is_beach, is_port, is_river, is_fishery]  → 6차원
   - 섬 노드 (23개):
     피처 = [lat, lon, 0, 0, 0, 0]  → 방출점이 아니므로 타입은 0
   - 타깃(Y): 섬 노드의 수량(개) 값, 방출점은 mask=False

2. 엣지 구성 (방출점 → 섬 방향):
   - 조건1: 두 노드 간 haversine 거리 < 80km → 엣지 생성
   - 조건2: 환경변수에서 평균 해류 방향이 방출점→섬 방향이면 가중치 × 1.5
   - 엣지 피처: [거리(km), 평균_u_current, 평균_v_current, 유속크기, source_count_비율]
     ※ source_count_비율 = 해당 방출점→섬 쌍의 상대적 입자 기여도 (있으면 활용)

3. PyTorch Geometric Data 객체로 저장:
   data.x          → [150, 6] 노드 피처
   data.edge_index → [2, E] 엣지 인덱스
   data.edge_attr  → [E, 5] 엣지 피처
   data.y          → [150] 타깃 (섬 노드만 유효값, 방출점은 0)
   data.train_mask → [150] bool (섬 노드만 True)
   저장: final_data/graph_data.pt

4. Folium으로 그래프 시각화 (output/graph_visualization.html):
   - 방출점: 타입별 색상 마커 (beach=노랑, port=파랑, river=초록, fishery=빨강)
   - 섬: 검정 원형 마커, 팝업에 수량(개) 표시
   - 엣지: 반투명 선 (거리 가까울수록 진하게)
   - 범례 추가

pip install torch torch-geometric이 필요하면 먼저 설치해줘.
```

---

### 프롬프트 3: LSTM 시계열 인코더

```
환경변수 시계열을 인코딩하는 LSTM 모듈을 만들어줘.

[입력]
- final_data/env_timeseries_merged.csv
  컬럼: datetime, tide, tide_pred, wind_speed, wind_dir,
        u_wind, v_wind, u_current, v_current  (9개 변수)

[해야 할 것]

1. 데이터 정규화 (MinMaxScaler, 0~1)

2. 슬라이딩 윈도우 데이터셋 생성:
   - 윈도우 크기: 168시간 (7일치)
   - 스텝: 24시간
   - 입력 shape: [batch, 168, 9]

3. LSTM 인코더 클래스 작성 (PyTorch):
   class LSTMEncoder(nn.Module):
     - input_dim=9, hidden_dim=64, num_layers=2, dropout=0.2
     - 출력: 마지막 hidden state → Linear(64, 32) → ReLU
     - 출력 shape: [batch, 32]  (시간 임베딩 벡터)

4. 저장:
   - src/model/lstm_encoder.py  (모듈 파일)
   - final_data/env_sequences.pt (전처리된 시퀀스 텐서 + 스케일러 파라미터)

5. 더미 데이터로 forward pass 동작 확인 후 출력 shape 프린트
```

---

### 프롬프트 4: GNN+LSTM 통합 모델 학습

```
LSTM 인코더와 GAT를 결합한 해양쓰레기 예측 모델을 만들고 학습시켜줘.

[입력]
- final_data/graph_data.pt (그래프 데이터)
- final_data/env_sequences.pt (LSTM 입력)
- src/model/lstm_encoder.py (LSTM 인코더)

[모델 구조]
class MarineDebrisGNN(nn.Module):

  1. LSTMEncoder: env_sequences → time_emb [batch, 32]
  2. time_emb을 모든 노드 피처(6차원)에 concat → [N, 38]
  3. GATConv Layer1: (38, 64, heads=4, concat=True) → [N, 256]
  4. ELU 활성화 + Dropout(0.3)
  5. GATConv Layer2: (256, 32, heads=1, concat=False) → [N, 32]
  6. 출력 레이어: Linear(32, 1) → 섬별 쓰레기 도달량 예측
  7. 손실: train_mask 적용 (섬 노드 23개만 학습에 사용)

[학습 전략]
- Leave-One-Out Cross Validation (LOOCV):
  * 23번 반복, 매번 섬 1개를 test로, 나머지 22개로 학습
  * source_count 결측 3개(해남묵동리, 여수반월, 마산봉암)는 LOOCV에서 제외 (20번)
  * 각 fold의 예측값 기록

- 손실함수: MSELoss
- 옵티마이저: Adam(lr=1e-3, weight_decay=1e-4)
- 에폭: 300, Early Stopping(patience=30)
- 물리 제약 정규화 항 추가:
  loss += 0.01 * torch.mean(torch.relu(pred - source_count_max))
  (MOHID source_count 최댓값보다 과도하게 높은 예측에 페널티)

[출력]
- output/model_namhae.pt  (학습된 모델 저장)
- output/training_loss.png (학습 곡선)
- output/loocv_results.csv (fold별 예측값 vs 실제값)
- output/pred_vs_actual.png (scatter plot)
- 터미널에 출력: MAE, RMSE, R², Pearson r
- output/attention_weights.csv (GAT attention weight, 6단계 기여도 분석용)
```

---

### 프롬프트 5: 여수 335개 섬 예측

```
학습된 모델로 여수 335개 섬의 해양쓰레기 도달량을 예측해줘.

[입력]
- output/model_namhae.pt (학습된 모델)
- output/attention_weights.csv (GAT attention weights)
- final_data/Mohid_prepare_data/여수_섬전체_위경도(335개).csv (여수 섬 목록)
- final_data/release_points_encoded.csv (방출점 127개)
- final_data/env_sequences.pt (환경변수)

[절차]
1. 여수 335개 섬을 새 노드로 추가 (방출점 127개는 동일 유지)
2. 방출점↔여수섬 엣지를 같은 규칙(거리<80km + 해류방향)으로 생성
3. 학습된 모델로 예측 (eval 모드, no_grad)
4. 정답지 4개 섬과 비교:
   여수반월(실제 246개), 여수백야도(525개), 여수거문도(4,207개), 여수안도(11,420개)

[산출 지표]
- predicted_count: 예측 쓰레기 도달량
- risk_index: Min-Max 정규화 후 0~100 스케일
- risk_grade: 상(risk_index > 66) / 중(33~66) / 하(< 33)
- top3_source_type: attention weight 기반 주요 기여 방출 타입 3개
  (예: "beach > port > river")

[출력]
- final_data/yeosu_335_prediction.csv
  컬럼: island_id, island_name, lat, lon, predicted_count,
        risk_index, risk_grade, top_source_type1, top_source_type2, top_source_type3
- output/yeosu_validation_4islands.png (정답지 4개 비교 바 차트)
```

---

### 프롬프트 6: 최종 시각화 및 대시보드

```
여수 335개 섬 예측 결과를 시각화해줘.

[입력]
- final_data/yeosu_335_prediction.csv

[시각화 목록]

1. Folium 인터랙티브 지도 (output/yeosu_risk_map.html):
   - 섬 마커: 원형, 크기 = 예측량에 비례, 색상 = 위험도 등급
     (상=빨강 #e74c3c, 중=주황 #f39c12, 하=초록 #27ae60)
   - 팝업: 섬 이름, 예측량, 위험도 지수, 주요 기여 방출 타입 Top3
   - 방출점 127개도 작은 마커로 함께 표시 (타입별 색상)
   - 위험도 상위 10개 섬의 방출점→섬 연결선 (점선)
   - 타일: OpenStreetMap 또는 CartoDB positron

2. 위험도 상위 20개 섬 바 차트 (output/yeosu_top20_risk.png):
   - 가로 막대그래프, 색상=위험도 등급
   - x축: risk_index, y축: 섬 이름
   - 한글 폰트: NanumGothic 또는 시스템 한글 폰트

3. 방출유형별 기여도 (output/source_contribution.png):
   - 파이차트: 전체 예측량에서 beach/port/river/fishery 기여 비율
   - 위험도 상위 10개 섬의 기여도 stacked bar

4. 정답지 4개 검증 요약 (output/validation_summary.png):
   - 실제 수량 vs AI 예측량 그룹 바 차트
   - 오차율(%) 표시

모든 파일은 output/ 폴더에 저장해줘.
한글 폰트 설정: matplotlib.rcParams['font.family'] = 'NanumGothic'
또는 환경에 맞게 자동 감지해줘.
```

---

## 부록: 기존 코드와의 관계

현재 프로젝트에 이미 있는 코드:

| 파일 | 내용 | 활용 방안 |
|------|------|-----------|
| src/pipeline/step1_data_pipeline.py | 격자 코드 생성, 지도 시각화 | GridCodeGenerator 재사용 가능 |
| src/pipeline/step2_graph_builder.py | MOHID 로더, 그래프 빌더 | MOHIDDataLoader 구조 참고 |
| src/pipeline/step3_priority_index.py | 우선순위 지수 계산 | 7단계 위험도 지수 산출에 활용 |
| src/analysis/train_ai.py | 단순 NN 학습 (현재 임시 데이터 사용) | GNN+LSTM으로 대체 |

---

## 요약 타임라인

```
[4단계] 모델 구축 ─────────────────── 약 1~2주
  4-1. 데이터 전처리 (프롬프트 1)        1일
  4-2. 그래프 구성 (프롬프트 2)          1일
  4-3. LSTM 인코더 (프롬프트 3)          1일
  4-4. 통합 모델 학습 (프롬프트 4)       3~5일

[5단계] 검증 ──────────────────────── 약 2~3일
  LOOCV 결과 분석 (프롬프트 4에 포함)

[6단계] 여수 예측 ─────────────────── 약 1~2일
  335개 섬 예측 (프롬프트 5)

[7단계] 시각화 ────────────────────── 약 2~3일
  대시보드 + 보고서 (프롬프트 6)
```

---

## 중요 체크리스트

- [ ] MOHID source_count 결측값(3개) 처리 방식 팀 내 합의
- [ ] 여수_섬전체_위경도(335개).csv 컬럼명 확인 (한글 깨짐 있음)
- [ ] 환경변수 기간: 유동 데이터는 2025Q1~2026Q2, 조위/기상은 기간 확인 필요
- [ ] 여수거문도 이상치 확인: MOHID 입자수(1,568)에 비해 실제 수거량(4,207)이 훨씬 많음
- [ ] GPU 환경 확인: GAT 학습 시 CUDA 사용 여부
