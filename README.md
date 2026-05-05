# 도동실 (DoDongSil) 🌊
### Marine Debris Prediction & Response System for Yeosu Island Archipelago

> **해양쓰레기 예측 모델 구축 프로젝트**  
> Physics-Informed Grey-box AI for Proactive Marine Debris Management

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-GAT-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![MOHID](https://img.shields.io/badge/MOHID-Lagrangian_Simulation-005F87)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

여수시 다도해(335개 섬)에서 발생하는 해양쓰레기 문제를 **사후 수거 → 선제 예측**으로 전환하는 AI 시스템입니다.  
국내 연안 해양쓰레기 분야 최초로 **MOHID 입자추적 시뮬레이션 + GNN + LSTM**을 결합한 Grey-box 하이브리드 구조를 시도합니다.

```
여수시 해양쓰레기 현존량  3,675 톤   (전남 2위)
연간 예산 투입             30억 원 이상
예측 대상 섬               335 개
방출점(오염원)             127 개소
예측 시간 범위             T+6 ~ T+24 시간
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  해수유동(u/v)  ·  조위  ·  풍속/풍향  ·  MOHID 입자 궤적        │
│  (국립해양조사원 · 기상청 · 조위관측소 · 해양환경공단)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │  preprocess.py
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  island_merged.csv         — 23개 섬 × MOHID + 실측 레이블       │
│  env_timeseries_merged.csv — 시간별 환경변수 (8개 feature)        │
│  release_points_encoded.csv— 127개 방출점 (4-class one-hot)      │
└──────────────────────┬──────────────────────────────────────────┘
                       │  build_graph.py  ·  prepare_lstm.py
                       ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│      GRAPH (PyG Data)        │  │    TIME-SERIES SEQUENCES     │
│  Nodes  : 150 (127 RP + 23 island) │  │  [453, 168, 8]               │
│  Edges  : current-weighted   │  │  sliding window 7days×8feat  │
│  Edge feat: dist/u/v/speed/  │  │  조위·풍속·풍향·u/v_wind·     │
│             source_ratio     │  │  u/v_current                 │
└──────────────┬───────────────┘  └──────────────┬───────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MarineDebrisGNN                              │
│                                                                 │
│  env_seq [1,168,8] ──► LSTM(64h,2L) ──► Linear ──► emb [1,32] │
│                                                      │          │
│  node_feat [N,6]  ──────────────── concat ──► [N,38]│          │
│                                        │             │          │
│                              GATConv(38→64×4heads)  │          │
│                                        │                        │
│                              GATConv(256→32×1head)             │
│                                        │                        │
│  sc_anchor (MOHID 물리 기준) ──────── + residual               │
│                                        │                        │
│                                   pred [N]  쓰레기 유입량       │
└──────────────────────────────┬──────────────────────────────────┘
                               │  LOOCV · 물리 제약 손실
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STREAMLIT DASHBOARD (app.py)                      │
│  Overview · Predictions · Route Optimization · Analysis · Report│
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Highlights

### 1. Grey-box Hybrid Model
순수 블랙박스 AI 대신 **물리 모델(MOHID) + 딥러닝** 결합으로 예측 근거를 확보합니다.

| 항목 | 블랙박스 DL | 두둥실 Grey-box |
|------|------------|----------------|
| 예측 근거 | 없음 | 해류·풍속 기반 물리 경로 |
| 이상 기상 대응 | 불안정 | 물리 제약으로 안정적 |
| 현장 신뢰도 | 낮음 | 높음 (원인 설명 가능) |

### 2. Physics-Constrained Loss Function
MOHID 시뮬레이션 상한선을 초과하는 예측에 패널티를 부여하여 물리적으로 타당한 출력을 유도합니다.

```python
L = MSE(pred, y) + λ × ReLU(pred - SC_MAX)
# SC_MAX = 34,385  (MOHID 시뮬레이션 최대 입자수)
# λ = 0.01
```

### 3. Graph Attention Network with Ocean Currents
섬과 방출점을 **해류 방향 가중 엣지**로 연결하여 쓰레기 이동 경로를 네트워크로 학습합니다.

```python
# 엣지 가중치: 해류 방향 도트곱 × source_count 비율
edge_feat = [distance, u_current, v_current, speed, source_count_ratio]

# GAT Layer 구조
GATConv(38 → 64, heads=4, edge_dim=5)  →  ELU + Dropout(0.3)
GATConv(256 → 32, heads=1, edge_dim=5) →  + MOHID anchor
```

### 4. Transfer Learning for Data-Scarce Environments
여수 실측 데이터는 단 4개 지점 — 전이학습으로 데이터 부족 문제를 해결합니다.

```
[Pre-training]  남해안 32개 관측소  →  일반적 해양쓰레기 유입 패턴 학습
[Fine-tuning]   여수 4개 지점      →  여수 지형·박람회·하천/항만 거리 반영
```

### 5. Leave-One-Out Cross-Validation (LOOCV)
레이블 데이터 23개의 극도로 희소한 환경에서 신뢰성 있는 검증을 수행합니다.

```
23개 섬 중 1개를 테스트로 제외 → 22개로 학습 → 23회 반복
평가 지표: MAE · RMSE · R² · Pearson 상관계수
```

---

## Project Structure

```
yeosu/
├── src/
│   ├── model/
│   │   ├── marine_debris_gnn.py   # MarineDebrisGNN (GAT + LSTM)
│   │   └── lstm_encoder.py        # LSTMEncoder
│   ├── pipeline/
│   │   ├── step1_data_pipeline.py # 원시 데이터 수집·통합
│   │   ├── step2_graph_builder.py # PyG 그래프 구성
│   │   └── step3_priority_index.py# 위험도 지수 산출
│   └── analysis/
│       ├── make_dataset.py        # 학습 데이터셋 빌드
│       └── train_ai.py            # 학습 스크립트
│
├── dashboard/
│   └── app.py                     # Streamlit 5탭 대시보드
│
├── preprocess.py                  # MOHID + 실측 데이터 전처리
├── build_graph.py                 # 그래프 객체 생성 (graph_data.pt)
├── prepare_lstm.py                # 슬라이딩 윈도우 시퀀스 생성
├── train.py                       # LOOCV 학습 메인 스크립트
├── gnn_lstm_pipeline.py           # 통합 파이프라인 (단일 실행)
├── pseudo_label_pipeline.py       # 의사 레이블 생성 파이프라인
│
├── final_data/
│   ├── graph_data.pt              # PyG 그래프 객체
│   ├── env_sequences.pt           # [453, 168, 8] 시계열 텐서
│   ├── island_merged.csv          # 23개 섬 학습 레이블
│   └── release_points_encoded.csv # 127개 방출점 인코딩
│
├── output3/
│   ├── model_namhae.pt            # 학습된 모델 가중치
│   ├── loocv_results.csv          # LOOCV 교차검증 결과
│   ├── attention_weights.csv      # GAT 어텐션 가중치
│   └── fig*.png                   # 시각화 산출물 (12종)
│
└── notebooks/
    ├── weather_collect_final.ipynb # 기상 데이터 수집 파이프라인
    ├── mohid_uv_converter.ipynb    # MOHID u/v 해류 변환
    └── eda_cch.ipynb               # 탐색적 데이터 분석
```

---

## Data Pipeline

| 단계 | 스크립트 | 입력 | 출력 |
|------|---------|------|------|
| 1. 원시 데이터 통합 | `preprocess.py` | MOHID 결과 + 실측 CSV | `island_merged.csv` |
| 2. 그래프 구성 | `build_graph.py` | island/release CSVs + 해류 | `graph_data.pt` |
| 3. 시계열 준비 | `prepare_lstm.py` | 환경변수 마스터 CSV | `env_sequences.pt` |
| 4. 모델 학습 | `train.py` | `.pt` 파일들 | `model_namhae.pt` |
| 5. 대시보드 | `dashboard/app.py` | 모델 + 데이터 | Streamlit UI |

---

## Tech Stack

| 영역 | 기술 |
|------|------|
| **물리 시뮬레이션** | MOHID Water (Lagrangian 입자 추적) |
| **딥러닝 프레임워크** | PyTorch · PyTorch Geometric |
| **모델 구조** | Graph Attention Network (GAT) · LSTM |
| **공간 분석** | QGIS · GeoPandas · Shapely · PyProj |
| **데이터 처리** | Pandas · NumPy · Scikit-learn · SciPy |
| **시각화** | Folium · PyDeck · Plotly · Matplotlib |
| **대시보드** | Streamlit |
| **경로 최적화** | NetworkX |

---

## Completed Milestones

- [x] QGIS 기반 해안선 세그먼트 분할 및 127개 방출점 공간 분석 도출
- [x] MOHID Lagrangian 입자 추적 시뮬레이션 (2023~2025, 12개 시기, 1시간 간격)
- [x] 섬 반경 1km 버퍼 × 입자 궤적 공간 중첩 → 정량적 유입량 산정
- [x] 남해안 32개 관측소 + 물리 모델 결과 통합 마스터 데이터셋 구축
- [x] MarineDebrisGNN (GAT 2층 + LSTM 인코더) 구현 및 LOOCV 검증
- [x] 물리 제약 손실 함수 및 MOHID source_count anchor 적용
- [x] Streamlit 5탭 대시보드 (Overview · Predictions · Route Optimization · Analysis · Report)

## Roadmap

- [ ] 여수 지역 특화 Fine-tuning (박람회 특성·산업단지·하천 거리 반영)
- [ ] 학습 모델 ↔ 대시보드 실시간 연결
- [ ] 자동 리포트 발송 시스템 (매일 아침 RPA 자동화)
- [ ] 청소선 GPS·배차 시스템 API 연동 → 최적 순회 라우팅 완전 자동화
- [ ] 전국 다도해 표준 플랫폼 확장 (신안군·통영시 등)

---

## Getting Started

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# PyTorch Geometric 별도 설치 (버전에 맞게)
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 2. 전체 파이프라인 실행
python preprocess.py
python build_graph.py
python prepare_lstm.py
python train.py

# 3. 대시보드 실행
cd dashboard && streamlit run app.py
```

---

## Background

2026 여수 세계섬박람회 개최를 앞두고, 여수시는 연간 30억 원 이상의 예산을 해양쓰레기 수거에 투입하고 있습니다.  
그러나 현재 대응 방식은 **발생 후 수거**라는 사후 대응에 머물러 있으며, 핫스팟을 사전에 파악하거나 정화 인력을 효율적으로 배치하는 데이터 기반 시스템이 없는 상황입니다.

두둥실은 이 문제를 **해양 물리학 + 그래프 딥러닝**으로 접근합니다.  
특정 섬에 쓰레기가 언제, 얼마나 유입될지를 사전에 예측함으로써 인력과 선박을 효율적으로 배치할 수 있도록 지원합니다.

---

*프로젝트 문의 · 협업 제안은 Issues 또는 이메일로 연락해주세요.*
