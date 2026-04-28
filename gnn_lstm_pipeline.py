"""
남해 해양쓰레기 유입량 예측 - GNN+LSTM 파이프라인
=================================================
연구 흐름:
  3단계 완료: MOHID 결과물 → 남해_전체섬_버퍼_1km_입자집계량.csv
  4단계: GNN+LSTM 모델 구축 및 학습 (정답지 보유 섬 기준)
  5단계: 검증 후 모델 저장 → 여수 335개 섬 예측
  6단계: 시각화 + 위험도 지수 산출

MOHID 파일 특이사항:
  - 2618개 섬 수록, 625개만 입자 도달 기록 (나머지 NaN → 0 처리)
  - 정답지 지역명("여수거문도")과 MOHID 섬이름("거문도") 형식 상이
    → 위경도 기반 최근접 매칭으로 해결
  - 정답지 일부(해남묵동리, 마산봉암 등)는 육상 관측소 → 섬 매칭 제외

필요 라이브러리:
  pip install torch torch_geometric pandas numpy matplotlib scikit-learn
  (torch_geometric 설치: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# torch_geometric 있으면 사용, 없으면 직접 구현한 GAT 사용
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
    USE_PYG = True
    print("✅ torch_geometric 사용")
except ImportError:
    USE_PYG = False
    print("⚠️  torch_geometric 없음 → 수동 구현 GAT 사용")


# ============================================================
# 0. 파일 경로 설정 (여기만 수정)
# ============================================================

# [완료된 데이터]
PATH_LABEL       = "남해_관측소_정답지데이터.csv"
PATH_MOHID_FULL  = "남해_전체섬_버퍼_1km_입자집계량.csv"   # ✅ 남해 전체 MOHID 결과

# [추후 추가될 데이터 - 파일 도착 시 경로 입력]
PATH_TIDE        = "namhae_tide_weather_master_1hr.csv"
PATH_WIND        = "namhae_weather_master_1hr_final2.csv"
PATH_FLOW        = "namhae_water_flow_2025_Q1.txt"
PATH_YEOSU       = "여수_섬전체_위경도(335개).csv"          # 여수 335개 섬

# 위경도 매칭 허용 거리 (단위: 도, 약 5km)
MATCH_THRESHOLD  = 0.05

# ★ 수동 보정: 위경도 매칭이 틀린 경우 직접 지정
# 형식: '정답지_지역명': '정확한_MOHID_섬이름'
MANUAL_MATCH = {
    '여수안도': '안도',      # 소부도로 오매칭 → 안도(34.485, 127.811)로 보정
    # '여수반월': None,      # MOHID에 대응 섬 없음 → None이면 제외
    # '신안임자도': '임자도', # 필요 시 추가
}

# 학습 타겟
TARGET  = '수량(개)'   # '무게(kg)'으로 변경 가능
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

# MOHID feature 컬럼 (새 파일 기준)
MOHID_FEATURES = ['source_count', 'velocity_0_count', 'velocity_1_count',
                   'velocity_2_count', 'state_count', 'age_count',
                   'Points_0_count', 'Points_1_count', 'Points_2_count']


# ============================================================
# 4단계 - [전처리 1] 정답지 + MOHID 위경도 기반 매칭
# ============================================================

def load_and_merge(path_label, path_mohid, threshold=MATCH_THRESHOLD):
    """
    정답지(실제 관측값) + MOHID 결과를 위경도 최근접 매칭으로 병합.
    - 정답지 지역명("여수거문도") ≠ MOHID 섬이름("거문도") → 이름 매칭 불가
    - 위경도 기준 가장 가까운 MOHID 섬을 매칭, threshold 이내만 허용
    - NaN source_count(입자 미도달) → 0으로 채움
    Returns: 병합된 DataFrame
    """
    df_label = pd.read_csv(path_label)
    df_mohid = pd.read_csv(path_mohid)

    # MOHID feature NaN → 0
    for col in MOHID_FEATURES:
        if col in df_mohid.columns:
            df_mohid[col] = df_mohid[col].fillna(0)

    mohid_coords = df_mohid[['위도', '경도']].values
    matched_rows = []
    skipped = []

    for _, row in df_label.iterrows():
        지역명 = row['지역명']

        # 수동 보정 우선 적용
        if 지역명 in MANUAL_MATCH:
            target_name = MANUAL_MATCH[지역명]
            if target_name is None:
                skipped.append(f"{지역명} (수동 제외)")
                continue
            manual_match = df_mohid[df_mohid['섬이름'] == target_name]
            if len(manual_match) == 0:
                skipped.append(f"{지역명} (수동지정 '{target_name}' MOHID에 없음)")
                continue
            # 수동 지정 섬 중 가장 가까운 것 선택
            sub_coords = manual_match[['위도', '경도']].values
            dists_sub  = np.sqrt((sub_coords[:, 0] - row['Latitude'])**2 +
                                 (sub_coords[:, 1] - row['Longitude'])**2)
            mohid_row  = manual_match.iloc[dists_sub.argmin()]
            best_dist  = dists_sub.min()
        else:
            dists     = np.sqrt((mohid_coords[:, 0] - row['Latitude'])**2 +
                                (mohid_coords[:, 1] - row['Longitude'])**2)
            best_idx  = dists.argmin()
            best_dist = dists[best_idx]

            if best_dist > threshold:
                skipped.append(f"{지역명} (최근접거리 {best_dist:.4f}° 초과)")
                continue
            mohid_row = df_mohid.iloc[best_idx]
        merged = {
            '지역명':    지역명,
            '섬이름':    mohid_row['섬이름'],
            'Latitude':  row['Latitude'],
            'Longitude': row['Longitude'],
            TARGET:      row[TARGET],
            '무게(kg)':  row['무게(kg)'],
            '매칭거리':  round(best_dist, 5),
        }
        for col in MOHID_FEATURES:
            if col in mohid_row.index:
                merged[col] = mohid_row[col]
        matched_rows.append(merged)

    df = pd.DataFrame(matched_rows)

    print(f"[전처리] 정답지 {len(df_label)}개 중 매칭 성공: {len(df)}개")
    if skipped:
        print(f"  매칭 제외 (육상/거리초과): {len(skipped)}개")
        for s in skipped:
            print(f"    ✗ {s}")
    print(f"\n  매칭된 섬 목록:")
    for _, r in df.iterrows():
        print(f"    {r['지역명']:12s} → {r['섬이름']:12s} "
              f"(source_count={r['source_count']:.0f}, 거리={r['매칭거리']}°)")
    return df


# ============================================================
# 4단계 - [전처리 2] 시계열 데이터 로드 (환경변수)
# ============================================================

def load_time_series(path_tide, path_wind, path_flow, window_size=24):
    """
    조위/풍속/유동 시계열 데이터를 슬라이딩 윈도우로 LSTM 입력 시퀀스 생성.
    파일이 아직 없으면 더미 데이터로 대체.

    Returns: tensor (N_windows, window_size, n_features)
    """
    files_exist = all(os.path.exists(p) for p in [path_tide, path_wind])

    if not files_exist:
        print("⚠️  시계열 파일 미수령 → 더미 데이터로 대체 (파일 도착 시 이 함수 수정)")
        # 더미: window_size 시간, 3개 feature (조위, 풍속, 유동)
        dummy = torch.zeros(1, window_size, 3)
        return dummy, 3

    # ── 실제 데이터 로드 ──────────────────────────────────
    df_tide = pd.read_csv(path_tide, parse_dates=True)
    df_wind = pd.read_csv(path_wind, parse_dates=True)

    # ★ 컬럼명은 실제 파일에 맞게 수정하세요
    # 예: df_tide['tide_level'], df_wind['wind_speed'], df_wind['wind_dir']
    tide_col  = df_tide.columns[1]   # 두 번째 컬럼이 조위값으로 가정
    wspd_col  = df_wind.columns[1]
    wdir_col  = df_wind.columns[2]

    # 유동 데이터 (txt 형식 - 필요 시 파싱 방식 수정)
    try:
        df_flow = pd.read_csv(path_flow, sep='\s+', header=None)
        flow_vals = df_flow.iloc[:, 0].values
    except Exception:
        flow_vals = np.zeros(len(df_tide))

    min_len = min(len(df_tide), len(df_wind), len(flow_vals))
    data = np.stack([
        df_tide[tide_col].values[:min_len].astype(float),
        df_wind[wspd_col].values[:min_len].astype(float),
        flow_vals[:min_len].astype(float)
    ], axis=1)  # (T, 3)

    # 슬라이딩 윈도우
    scaler = StandardScaler()
    data   = scaler.fit_transform(data)
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    seq_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    print(f"[전처리] 시계열 윈도우 수: {len(seq_tensor)}, 형태: {seq_tensor.shape}")
    return seq_tensor, data.shape[1]


# ============================================================
# 4단계 - [전처리 3] 공간 그래프 구성
# ============================================================

def build_graph(df, k=5):
    """
    섬 위경도 기반 k-NN 그래프 구성.
    노드 피처: [Latitude, Longitude] + MOHID_FEATURES 전체
    엣지: 가장 가까운 k개 이웃 연결

    Returns: (node_features tensor, edge_index tensor, scaler)
    """
    coords = df[['Latitude', 'Longitude']].values

    # 사용 가능한 MOHID feature만 선택
    avail_mohid = [c for c in MOHID_FEATURES if c in df.columns]
    mohid_vals  = df[avail_mohid].fillna(0).values

    # 노드 피처 정규화
    scaler    = StandardScaler()
    node_feat = scaler.fit_transform(
        np.hstack([coords, mohid_vals])
    )  # (N, 2 + len(avail_mohid))

    # k-NN 엣지 구성 (유클리드 거리 기반)
    n = len(df)
    src_list, dst_list = [], []
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            src_list.append(i)
            dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    node_feat  = torch.tensor(node_feat, dtype=torch.float32)

    print(f"[그래프] 노드: {n}개, 엣지: {edge_index.shape[1]}개 (k={k})")
    print(f"  노드 피처 차원: {node_feat.shape[1]} "
          f"(위경도 2 + MOHID {len(avail_mohid)}개)")
    return node_feat, edge_index, scaler, len(avail_mohid)


# ============================================================
# 4단계 - [모델] LSTM 인코더
# ============================================================

class LSTMEncoder(nn.Module):
    """
    시계열 환경변수(조위, 풍속, 유동) → 시간 임베딩 벡터
    """
    def __init__(self, input_size=3, hidden_size=32, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h, _) = self.lstm(x)
        return h[-1]   # (batch, hidden_size) — 마지막 레이어 hidden


# ============================================================
# 4단계 - [모델] 수동 구현 GAT (torch_geometric 없을 때)
# ============================================================

class ManualGATConv(nn.Module):
    """
    단순 Graph Attention Layer (torch_geometric 미설치 환경용)
    """
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.heads      = heads
        self.out_per_h  = out_channels // heads
        self.W  = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Linear(2 * out_channels, heads, bias=False)

    def forward(self, x, edge_index):
        n    = x.size(0)
        Wx   = self.W(x)   # (N, out)
        src, dst = edge_index[0], edge_index[1]

        # 어텐션 가중치 계산
        pair = torch.cat([Wx[src], Wx[dst]], dim=-1)   # (E, 2*out)
        e    = self.att(pair)                           # (E, heads)
        e    = F.leaky_relu(e, 0.2)

        # Softmax per node (scatter)
        alpha = torch.zeros(n, e.size(1), device=x.device)
        for h in range(self.heads):
            a_h = e[:, h]
            exp_a = torch.exp(a_h - a_h.max())
            denom = torch.zeros(n, device=x.device).scatter_add(0, dst, exp_a) + 1e-9
            alpha[:, h] = denom

        # 집계
        out = torch.zeros_like(Wx)
        for h in range(self.heads):
            a_h   = torch.exp(e[:, h] - e[:, h].max())
            a_h   = a_h / (alpha[dst, h] + 1e-9)
            msg   = Wx[src] * a_h.unsqueeze(-1)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return F.elu(out / self.heads)


# ============================================================
# 4단계 - [모델] 전체 GNN+LSTM
# ============================================================

class TrashPredictor(nn.Module):
    """
    LSTM 인코더 → GAT → FC → 섬별 쓰레기 도달량 예측
    """
    def __init__(self, node_feat_dim=3, ts_input_dim=3,
                 lstm_hidden=32, gat_hidden=64, gat_heads=4,
                 fc_hidden=32, dropout=0.3):
        super().__init__()

        # LSTM: 시계열 환경변수 인코딩
        self.lstm_enc = LSTMEncoder(ts_input_dim, lstm_hidden)

        # GAT 입력 = 노드 피처 + LSTM 임베딩
        gat_in = node_feat_dim + lstm_hidden

        if USE_PYG:
            self.gat1 = GATConv(gat_in, gat_hidden,
                                 heads=gat_heads, dropout=dropout)
            self.gat2 = GATConv(gat_hidden * gat_heads, gat_hidden,
                                 heads=1, dropout=dropout)
            gat_out = gat_hidden
        else:
            self.gat1 = ManualGATConv(gat_in, gat_hidden, heads=gat_heads)
            self.gat2 = ManualGATConv(gat_hidden, gat_hidden, heads=gat_heads)
            gat_out = gat_hidden

        # FC: 최종 예측
        self.fc = nn.Sequential(
            nn.Linear(gat_out, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, node_feat, edge_index, ts_seq):
        """
        node_feat: (N, node_feat_dim)
        edge_index: (2, E)
        ts_seq:    (1, seq_len, ts_input_dim) — 전체 노드 공유 시계열
        """
        N = node_feat.size(0)

        # LSTM 임베딩 → 모든 노드에 브로드캐스트
        ts_emb = self.lstm_enc(ts_seq)            # (1, lstm_hidden)
        ts_emb = ts_emb.expand(N, -1)            # (N, lstm_hidden)

        # 노드 피처 + 시간 임베딩 결합
        x = torch.cat([node_feat, ts_emb], dim=-1)   # (N, gat_in)

        # GAT 레이어
        if USE_PYG:
            x = F.elu(self.gat1(x, edge_index))
            x = self.gat2(x, edge_index)
        else:
            x = self.gat1(x, edge_index)
            x = self.gat2(x, edge_index)

        # 예측
        out = self.fc(x).squeeze(-1)   # (N,)
        return out


# ============================================================
# 4단계 - [학습]
# ============================================================

def train_model(df_train, node_feat, edge_index, ts_seq,
                epochs=200, lr=1e-3, k_fold=5):
    """
    GNN+LSTM 모델 학습 (K-Fold 교차검증)
    Returns: 학습된 모델
    """
    print("\n" + "=" * 55)
    print("4단계: GNN+LSTM 모델 학습")
    print("=" * 55)

    y_all  = torch.tensor(df_train[TARGET].values, dtype=torch.float32)
    n      = len(df_train)
    idx    = torch.randperm(n)

    ts_input_dim = ts_seq.shape[-1] if ts_seq.ndim == 3 else 3
    fold_size  = n // k_fold
    all_r2, all_mae = [], []

    for fold in range(k_fold):
        val_idx   = idx[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([idx[:fold * fold_size],
                                idx[(fold + 1) * fold_size:]])

        model     = TrashPredictor(
                        node_feat_dim=node_feat.shape[1],
                        ts_input_dim=ts_input_dim
                    ).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        nf = node_feat.to(DEVICE)
        ei = edge_index.to(DEVICE)
        ts = ts_seq[:1].to(DEVICE) if ts_seq.ndim == 3 else \
             torch.zeros(1, 24, ts_input_dim).to(DEVICE)

        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred  = model(nf, ei, ts)
            loss  = criterion(pred[train_idx],
                              y_all[train_idx].to(DEVICE))
            loss.backward()
            optimizer.step()

            # 검증
            model.eval()
            with torch.no_grad():
                val_pred = model(nf, ei, ts)[val_idx].cpu().numpy()
                val_true = y_all[val_idx].numpy()
                val_loss = mean_absolute_error(val_true, val_pred)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone()
                                 for k, v in model.state_dict().items()}

            if epoch % 50 == 0:
                print(f"  [Fold {fold+1}] Epoch {epoch:3d} "
                      f"| Train Loss: {loss.item():.4f} "
                      f"| Val MAE: {val_loss:.2f}")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_pred = model(nf, ei, ts)[val_idx].cpu().numpy()
        r2  = r2_score(y_all[val_idx].numpy(), final_pred)
        mae = mean_absolute_error(y_all[val_idx].numpy(), final_pred)
        all_r2.append(r2)
        all_mae.append(mae)
        print(f"  → Fold {fold+1} 완료 | R²={r2:.3f} | MAE={mae:.1f}")

    print(f"\n✅ 교차검증 결과 | R²={np.mean(all_r2):.3f}±{np.std(all_r2):.3f}"
          f" | MAE={np.mean(all_mae):.1f}±{np.std(all_mae):.1f}")

    # 전체 데이터로 최종 학습
    final_model = TrashPredictor(
                    node_feat_dim=node_feat.shape[1],
                    ts_input_dim=ts_input_dim
                  ).to(DEVICE)
    opt = Adam(final_model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        final_model.train()
        opt.zero_grad()
        pred = final_model(nf, ei, ts)
        loss = criterion(pred, y_all.to(DEVICE))
        loss.backward()
        opt.step()

    return final_model, ts


# ============================================================
# 5단계 - [검증 + 모델 저장]
# ============================================================

def validate_and_save(model, df, node_feat, edge_index, ts, save_path="model.pt"):
    """
    전체 데이터에 대해 예측 vs 실제 비교 후 모델 저장
    """
    print("\n" + "=" * 55)
    print("5단계: 검증 및 모델 저장")
    print("=" * 55)

    model.eval()
    with torch.no_grad():
        preds = model(node_feat.to(DEVICE),
                      edge_index.to(DEVICE),
                      ts.to(DEVICE)).cpu().numpy()

    y_true = df[TARGET].values
    r2  = r2_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)

    print(f"전체 학습 데이터 성능 | R²={r2:.3f} | MAE={mae:.1f}")
    print(f"  실제 평균: {y_true.mean():.1f}, 예측 평균: {preds.mean():.1f}")

    # 섬별 결과 출력
    result = df[['지역명', 'Latitude', 'Longitude', TARGET]].copy()
    result['예측값'] = preds
    result['오차']   = abs(result[TARGET] - result['예측값'])
    print("\n[섬별 예측 결과]")
    print(result.to_string(index=False))

    # 저장
    torch.save({
        'model_state': model.state_dict(),
        'r2': r2,
        'mae': mae
    }, save_path)
    print(f"\n✅ 모델 저장 완료 → {save_path}")

    return result


# ============================================================
# 5단계 - [여수 335개 섬 예측]
# ============================================================

def predict_yeosu(model, path_yeosu, path_mohid_yeosu,
                   graph_scaler, ts, save_path="여수섬_전체_예측력.csv"):
    """
    저장된 모델로 여수 335개 섬 쓰레기 도달량 예측.

    path_mohid_yeosu: 여수 섬들의 MOHID source_count 파일
                      (남해 전체 MOHID 결과에서 여수 부분 필터링하거나
                       별도 파일로 제공)
    """
    print("\n" + "=" * 55)
    print("5단계: 여수 335개 섬 예측")
    print("=" * 55)

    df_yeosu = pd.read_csv(path_yeosu)
    df_mohid = pd.read_csv(path_mohid_yeosu)

    # ★ 여수 파일 컬럼명에 맞게 수정 (예: 'island_name', 'lat', 'lon')
    # 현재 가정: 컬럼명이 'name'/'지역명', 'Latitude', 'Longitude', 'source_count'
    df = pd.merge(df_yeosu, df_mohid, on='name', how='left')
    df['source_count'] = df['source_count'].fillna(0)

    coords   = df[['Latitude', 'Longitude']].values
    src_cnt  = df['source_count'].values.reshape(-1, 1)
    node_feat = graph_scaler.transform(np.hstack([coords, src_cnt]))
    node_feat = torch.tensor(node_feat, dtype=torch.float32)

    # 그래프 엣지 (k=5 이웃)
    n = len(df)
    src_list, dst_list = [], []
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = np.inf
        for j in np.argsort(dists)[:5]:
            src_list.append(i); dst_list.append(j)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        preds = model(node_feat.to(DEVICE),
                      edge_index.to(DEVICE),
                      ts.to(DEVICE)).cpu().numpy()
    preds = np.maximum(preds, 0)

    df['예측_쓰레기도달량'] = preds
    df = add_risk_index(df)

    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 여수 예측 완료 → {save_path}")
    print(f"   예측량 상위 5개 섬:\n{df.nlargest(5, '예측_쓰레기도달량')[['name','예측_쓰레기도달량','위험도지수']].to_string(index=False)}")

    return df


# ============================================================
# 6단계 - [파생 지표] 위험도 지수
# ============================================================

def add_risk_index(df, pred_col='예측_쓰레기도달량'):
    """
    예측 도달량 기반 위험도 지수 산출 (0~100 정규화)
    추가 가능 지표: 누적위험도, 계절성 지수 등
    """
    v = df[pred_col].values
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        risk = (v - v_min) / (v_max - v_min) * 100
    else:
        risk = np.zeros_like(v)

    df['위험도지수']  = risk.round(1)
    df['위험등급']    = pd.cut(risk,
                               bins=[0, 25, 50, 75, 100],
                               labels=['낮음', '보통', '높음', '매우높음'],
                               include_lowest=True)
    return df


# ============================================================
# 6단계 - [시각화]
# ============================================================

def visualize_results(df_result, df_yeosu_pred=None):
    """
    1) 남해 검증: 실제 vs 예측 산점도
    2) 여수 섬별 위험도 지도 (위경도 기반)
    """
    print("\n" + "=" * 55)
    print("6단계: 시각화")
    print("=" * 55)

    fig, axes = plt.subplots(1, 2 if df_yeosu_pred is not None else 1,
                              figsize=(14, 6))
    if df_yeosu_pred is None:
        axes = [axes]

    # --- 플롯 1: 남해 실제 vs 예측 ---
    ax = axes[0]
    ax.scatter(df_result[TARGET], df_result['예측값'],
               c='steelblue', alpha=0.7, edgecolors='k', linewidths=0.5)
    lims = [min(df_result[TARGET].min(), df_result['예측값'].min()) * 0.9,
            max(df_result[TARGET].max(), df_result['예측값'].max()) * 1.1]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y=x (완벽한 예측)')
    for _, row in df_result.iterrows():
        ax.annotate(row['지역명'], (row[TARGET], row['예측값']),
                    fontsize=7, alpha=0.7)
    r2  = r2_score(df_result[TARGET], df_result['예측값'])
    mae = mean_absolute_error(df_result[TARGET], df_result['예측값'])
    ax.set_xlabel('실제 쓰레기 수량'); ax.set_ylabel('예측값')
    ax.set_title(f'남해 검증 결과\nR²={r2:.3f}, MAE={mae:.1f}')
    ax.legend()

    # --- 플롯 2: 여수 위험도 지도 ---
    if df_yeosu_pred is not None:
        ax2 = axes[1]
        sc = ax2.scatter(df_yeosu_pred['Longitude'],
                          df_yeosu_pred['Latitude'],
                          c=df_yeosu_pred['위험도지수'],
                          cmap='RdYlGn_r', s=40, alpha=0.8,
                          vmin=0, vmax=100, edgecolors='k',
                          linewidths=0.3)
        plt.colorbar(sc, ax=ax2, label='위험도 지수 (0~100)')
        ax2.set_xlabel('경도'); ax2.set_ylabel('위도')
        ax2.set_title('여수 335개 섬 쓰레기 위험도 지도')

    plt.tight_layout()
    plt.savefig('result_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ 시각화 저장 완료 → result_visualization.png")
    plt.show()


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("남해 해양쓰레기 GNN+LSTM 파이프라인 시작")
    print(f"디바이스: {DEVICE}")
    print("=" * 55)

    # ── 4단계: 데이터 전처리 ───────────────────────────────
    df_train = load_and_merge(PATH_LABEL, PATH_MOHID_FULL)

    ts_seq, ts_dim = load_time_series(PATH_TIDE, PATH_WIND, PATH_FLOW)

    node_feat, edge_index, graph_scaler, n_mohid_feat = build_graph(df_train, k=5)

    # ── 4단계: 모델 학습 ───────────────────────────────────
    model, ts = train_model(df_train, node_feat, edge_index, ts_seq,
                             epochs=200, lr=1e-3,
                             k_fold=min(5, len(df_train) // 2))

    # ── 5단계: 검증 + 저장 ────────────────────────────────
    df_result = validate_and_save(model, df_train, node_feat,
                                   edge_index, ts, save_path="gnn_lstm_model.pt")

    # ── 5단계: 여수 예측 ──────────────────────────────────
    # ★ 여수 MOHID 결과 파일 준비 후 아래 주석 해제
    #
    # df_yeosu = predict_yeosu(
    #     model,
    #     path_yeosu=PATH_YEOSU,
    #     path_mohid_yeosu="여수_MOHID_결과.csv",  # 남해 MOHID에서 여수 필터링
    #     graph_scaler=graph_scaler,
    #     ts=ts
    # )

    # ── 6단계: 시각화 ─────────────────────────────────────
    visualize_results(df_result)
    # visualize_results(df_result, df_yeosu)  # 여수 예측 후 주석 해제

    print("\n✅ 파이프라인 완료")
