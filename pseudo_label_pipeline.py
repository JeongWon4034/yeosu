"""
남해 해양 쓰레기 유입량 예측 - Pseudo-Labeling 파이프라인
=====================================================
전략:
  1단계. Teacher Model (얕은 학습): 여수 소수 정답지 + MOHID → RF/XGBoost 학습
  2단계. Pseudo-labeling:          남해 전체 MOHID → Teacher Model로 가짜 정답 생성
  3단계. Deep Learning Pre-training: 대규모 pseudo-label 데이터로 LSTM 사전학습
  4단계. Fine-tuning:              진짜 정답지(소수)로 미세조정

필요 라이브러리:
  pip install scikit-learn xgboost torch pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 0. 파일 경로 설정 (여기만 수정하세요)
# ===========================================================

PATH_LABEL   = "남해_관측소_정답지데이터.csv"      # 소수 정답지 (31개 관측소)
PATH_MOHID_PARTIAL = "남해_섬_MOHID_1분기_결과.csv" # 현재 보유한 MOHID (23개)
PATH_MOHID_FULL    = "남해_전체_MOHID_결과.csv"     # ★ 아직 미수령 — 도착하면 이 경로로 저장

# 학습 타겟 선택: '수량(개)' 또는 '무게(kg)'
TARGET = '수량(개)'

# ===========================================================
# 1단계. Teacher Model — Random Forest / XGBoost 얕은 학습
# ===========================================================

def step1_train_teacher(path_label, path_mohid):
    """
    보유한 정답지 + MOHID를 병합하여 Teacher Model을 학습합니다.
    Returns: 학습된 모델(rf, xgb_model), scaler, feature 컬럼명
    """
    print("=" * 55)
    print("1단계: Teacher Model 학습 (얕은 학습)")
    print("=" * 55)

    df_label = pd.read_csv(path_label)
    df_mohid = pd.read_csv(path_mohid)

    # 지역명 기준 병합 (정답지 ∩ MOHID)
    df = pd.merge(df_label, df_mohid, left_on='지역명', right_on='name', how='inner')
    print(f"학습 샘플 수: {len(df)}개")

    FEATURES = ['Latitude', 'Longitude', 'source_count']
    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                min_samples_leaf=2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_r2 = cross_val_score(rf, X_scaled, y, cv=cv, scoring='r2')
    rf_mae = cross_val_score(rf, X_scaled, y, cv=cv,
                             scoring='neg_mean_absolute_error')
    print(f"\n[Random Forest] R²: {rf_r2.mean():.3f} ± {rf_r2.std():.3f}")
    print(f"               MAE: {-rf_mae.mean():.1f} ± {rf_mae.std():.1f}")
    rf.fit(X_scaled, y)

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, subsample=0.8,
                                   random_state=42, verbosity=0)
    xgb_r2 = cross_val_score(xgb_model, X_scaled, y, cv=cv, scoring='r2')
    xgb_mae = cross_val_score(xgb_model, X_scaled, y, cv=cv,
                               scoring='neg_mean_absolute_error')
    print(f"\n[XGBoost]       R²: {xgb_r2.mean():.3f} ± {xgb_r2.std():.3f}")
    print(f"               MAE: {-xgb_mae.mean():.1f} ± {xgb_mae.std():.1f}")
    xgb_model.fit(X_scaled, y)

    # Feature Importance (XGBoost 기준)
    fi = pd.Series(xgb_model.feature_importances_, index=FEATURES)
    print(f"\n[Feature Importance]\n{fi.sort_values(ascending=False).to_string()}")

    # 더 좋은 모델 선택
    best_model = rf if rf_r2.mean() >= xgb_r2.mean() else xgb_model
    best_name  = "Random Forest" if rf_r2.mean() >= xgb_r2.mean() else "XGBoost"
    print(f"\n✅ Teacher Model 선택: {best_name}")

    return best_model, scaler, FEATURES


# ===========================================================
# 2단계. Pseudo-labeling — 남해 전체 MOHID에 가짜 정답 생성
# ===========================================================

def step2_generate_pseudo_labels(teacher_model, scaler, features,
                                  path_mohid_full, path_label,
                                  confidence_threshold=0.0):
    """
    남해 전체 MOHID 결과에 Teacher Model을 적용해 pseudo-label을 생성합니다.
    confidence_threshold: 예측값이 이 값 미만이면 제외 (음수 예측 필터링용)

    Returns: pseudo-label이 추가된 DataFrame
    """
    print("\n" + "=" * 55)
    print("2단계: Pseudo-label 생성")
    print("=" * 55)

    df_mohid_full = pd.read_csv(path_mohid_full)
    df_label = pd.read_csv(path_label)

    # 이미 정답지가 있는 지역은 pseudo-label 대신 실제 정답 사용
    labeled_names = set(df_label['지역명'])

    # MOHID 전체에서 정답 없는 지역만 pseudo-label 대상
    df_unlabeled = df_mohid_full[~df_mohid_full['name'].isin(labeled_names)].copy()
    df_already_labeled = df_mohid_full[df_mohid_full['name'].isin(labeled_names)].copy()

    print(f"MOHID 전체 지역 수:       {len(df_mohid_full)}개")
    print(f"정답지 보유 지역:          {len(df_already_labeled)}개 → 실제 정답 사용")
    print(f"Pseudo-label 생성 대상:   {len(df_unlabeled)}개")

    # ★ MOHID 전체 파일의 컬럼명에 맞게 아래를 수정하세요
    # 예상 컬럼: ADM_CD, ADM_NM, name, source_count, Latitude, Longitude
    # (Lat/Lon이 없는 경우 별도 지역 좌표 파일과 merge 필요)
    X_unlabeled = df_unlabeled[features].values
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    pseudo_labels = teacher_model.predict(X_unlabeled_scaled)
    pseudo_labels = np.maximum(pseudo_labels, 0)  # 음수 방지

    df_unlabeled[TARGET] = pseudo_labels
    df_unlabeled['is_pseudo'] = True

    # 실제 정답 지역도 병합
    df_real = pd.merge(df_already_labeled, df_label,
                       left_on='name', right_on='지역명', how='left')
    df_real['is_pseudo'] = False

    # 전체 통합 데이터셋
    pseudo_cols = ['name', 'source_count', 'Latitude', 'Longitude',
                   TARGET, 'is_pseudo']
    df_pseudo = pd.concat([
        df_unlabeled[pseudo_cols],
        df_real.rename(columns={'지역명': 'name'})[pseudo_cols]
    ], ignore_index=True)

    print(f"\n✅ 최종 pseudo-labeled 데이터셋: {len(df_pseudo)}개")
    print(f"   - 실제 정답: {(~df_pseudo['is_pseudo']).sum()}개")
    print(f"   - 가짜 정답: {df_pseudo['is_pseudo'].sum()}개")

    df_pseudo.to_csv("pseudo_labeled_dataset.csv", index=False, encoding='utf-8-sig')
    print("   → 저장: pseudo_labeled_dataset.csv")

    return df_pseudo


# ===========================================================
# 3단계. LSTM Pre-training — pseudo-label 데이터로 사전학습
# ===========================================================

class LSTMRegressor(nn.Module):
    """
    공간 feature를 받아 유입량을 예측하는 LSTM 모델.
    입력: (batch, seq_len, input_size)
    출력: (batch, 1)
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # 마지막 timestep
        return self.fc(out).squeeze(-1)


def step3_pretrain_lstm(df_pseudo, features, epochs=100, batch_size=16,
                         lr=1e-3, hidden_size=64, num_layers=2):
    """
    Pseudo-labeled 데이터 전체로 LSTM을 사전학습합니다.
    Returns: 사전학습된 모델, scaler
    """
    print("\n" + "=" * 55)
    print("3단계: LSTM 사전학습 (Pre-training)")
    print("=" * 55)

    scaler_dl = StandardScaler()
    X = scaler_dl.fit_transform(df_pseudo[features].values)
    y = df_pseudo[TARGET].values.astype(np.float32)

    # LSTM 입력: (N, seq_len=1, features) — 단일 시점 공간 데이터
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device} | 샘플 수: {len(df_pseudo)}개 | Epoch: {epochs}")

    model = LSTMRegressor(input_size=len(features),
                           hidden_size=hidden_size,
                           num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "lstm_pretrained.pt")
    print("✅ 사전학습 완료 → 저장: lstm_pretrained.pt")

    return model, scaler_dl, loss_history


# ===========================================================
# 4단계. Fine-tuning — 진짜 정답지(소수)로 미세조정
# ===========================================================

def step4_finetune(pretrained_model, scaler_dl, features,
                    path_label, path_mohid_partial,
                    epochs=50, lr=1e-4):
    """
    사전학습된 LSTM을 진짜 정답지(소수)로 미세조정합니다.
    낮은 learning rate + 초기 레이어 동결(freeze)로 과적합 방지.
    Returns: 최종 모델
    """
    print("\n" + "=" * 55)
    print("4단계: Fine-tuning (진짜 정답지로 미세조정)")
    print("=" * 55)

    df_label = pd.read_csv(path_label)
    df_mohid = pd.read_csv(path_mohid_partial)
    df = pd.merge(df_label, df_mohid, left_on='지역명', right_on='name',
                  how='inner')
    print(f"Fine-tuning 샘플 수: {len(df)}개")

    X = scaler_dl.transform(df[features].values)
    y = df[TARGET].values.astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=8, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = pretrained_model.to(device)

    # LSTM 레이어 동결 → FC 레이어만 먼저 학습
    for param in model.lstm.parameters():
        param.requires_grad = False
    print("  [Phase 1] LSTM 동결, FC 레이어만 학습 (20 epoch)")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    def _train(n_epochs, tag=""):
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg = epoch_loss / len(dataset)
            if epoch % 10 == 0:
                print(f"  {tag} Epoch {epoch:3d} | Loss: {avg:.4f}")

    _train(20, tag="[Phase1]")

    # LSTM 동결 해제 → 전체 미세조정
    for param in model.lstm.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
    print(f"  [Phase 2] 전체 레이어 미세조정 ({epochs - 20} epoch)")
    _train(epochs - 20, tag="[Phase2]")

    # 최종 평가
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()
    r2  = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"\n✅ Fine-tuning 완료")
    print(f"   Train R²: {r2:.3f} | Train MAE: {mae:.1f}")

    torch.save(model.state_dict(), "lstm_finetuned.pt")
    print("   → 저장: lstm_finetuned.pt")

    return model


# ===========================================================
# 메인 실행
# ===========================================================

if __name__ == "__main__":

    # ── 1단계: Teacher Model 학습 ──────────────────────────
    teacher, scaler_ml, features = step1_train_teacher(
        PATH_LABEL, PATH_MOHID_PARTIAL
    )

    # ── 2단계: Pseudo-label 생성 ───────────────────────────
    # ★ 남해 전체 MOHID 데이터가 도착하면 아래 주석 해제 후 실행
    #
    # df_pseudo = step2_generate_pseudo_labels(
    #     teacher, scaler_ml, features,
    #     path_mohid_full=PATH_MOHID_FULL,
    #     path_label=PATH_LABEL
    # )

    # ── 3단계: LSTM 사전학습 ────────────────────────────────
    # ★ 2단계 완료 후 실행
    #
    # lstm_model, scaler_dl, loss_hist = step3_pretrain_lstm(
    #     df_pseudo, features,
    #     epochs=100, hidden_size=64, num_layers=2
    # )

    # ── 4단계: Fine-tuning ─────────────────────────────────
    # ★ 3단계 완료 후 실행
    #
    # final_model = step4_finetune(
    #     lstm_model, scaler_dl, features,
    #     PATH_LABEL, PATH_MOHID_PARTIAL,
    #     epochs=50, lr=1e-4
    # )

    print("\n[현재] 1단계까지 실행 완료.")
    print("[대기] PATH_MOHID_FULL 파일 수령 후 2~4단계 주석 해제하여 실행하세요.")
