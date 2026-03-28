import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os

print("🧠 [Phase 2] 해양 쓰레기 예측 딥러닝(AI) 모델 학습 시작...\n")

# 1. 아까 만든 '문제집' 불러오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '../../final_data/AI_Training_Dataset.csv')
df = pd.read_csv(data_path)

# -------------------------------------------------------------
# 🌟 [임시 조치] 현재 스페인 바다 데이터라 입자 수가 0입니다. 
# AI가 학습하는 모습을 보기 위해, 실제 쓰레기 수량과 비슷한 패턴의 
# '가상 MOHID 데이터'를 임시로 주입합니다. (나중에 진짜 데이터가 오면 이 부분만 지우면 됩니다!)
np.random.seed(42)
# 실제 쓰레기 수량에 비례해서 가상 입자가 밀려왔다고 가정 (오차 10% 추가)
df['mohid_particle_count'] = df['수량(개)'] * np.random.uniform(50, 100, size=len(df))
# -------------------------------------------------------------

# 2. AI에게 줄 입력값(X)과 정답(Y) 준비하기
X = df[['mohid_particle_count']].values # 입력: MOHID 가상 입자 유입량
Y = df[['수량(개)']].values             # 정답: 실제 쓰레기 수량

# 데이터 크기 맞추기 (AI가 숫자 크기에 압도되지 않게 0~1 사이로 스케일링)
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# 파이토치(PyTorch) 텐서로 변환 (AI가 이해하는 언어로 번역)
X_tensor = torch.FloatTensor(X_scaled)
Y_tensor = torch.FloatTensor(Y_scaled)

# 3. 인공지능(Neural Network) 뇌 구조 만들기
class MarineDebrisPredictor(nn.Module):
    def __init__(self):
        super(MarineDebrisPredictor, self).__init__()
        # AI의 뇌 신경망 레이어 구성 (블랙박스를 그레이박스로!)
        self.layer1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)

# 모델 생성 및 학습 규칙 설정
model = MarineDebrisPredictor()
criterion = nn.MSELoss() # 오차를 계산하는 함수
optimizer = optim.Adam(model.parameters(), lr=0.01) # 오차를 줄여나가는 최적화 함수

# 4. AI 학습시키기 (수능 문제집 풀게 하기)
epochs = 500 # 문제집을 500번 반복해서 풉니다.
print("📚 AI가 남해안 해양 물리 법칙을 학습 중입니다...\n")

for epoch in range(epochs):
    optimizer.zero_grad()       # 머리 비우기
    predictions = model(X_tensor) # 예측해보기
    loss = criterion(predictions, Y_tensor) # 정답과 비교해서 오차 계산
    loss.backward()             # 오차의 원인 파악 (오답노트)
    optimizer.step()            # 다음엔 더 잘 맞추도록 뇌(가중치) 업데이트

    if (epoch+1) % 100 == 0:
        print(f"🔄 학습 {epoch+1}번 완료 | 현재 예측 오차율(Loss): {loss.item():.4f}")

# 5. 모의고사 채점! (예측력 확인하기)
print("\n🎯 학습 완료! 여수 데이터로 모의고사 예측을 수행합니다.")
model.eval()

# 진짜 데이터로 예측해보기 (스케일링 복구)
with torch.no_grad():
    predicted_scaled = model(X_tensor)
    predicted_real = scaler_Y.inverse_transform(predicted_scaled.numpy())

# 결과 비교 표 만들기
df['AI_예측_수량(개)'] = np.round(predicted_real)
print("\n📊 [최종 결과] 실제 수거량 vs AI가 예측한 수거량 비교:")
print(df[['지역명', '수량(개)', 'AI_예측_수량(개)']].head(10))