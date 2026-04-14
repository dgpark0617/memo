import numpy as np  # 수치 계산 라이브러리

# =========================
# 1. 랜덤 1분봉 데이터 생성
# =========================
def generate_data(n=500):  # n개의 캔들 생성
    price = 100  # 초기 가격
    ohlcv = []  # OHLCV 저장 리스트
    delta_list = []  # delta 저장
    intensity_list = []  # intensity 저장

    for _ in range(n):  # n번 반복
        change = np.random.randn() * 0.5  # 랜덤 가격 변화

        open_p = price  # 시가
        close_p = price + change  # 종가

        high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2  # 고가
        low_p  = min(open_p, close_p) - abs(np.random.randn()) * 0.2  # 저가

        volume = abs(np.random.randn()) * 10  # 거래량

        delta = np.random.randn() * volume  # 매수/매도 힘
        intensity = abs(np.random.randn()) * 10  # 체결 강도

        ohlcv.append([open_p, high_p, low_p, close_p, volume])  # 저장
        delta_list.append(delta)
        intensity_list.append(intensity)

        price = close_p  # 다음 캔들을 위해 가격 갱신

    return np.array(ohlcv), np.array(delta_list), np.array(intensity_list)


# =========================
# 2. Feature 생성
# =========================
def build_features(ohlcv, delta, intensity):
    O = ohlcv[:, 0]  # 시가
    H = ohlcv[:, 1]  # 고가
    L = ohlcv[:, 2]  # 저가
    C = ohlcv[:, 3]  # 종가
    V = ohlcv[:, 4]  # 거래량

    range_ = H - L  # 전체 변동폭
    body = C - O  # 몸통

    upper_wick = H - np.maximum(O, C)  # 윗꼬리
    lower_wick = np.minimum(O, C) - L  # 아래꼬리

    eps = 1e-6  # 0 나눗셈 방지

    body_ratio = body / (range_ + eps)  # 몸통 비율
    wick_ratio = (upper_wick + lower_wick) / (range_ + eps)  # 꼬리 비율

    delta_ratio = delta / (V + eps)  # 방향성 정규화

    features = np.stack([
        O, H, L, C, V,
        range_,
        body,
        upper_wick,
        lower_wick,
        body_ratio,
        wick_ratio,
        delta,
        delta_ratio,
        intensity
    ], axis=1)

    return features


# =========================
# 3. LNN 모델
# =========================
class SimpleLNN:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim)  # feature별 weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # 시그모이드

    def forward(self, X):
        h = X[0, 3]  # 초기 상태 (첫 close)
        outputs = []

        for x in X:
            score = np.dot(self.w, x)  # feature 가중합
            dt = self.sigmoid(score)  # adaptive 반응 속도

            h = h + dt * (x[3] - h)  # 상태 업데이트
            outputs.append(h)

        return np.array(outputs)

    def loss(self, pred, target):
        return np.mean((pred - target) ** 2)  # MSE

    def train(self, X, epochs=200, lr=0.001):
        target = X[:, 3]  # close 기준 예측

        for epoch in range(epochs):
            pred = self.forward(X)
            loss = self.loss(pred[:-1], target[1:])

            grad = np.zeros_like(self.w)
            eps = 1e-5

            for i in range(len(self.w)):
                original = self.w[i]

                self.w[i] = original + eps
                pred_eps = self.forward(X)
                loss_eps = self.loss(pred_eps[:-1], target[1:])

                grad[i] = (loss_eps - loss) / eps
                self.w[i] = original

            self.w -= lr * grad

            if epoch % 50 == 0:
                print(f"epoch {epoch}, loss: {loss:.6f}")

    def save(self, path="lnn_model.npy"):
        np.save(path, self.w)

    def load(self, path="lnn_model.npy"):
        self.w = np.load(path)


# =========================
# 4. 실행
# =========================
ohlcv, delta, intensity = generate_data()  # 데이터 생성
X = build_features(ohlcv, delta, intensity)  # feature 생성

model = SimpleLNN(input_dim=X.shape[1])  # 모델 생성
model.train(X)  # 학습

model.save()  # 저장

# =========================
# 5. 재사용
# =========================
model2 = SimpleLNN(input_dim=X.shape[1])
model2.load()

pred = model2.forward(X)

print("Loaded weight:", model2.w)