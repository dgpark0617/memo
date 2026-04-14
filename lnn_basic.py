import numpy as np

# =========================
# 데이터 생성
# =========================
def generate_data(n=500):
    price = 100
    ohlcv, delta_list, intensity_list = [], [], []

    for _ in range(n):
        change = np.random.randn() * 0.5

        open_p = price
        close_p = price + change

        high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2
        low_p  = min(open_p, close_p) - abs(np.random.randn()) * 0.2

        volume = abs(np.random.randn()) * 10
        delta = np.random.randn() * volume
        intensity = abs(np.random.randn()) * 10

        ohlcv.append([open_p, high_p, low_p, close_p, volume])
        delta_list.append(delta)
        intensity_list.append(intensity)

        price = close_p

    return np.array(ohlcv), np.array(delta_list), np.array(intensity_list)


# =========================
# Feature 생성
# =========================
def build_features(ohlcv, delta, intensity):
    O, H, L, C, V = ohlcv.T

    range_ = H - L
    body = C - O

    upper_wick = H - np.maximum(O, C)
    lower_wick = np.minimum(O, C) - L

    eps = 1e-6

    body_ratio = body / (range_ + eps)
    wick_ratio = (upper_wick + lower_wick) / (range_ + eps)
    delta_ratio = delta / (V + eps)

    return np.stack([
        O, H, L, C, V,
        range_, body,
        upper_wick, lower_wick,
        body_ratio, wick_ratio,
        delta, delta_ratio,
        intensity
    ], axis=1)


# =========================
# LNN 모델
# =========================
class SimpleLNN:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        h = X[0, 3]
        outputs = []

        for x in X:
            dt = self.sigmoid(np.dot(self.w, x))
            h = h + dt * (x[3] - h)
            outputs.append(h)

        return np.array(outputs)

    def loss(self, pred, target):
        return np.mean((pred - target) ** 2)

    def train(self, X, epochs=200, lr=0.001):
        target = X[:, 3]

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
                print(f"[TRAIN] epoch {epoch}, loss: {loss:.6f}")

# =========================
# 실행 + 평가
# =========================
ohlcv, delta, intensity = generate_data()
X = build_features(ohlcv, delta, intensity)

model = SimpleLNN(input_dim=X.shape[1])

# 👉 학습 전 성능
pred_before = model.forward(X)
loss_before = model.loss(pred_before[:-1], X[1:, 3])

print("\n=== BEFORE TRAIN ===")
print("MSE:", loss_before)

# 👉 학습
model.train(X)

# 👉 학습 후 성능
pred_after = model.forward(X)
loss_after = model.loss(pred_after[:-1], X[1:, 3])

print("\n=== AFTER TRAIN ===")
print("MSE:", loss_after)

# 👉 개선율
improvement = (loss_before - loss_after) / loss_before * 100

print("\n=== IMPROVEMENT ===")
print(f"Improvement: {improvement:.2f}%")