"""
coin_mlp_fast.py - 코인 포지션 수익/손실 분류기
Value 클래스 없음. gradient 수식 직접 계산.
import: math, random 딱 2개.

구조: 3 → 8 → 8 → 1 (sigmoid)
"""

import math
import random

random.seed(42)

# ── 1. ACTIVATIONS ───────────────────────────────────────────────────────────

def relu(x):     return max(0.0, x)
def drelu(x):    return 1.0 if x > 0 else 0.0
def sigmoid(x):  return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
def dsigmoid(s): return s * (1.0 - s)  # s = sigmoid(x) 이미 계산된 값


# ── 2. NETWORK INIT ──────────────────────────────────────────────────────────

def make_layer(n_in, n_out):
    k = math.sqrt(2.0 / n_in)
    W = [[random.gauss(0, k) for _ in range(n_in)] for _ in range(n_out)]
    b = [0.0] * n_out
    return W, b

def init_network(layer_sizes):
    """layer_sizes = [3, 8, 8, 1]"""
    layers = []
    for i in range(len(layer_sizes) - 1):
        W, b = make_layer(layer_sizes[i], layer_sizes[i+1])
        layers.append((W, b))
    return layers


# ── 3. FORWARD ───────────────────────────────────────────────────────────────

def forward(layers, x):
    """
    returns:
      activations : 각 레이어 출력값 (relu 적용 후)
      pre_acts    : 각 레이어 출력값 (활성화 함수 적용 전)
    """
    activations = [x]
    pre_acts = []

    for i, (W, b) in enumerate(layers):
        prev = activations[-1]
        z = [sum(W[j][k] * prev[k] for k in range(len(prev))) + b[j]
             for j in range(len(b))]
        pre_acts.append(z)

        # 마지막 레이어 sigmoid, 나머지 relu
        if i == len(layers) - 1:
            a = [sigmoid(v) for v in z]
        else:
            a = [relu(v) for v in z]

        activations.append(a)

    return activations, pre_acts


def predict(layers, x):
    activations, _ = forward(layers, x)
    return activations[-1][0]


# ── 4. BACKWARD ──────────────────────────────────────────────────────────────

def backward(layers, activations, pre_acts, y_true):
    """
    Binary Cross Entropy + sigmoid 출력층 gradient 직접 계산.
    returns: grads = [(dW, db), ...] 각 레이어별
    """
    n_layers = len(layers)
    grads = [(None, None)] * n_layers

    # 출력층 delta: BCE + sigmoid → d_loss/d_z = pred - y
    pred = activations[-1][0]
    delta = [pred - y_true]

    for i in reversed(range(n_layers)):
        W, b = layers[i]
        a_in = activations[i]     # 이 레이어 입력
        a_out = activations[i+1]  # 이 레이어 출력
        z = pre_acts[i]

        n_out = len(b)
        n_in  = len(a_in)

        # gradient 계산
        dW = [[delta[j] * a_in[k] for k in range(n_in)] for j in range(n_out)]
        db = [delta[j] for j in range(n_out)]
        grads[i] = (dW, db)

        # 이전 레이어로 delta 전파 (첫 레이어면 불필요)
        if i > 0:
            delta_prev = []
            for k in range(n_in):
                d = sum(W[j][k] * delta[j] for j in range(n_out))
                d *= drelu(pre_acts[i-1][k])
                delta_prev.append(d)
            delta = delta_prev

    return grads


# ── 5. UPDATE ────────────────────────────────────────────────────────────────

def update(layers, grads, lr):
    for i, (W, b) in enumerate(layers):
        dW, db = grads[i]
        for j in range(len(b)):
            for k in range(len(W[j])):
                W[j][k] -= lr * dW[j][k]
            b[j] -= lr * db[j]


def zero_grads(n_layers, layer_sizes):
    return [([[0.0]*layer_sizes[i] for _ in range(layer_sizes[i+1])],
              [0.0]*layer_sizes[i+1])
            for i in range(n_layers)]


# ── 6. DATA ──────────────────────────────────────────────────────────────────

def generate_data(n=300):
    data = []
    for _ in range(n):
        band_width = random.uniform(1.0, 15.0)
        band_pos   = random.uniform(-10.0, 110.0)
        slope      = random.uniform(-5.0, 5.0)

        score = 0.0
        score += slope * 0.4
        score += (band_pos - 50) * 0.03
        if band_width < 4.0 and band_pos > 70:  score += 1.5
        if band_width < 4.0 and band_pos < 30:  score -= 1.5
        if band_width > 10.0 and slope > 2.0:   score += 1.0
        if band_width > 10.0 and slope < -2.0:  score -= 1.0
        score += random.gauss(0, 0.8)

        label = 1.0 if score > 0.5 else 0.0
        data.append(([band_width, band_pos, slope], label))
    return data


def normalize(data):
    n_f = len(data[0][0])
    mins = [min(d[0][i] for d in data) for i in range(n_f)]
    maxs = [max(d[0][i] for d in data) for i in range(n_f)]
    normed = [
        ([(x[i]-mins[i])/(maxs[i]-mins[i]+1e-8) for i in range(n_f)], y)
        for x, y in data
    ]
    return normed, mins, maxs


def norm_input(x, mins, maxs):
    return [(x[i]-mins[i])/(maxs[i]-mins[i]+1e-8) for i in range(len(x))]


# ── 7. LOSS ──────────────────────────────────────────────────────────────────

def bce(pred, y):
    pred = max(1e-7, min(1-1e-7, pred))
    return -(y * math.log(pred) + (1-y) * math.log(1-pred))


# ── 8. TRAIN ─────────────────────────────────────────────────────────────────

def train(layers, layer_sizes, dataset, epochs=500, lr=0.05):
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        # 누적 gradient
        acc_grads = zero_grads(len(layers), layer_sizes)

        for x, y in dataset:
            activations, pre_acts = forward(layers, x)
            pred = activations[-1][0]

            total_loss += bce(pred, y)
            correct += 1 if (pred > 0.5) == (y > 0.5) else 0

            grads = backward(layers, activations, pre_acts, y)

            # gradient 누적
            for i in range(len(layers)):
                dW, db = grads[i]
                aW, ab = acc_grads[i]
                for j in range(len(db)):
                    for k in range(len(dW[j])):
                        aW[j][k] += dW[j][k]
                    ab[j] += db[j]

        # 평균 gradient로 업데이트
        n = len(dataset)
        avg_grads = [
            ([[acc_grads[i][0][j][k]/n for k in range(len(acc_grads[i][0][j]))]
              for j in range(len(acc_grads[i][1]))],
             [acc_grads[i][1][j]/n for j in range(len(acc_grads[i][1]))])
            for i in range(len(layers))
        ]
        update(layers, avg_grads, lr)

        if epoch % 100 == 0 or epoch == epochs - 1:
            acc = correct / len(dataset) * 100
            avg_loss = total_loss / len(dataset)
            print(f"epoch {epoch:4d} | loss {avg_loss:.4f} | acc {acc:.1f}%")


# ── 9. EVALUATE ──────────────────────────────────────────────────────────────

def evaluate(layers, dataset, mins, maxs):
    tp = fp = tn = fn = 0
    for x, y in dataset:
        xn = norm_input(x, mins, maxs)
        p = predict(layers, xn)
        pred = 1 if p > 0.5 else 0
        actual = int(y)
        if pred == 1 and actual == 1: tp += 1
        elif pred == 1 and actual == 0: fp += 1
        elif pred == 0 and actual == 0: tn += 1
        else: fn += 1

    acc  = (tp+tn) / (tp+fp+tn+fn) * 100
    prec = tp / (tp+fp+1e-8) * 100
    rec  = tp / (tp+fn+1e-8) * 100
    print(f"\n{'='*50}")
    print(f"테스트 결과 ({len(dataset)}개)")
    print(f"{'='*50}")
    print(f"정확도 : {acc:.1f}%")
    print(f"정밀도 : {prec:.1f}%")
    print(f"재현율 : {rec:.1f}%")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")


# ── 10. MAIN ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("코인 포지션 수익/손실 분류기 (fast ver.)")
    print("import: math, random 딱 2개\n")

    layer_sizes = [3, 8, 8, 1]

    all_data = generate_data(n=400)
    random.shuffle(all_data)
    split = int(len(all_data) * 0.8)
    train_raw = all_data[:split]
    test_raw  = all_data[split:]

    train_data, mins, maxs = normalize(train_raw)

    layers = init_network(layer_sizes)
    n_params = sum(len(b) + sum(len(w) for w in W) for W, b in layers)
    print(f"파라미터 수  : {n_params}")
    print(f"학습 데이터  : {len(train_data)}개")
    print(f"테스트 데이터: {len(test_raw)}개\n")

    train(layers, layer_sizes, train_data, epochs=500, lr=0.05)
    evaluate(layers, test_raw, mins, maxs)

    print(f"\n{'='*50}")
    print("[ 시나리오 예측 ]")
    print(f"{'='*50}")

    scenarios = [
        ("좁은밴드 + 상단 + 강한상승", 2.5,  95.0,  3.5),
        ("넓은밴드 + 하단 + 강한하락", 12.0,  5.0, -4.0),
        ("중간밴드 + 중앙 + 횡보",      6.0,  50.0,  0.2),
        ("좁은밴드 + 하단 + 하락",      3.0,   8.0, -2.5),
        ("넓은밴드 + 상단 + 강한상승", 11.0,  92.0,  4.0),
    ]

    for name, bw, bp, sl in scenarios:
        xn = norm_input([bw, bp, sl], mins, maxs)
        prob = predict(layers, xn)
        result = "수익 예상 ✓" if prob > 0.5 else "손실 예상 ✗"
        print(f"\n{name}")
        print(f"  밴드폭={bw}% | 위치={bp}% | 기울기={sl}%")
        print(f"  → {result} (확률: {prob:.3f})")
