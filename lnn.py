import numpy as np

# 데이터: 같은 파형인데 속도만 다름
t = np.linspace(0, 10, 200)
slow = np.sin(t)        # 느림 (학습했다고 가정)
fast = np.sin(t * 3)    # 빠름 (새로운 상황)

# 1. "기존 방식" (고정 반응 = 둔함)
def fixed_model(data):
    out = []
    for i in range(len(data)):
        if i == 0:
            out.append(data[i])
        else:
            # 과거 평균 → 느리게 반응
            out.append(0.9 * out[-1] + 0.1 * data[i])
    return np.array(out)

# 2. "LNN 느낌" (상황에 따라 반응 속도 바뀜)
def adaptive_model(data):
    out = []
    h = 0
    for x in data:
        # 변화가 크면 빠르게 따라감
        speed = abs(x - h)

        # 핵심: 반응 속도가 계속 변함
        dt = 0.1 + 0.9 * speed

        h = h + dt * (x - h)
        out.append(h)
    return np.array(out)

# 실행
fixed_fast = fixed_model(fast)
adapt_fast = adaptive_model(fast)

# 오차 비교
def mse(a, b):
    return np.mean((a - b)**2)

print("고정 모델 오차:", mse(fixed_fast, fast))
print("적응 모델 오차:", mse(adapt_fast, fast))