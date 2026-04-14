import numpy as np

# =========================
# 1. Generate Data
# =========================
t = np.linspace(0, 10, 300)

slow = np.sin(t)        # training-like condition (slow signal)
fast = np.sin(t * 3)    # new condition (faster signal)

# =========================
# 2. EMA (fixed response speed)
# =========================
def ema(data, alpha=0.1):
    h = data[0]
    result = []

    for x in data:
        # constant update rate
        h = h + alpha * (x - h)
        result.append(h)

    return np.array(result)

# =========================
# 3. LNN-like (adaptive response speed)
# =========================
def lnn_like(data):
    h = data[0]
    result = []

    for x in data:
        # adaptive step size based on change magnitude
        dt = 0.1 + 0.9 * abs(x - h)

        # continuous-time style update
        h = h + dt * (x - h)
        result.append(h)

    return np.array(result)

# =========================
# 4. Run Models
# =========================
ema_slow = ema(slow)
ema_fast = ema(fast)

lnn_slow = lnn_like(slow)
lnn_fast = lnn_like(fast)

# =========================
# 5. Compute Error (MSE)
# =========================
def mse(a, b):
    return np.mean((a - b) ** 2)

print("=== Slow Data (training condition) ===")
print("EMA  error:", mse(ema_slow, slow))
print("LNN  error:", mse(lnn_slow, slow))

print("\n=== Fast Data (changed condition) ===")
print("EMA  error:", mse(ema_fast, fast))
print("LNN  error:", mse(lnn_fast, fast))

# =========================
# 6. Sample Output Comparison
# =========================
print("\n=== Sample comparison (fast data) ===")
for i in range(10):
    print(f"real={fast[i]: .3f} | ema={ema_fast[i]: .3f} | lnn={lnn_fast[i]: .3f}")