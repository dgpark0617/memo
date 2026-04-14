import numpy as np

def make_data(speed=1.0, n=200):
    t = np.linspace(0, 10, n)
    return np.sin(t * speed)

train = make_data(1.0)
test  = make_data(3.0)



______




def fixed_step_predict(data, window=10):
    preds = []
    for i in range(len(data)-window):
        preds.append(np.mean(data[i:i+window]))
    return np.array(preds)

pred_train_fixed = fixed_step_predict(train)
pred_test_fixed  = fixed_step_predict(test)

def mse(a, b):
    return np.mean((a - b)**2)

print("Fixed model train:", mse(pred_train_fixed, train[10:]))
print("Fixed model test :", mse(pred_test_fixed, test[10:]))




_______




def lnn_like_predict(data, dt=0.1):
    h = 0.0
    preds = []

    for x in data:
        dh = (x - h)          # 상태가 입력을 따라감
        h = h + dt * dh       # continuous update
        preds.append(h)

    return np.array(preds)

pred_train_lnn = lnn_like_predict(train)
pred_test_lnn  = lnn_like_predict(test)

print("LNN-like train:", mse(pred_train_lnn, train))
print("LNN-like test :", mse(pred_test_lnn, test))