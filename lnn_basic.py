import numpy as np  # 수치 계산을 위한 numpy 라이브러리 사용

# =========================
# 1. 데이터 생성 (가격 형태)
# =========================
t = np.linspace(0, 10, 500)  # 0~10 구간을 500개로 나눈 시간축 생성
price = np.cumsum(np.sin(t)) + 100  # 사인파를 누적합하여 실제 가격처럼 보이게 만들고 100을 더해 양수 유지

# =========================
# 2. 모델 클래스 정의
# =========================
class SimpleLNN:  # LNN의 핵심 개념을 단순화한 모델 클래스 정의

    def __init__(self):  # 클래스 초기화 함수
        self.w = np.random.randn()  # 학습될 파라미터 w를 랜덤값으로 초기화

    def sigmoid(self, x):  # 시그모이드 함수 정의 (0~1 사이 값으로 변환)
        return 1 / (1 + np.exp(-x))  # 시그모이드 계산식

    def forward(self, data):  # 입력 데이터를 받아 예측을 수행하는 함수
        h = data[0]  # 초기 상태값을 첫 데이터로 설정
        outputs = []  # 각 시점의 상태값을 저장할 리스트

        for x in data:  # 데이터 전체를 순회하면서
            dt = self.sigmoid(self.w * abs(x - h))  # 현재 상태와 입력의 차이를 기반으로 적응형 반응 속도 계산
            h = h + dt * (x - h)  # 상태를 업데이트 (LNN의 핵심 구조)
            outputs.append(h)  # 업데이트된 상태를 결과 리스트에 추가

        return np.array(outputs)  # 결과를 numpy 배열로 변환하여 반환

    def loss(self, pred, target):  # 손실 함수 정의 (평균제곱오차)
        return np.mean((pred - target) ** 2)  # 예측값과 실제값의 차이를 제곱하여 평균 계산

    def train(self, data, epochs=200, lr=0.01):  # 모델을 학습시키는 함수
        for epoch in range(epochs):  # 지정한 epoch 수만큼 반복 학습 수행
            pred = self.forward(data)  # 현재 파라미터로 전체 데이터 예측 수행
            loss = self.loss(pred[:-1], data[1:])  # 다음 시점 예측을 기준으로 손실 계산

            eps = 1e-5  # 수치 미분을 위한 아주 작은 값 설정
            original_w = self.w  # 현재 파라미터를 임시로 저장

            self.w = original_w + eps  # 파라미터를 미세하게 증가시켜
            pred_eps = self.forward(data)  # 다시 예측 수행
            loss_eps = self.loss(pred_eps[:-1], data[1:])  # 증가된 상태에서의 손실 계산

            grad = (loss_eps - loss) / eps  # 수치 미분을 이용해 기울기(gradient) 계산

            self.w = original_w - lr * grad  # 기울기를 이용해 파라미터 업데이트 (경사하강법)

            if epoch % 50 == 0:  # 50 epoch마다 학습 상태 출력
                print(f"epoch {epoch}, loss: {loss:.6f}, w: {self.w:.4f}")  # 현재 손실과 파라미터 값 출력

    def save(self, path="lnn_model.npy"):  # 모델 저장 함수
        np.save(path, self.w)  # 현재 학습된 파라미터 w를 파일로 저장

    def load(self, path="lnn_model.npy"):  # 모델 불러오기 함수
        self.w = np.load(path)  # 저장된 파일에서 파라미터를 불러와 적용

# =========================
# 3. 모델 생성 및 학습
# =========================
model = SimpleLNN()  # 모델 객체 생성
model.train(price)  # 생성된 가격 데이터를 이용해 모델 학습 수행

# =========================
# 4. 모델 저장
# =========================
model.save()  # 학습된 모델을 파일로 저장

# =========================
# 5. 모델 불러오기 및 재사용
# =========================
model2 = SimpleLNN()  # 새로운 모델 객체 생성
model2.load()  # 저장된 파라미터를 불러와 적용

pred = model2.forward(price)  # 불러온 모델로 동일 데이터에 대해 예측 수행

print("Loaded model weight:", model2.w)  # 불러온 모델의 파라미터 값 출력