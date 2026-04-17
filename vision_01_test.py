# mnist_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 1. 하이퍼파라미터 ──────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 3
LR         = 0.001

# ── 2. 데이터 로딩 ────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),               # PIL → Tensor (0~255 → 0.0~1.0)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차로 정규화
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── 3. 모델 정의 ──────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 특징 추출부 (Feature Extractor)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28x1  → 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28x32 → 14x14x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14x32 → 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14x64 → 7x7x64
        )
        
        # 분류부 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),          # 7x7x64 = 3136
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(0.3),       # 과적합 방지
            nn.Linear(128, 10),    # 10개 클래스 (0~9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()
print(model)
print(f"\n학습 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ── 4. 손실함수 & 옵티마이저 ──────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── 5. 학습 루프 ──────────────────────────────────────────
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()         # 기울기 초기화
        outputs = model(images)       # forward
        loss = criterion(outputs, labels)  # loss 계산
        loss.backward()               # backward (기울기 계산)
        optimizer.step()              # 가중치 업데이트
        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} | Step {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

# ── 6. 평가 루프 ──────────────────────────────────────────
def evaluate():
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():   # 평가 시엔 기울기 계산 불필요
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"테스트 정확도: {acc:.2f}%")
    return acc

# ── 7. 실행 ───────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    avg_loss = train(epoch)
    print(f"── Epoch {epoch} 평균 Loss: {avg_loss:.4f}")
    evaluate()

# ── 8. 모델 저장 ──────────────────────────────────────────
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\n모델 저장 완료: mnist_cnn.pth")