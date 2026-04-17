# opencv_basic.py

import cv2
import numpy as np

# ── 1. 이미지 읽기/쓰기/보기 ──────────────────────────────
img = cv2.imread('test.jpg')          # BGR로 읽음 (RGB 아님! OpenCV 특징)
print(f"shape: {img.shape}")          # (height, width, channels)

cv2.imshow('원본', img)
cv2.waitKey(0)                        # 키 입력 대기
cv2.destroyAllWindows()

cv2.imwrite('output.jpg', img)        # 저장

# ── 2. 색공간 변환 ────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 그레이스케일
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # HSV (색상 기반 필터링에 유용)

# ── 3. 크기 조정 / 자르기 ─────────────────────────────────
resized = cv2.resize(img, (640, 480))           # 고정 크기
cropped = img[100:300, 200:400]                 # [y1:y2, x1:x2] — ROI 자르기

# ── 4. 전처리 (공장에서 자주 쓰는 것들) ──────────────────
# 블러 — 노이즈 제거
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 엣지 탐지 — 스크래치, 균열 찾기에 유용
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# 이진화 — 밝기 기준으로 흑/백 분리
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 적응형 이진화 — 조명이 불균일할 때 (공장에서 더 유용)
adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# ── 5. 컨투어 (윤곽선) 찾기 ──────────────────────────────
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:                              # 너무 작은 건 무시
        x, y, w, h = cv2.boundingRect(cnt)     # 바운딩 박스
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 초록 박스

# ── 6. 카메라/영상 스트림 ─────────────────────────────────
cap = cv2.VideoCapture(0)    # 0 = 웹캠, 'video.mp4' = 파일

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('카메라', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()