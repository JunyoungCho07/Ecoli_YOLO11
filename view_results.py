from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# 모델 경로 및 로드
model_path = 'C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11_det/runs/train/gb4_exp22/weights/best.pt'
model = YOLO(model_path, task='detect')  # YOLOv11-DET 모델은 'detect'

# 이미지 폴더
folder_path = 'C:/Users/cho-j/OneDrive/바탕 화면/data2nd/images/Val'
valid_exts = ('.jpg', '.jpeg', '.png')

# 디바이스 설정 (GPU 우선)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# 이미지 반복 처리
for filename in os.listdir(folder_path):
    if filename.lower().endswith(valid_exts):
        image_path = os.path.join(folder_path, filename)
        img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            print(f"[WARNING] 이미지 로드 실패: {filename}")
            continue

        # RGB로 변환
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 예측 수행
        results = model.predict(
            source=img_rgb,
            save=False,
            conf=0.25,
            imgsz=1024,         # <-- 이미지 사이즈 설정
            device=device
        )

        result = results[0]

        if result.boxes is None or result.boxes.xyxy.shape[0] == 0:
            print(f"[INFO] 박스 없음: {filename}")
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        img_vis = img_rgb.copy()
        colors = [[random.randint(100, 255) for _ in range(3)] for _ in range(len(boxes))]

        for i, (box, cls_id) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            color = colors[i]
            label = f"Colony"  # 클래스가 1개라면 고정

            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 시각화
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img_vis)
        # plt.title(f"Prediction: {filename}")
        # plt.axis('off')
        # plt.show()

        # 시각화 대신 결과 저장도 가능
        output_dir = 'predictions2'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
