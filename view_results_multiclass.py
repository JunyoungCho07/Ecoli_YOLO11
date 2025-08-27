import os
import cv2
import torch
import random
import numpy as np
from ultralytics import YOLO

# --- 설정 ---
# 1. 학습된 best.pt 모델 파일의 경로를 지정하세요.
model_path = "C:/Users/cho-j/Downloads/runs3_backup/train/gb4_exp1/weights/best.pt" # 예시 경로, 실제 경로로 수정 필요

# 2. 예측을 수행할 이미지가 담긴 폴더 경로를 지정하세요.
input_folder_path = 'C:/Users/cho-j/Downloads/3rd ori/images/Validation' # 예시 경로

# 3. 예측 결과 이미지를 저장할 폴더 이름을 지정하세요.
output_folder_name = 'predictions_multiclass_styled'
# --- 설정 끝 ---

# 디바이스 설정 (GPU가 있으면 CUDA 사용, 없으면 CPU 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] 사용 디바이스: {device}")

try:
    # 모델 로드
    print(f"[INFO] 모델 로드 중... 경로: {model_path}")
    model = YOLO(model_path)
    model.to(device) # 모델을 지정된 디바이스로 이동

    # 모델의 클래스 이름 가져오기
    class_names = model.names
    print(f"[INFO] 모델 클래스: {class_names}")

    # 각 클래스에 대한 고유 색상 생성 (BGR 형식)
    random.seed(42) # 항상 같은 색상을 사용하도록 시드 고정
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]

    # 결과 저장 폴더 생성
    os.makedirs(output_folder_name, exist_ok=True)
    print(f"[INFO] 결과 저장 폴더: '{output_folder_name}'")

    # 유효한 이미지 확장자 정의
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # 지정된 폴더의 모든 이미지 파일에 대해 예측 수행
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(valid_exts):
            image_path = os.path.join(input_folder_path, filename)

            print(f"[INFO] '{filename}' 예측 수행 중...")

            # 모델 예측 수행
            results = model.predict(
                source=image_path,
                conf=0.25,
                imgsz=1024,
                device=device,
                save=False
            )

            # 첫 번째 결과 사용
            result = results[0]
            
            # --- 개선된 시각화 시작 ---
            img_vis = result.orig_img.copy() # 원본 이미지 복사 (BGR)
            
            # 감지된 객체가 없을 경우 건너뛰기
            if result.boxes is None:
                continue

            # 감지된 각 객체에 대해 반복
            for box in result.boxes:
                # 좌표, 신뢰도, 클래스 ID 가져오기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # 클래스에 해당하는 색상 및 이름 가져오기
                color = colors[cls_id]
                class_name = class_names[cls_id]
                
                # 레이블 텍스트 생성 (예: "Colony 0.92")
                label = f'{class_name} {conf:.2f}'
                
                # 텍스트 크기 계산
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 레이블 배경 그리기 (반투명 효과)
                overlay = img_vis.copy()
                cv2.rectangle(overlay, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                alpha = 0.6  # 투명도
                img_vis = cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0)

                # 경계 상자 그리기
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # 레이블 텍스트 그리기 (흰색)
                cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # --- 개선된 시각화 끝 ---

            # 결과 이미지 저장
            save_path = os.path.join(output_folder_name, filename)
            cv2.imwrite(save_path, img_vis)

    print(f"\n🎉 모든 예측 완료! 결과가 '{output_folder_name}' 폴더에 저장되었습니다.")

except FileNotFoundError:
    print(f"[ERROR] 모델 파일 또는 이미지 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
    print(f"모델 경로: {model_path}")
    print(f"이미지 폴더 경로: {input_folder_path}")
except Exception as e:
    print(f"[ERROR] 오류 발생: {e}")
