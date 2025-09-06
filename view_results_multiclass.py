import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- 설정 ---
# 1. 학습된 모델 파일의 경로를 지정하세요. (.pt 또는 .onnx)
model_path = "C:/Users/cho-j/Downloads/runs3_backup/train/gb4_exp1/weights/best.pt"

# 2. 예측을 수행할 이미지가 담긴 폴더 경로를 지정하세요.
input_folder_path = 'C:/Users/cho-j/Downloads/tmp/preprocessed'

# 3. 예측 결과 이미지를 저장할 폴더 이름을 지정하세요.
output_folder_name = 'C:/Users/cho-j/Downloads/tmp/preprocessed/predictions_with_measurements'

# 4. 이미지 크기 및 실제 세계 크기 설정 (실시간 탐지 코드와 동일하게)
IMG_SIZE = 1024.0  # 이미지 크기 (픽셀)
REAL_WORLD_MM = 118.0  # 이미지에 해당하는 실제 길이 (mm)
MM_PER_PIXEL = REAL_WORLD_MM / IMG_SIZE # 픽셀-밀리미터 변환 계수
# --- 설정 끝 ---


# 디바이스 설정 (GPU가 있으면 CUDA 사용, 없으면 CPU 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

try:
    # 모델 로드
    print(f"[INFO] Loading model... Path: {model_path}")
    model = YOLO(model_path)
    model.to(device) # 모델을 지정된 디바이스로 이동

    # 모델의 클래스 이름 가져오기
    class_names = model.names
    print(f"[INFO] Model classes: {class_names}")

    # 클래스별 고정 색상 지정 (BGR 형식)
    # data.yaml 순서에 따라 'Colony'는 0, 'InhibitionZone'은 1로 가정
    class_colors = {
        0: (0, 255, 0),  # Colony: 초록색
        1: (255, 0, 0)   # InhibitionZone: 파란색
    }

    # 결과 저장 폴더 생성
    os.makedirs(output_folder_name, exist_ok=True)
    print(f"[INFO] Output folder: '{output_folder_name}'")

    # 유효한 이미지 확장자 정의
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # 지정된 폴더의 모든 이미지 파일에 대해 예측 수행
    for filename in os.listdir(input_folder_path):
        if filename.lower().endswith(valid_exts):
            image_path = os.path.join(input_folder_path, filename)

            print(f"[INFO] Processing '{filename}'...")

            # 모델 예측 수행
            results = model.predict(
                source=image_path,
                conf=0.05,
                imgsz=int(IMG_SIZE),
                device=device,
                save=False,
                verbose=False
            )

            # 첫 번째 결과 사용
            result = results[0]
            
            # 시각화를 위해 원본 이미지 복사 (BGR)
            img_vis = result.orig_img.copy() 
            
            # 감지된 객체가 없을 경우 건너뛰기
            if result.boxes is None:
                print(f"[INFO] No objects detected in '{filename}'. Skipping.")
                continue

            # 현재 이미지의 콜로니 개수를 저장할 변수 초기화
            colony_count = 0

            # 감지된 각 객체에 대해 반복
            for box in result.boxes:
                # 좌표, 신뢰도, 클래스 ID 가져오기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # 클래스에 해당하는 색상 및 이름 가져오기
                color = class_colors.get(cls_id, (255, 255, 255)) # 기본값 흰색
                class_name = class_names[cls_id]
                
                label = ""
                # 클래스 이름에 따라 레이블 다르게 생성
                if class_name == 'InhibitionZone':
                    # 억제 구역의 실제 지름 및 넓이 계산
                    bbox_w_px = x2 - x1
                    bbox_h_px = y2 - y1
                    diameter_px = min(bbox_w_px, bbox_h_px)
                    
                    diameter_mm = diameter_px * MM_PER_PIXEL
                    radius_mm = diameter_mm / 2
                    area_mm2 = np.pi * (radius_mm ** 2)
                    
                    label = f"Dia:{diameter_mm:.1f}mm Area:{area_mm2:.1f}mm2"
                
                elif class_name == 'Colony':
                    # 콜로니는 신뢰도 표시
                    label = f'{class_name} {conf:.2f}'
                    # 콜로니 개수 1 증가
                    colony_count += 1

                # 텍스트 크기 계산
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 경계 상자 그리기
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # 레이블 배경 및 텍스트 그리기
                cv2.rectangle(img_vis, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(img_vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 이미지에 총 콜로니 개수 표시
            count_text = f"Colony Count: {colony_count}"
            # 텍스트 색상을 빨간색(BGR: 0, 0, 255)으로, 크기는 1, 두께는 2로 설정
            cv2.putText(img_vis, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 결과 이미지 저장
            save_path = os.path.join(output_folder_name, filename)
            cv2.imwrite(save_path, img_vis)

    print(f"\n[INFO] All predictions are complete! Results saved in '{output_folder_name}' folder.")

except FileNotFoundError:
    print(f"[ERROR] Could not find the model file or image folder. Please check the paths.")
    print(f"Model path: {model_path}")
    print(f"Image folder path: {input_folder_path}")
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

