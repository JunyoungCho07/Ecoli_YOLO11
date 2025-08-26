# from ultralytics import YOLO
# import multiprocessing

# if __name__ == '__main__':
#     multiprocessing.freeze_support()

#     # YOLOv11-Seg 모델 로드
#     model = YOLO('yolov11m.pt')
#     # n s m l x 모델 크기별

#     # 학습 실행
#     model.train(
#         data="C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11",
#         imgsz=1024,# 512 1024
#         epochs=50,
#         batch=8,
#         lr0=1e-3,
#         lrf=0.01,
#         device=0,
#         project='C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_2025/runs',
#         name='exp_det_v11',
#         task='detection',
#         augment=True,
#         hsv_h=0.015,  # 색조 조정
#         hsv_s=0.7,    # 채도 조정
#         hsv_v=0.4,    # 명도 조정
#         degrees=0.0,  # 회전 각도
#         translate=0.1,  # 이동ㄴ
#         scale=0.5,    # 스케일링
#         shear=0.0,    # 기울기
#         perspective=0.0,  # 투시 왜곡
#         flipud=0.0,   # 위아래 뒤집기
#         fliplr=0.5,   # 좌우 뒤집기 확률
#         mosaic=1.0,   # Mosaic 비율
#         mixup=0.0     # MixUp 비율
#     )



from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 1) 모델 로드
    model = YOLO("yolo11s.pt")



    # 3) 학습 실행
    model.train(
        # data="C:/Users/cho-j/OneDrive/바탕 화면/data2nd/data.yaml",
        data="C:/Users/cho-j/Downloads/3rd/data.yaml",
        epochs=50,
        imgsz=1024,
        batch=-1,         # 60 % VRAM 자동
        device=0,
        close_mosaic=10,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015, 
        hsv_s=0.7, 
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.1, 
        scale=0.5,
        project="runs3/train",
        name="gb4_exp1",
    )







# import torch
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from model import YOLOv1
# from dataset import EcoliDataset
# from utils import (
#     non_max_suppression,
#     mean_average_precision,
#     intersection_over_union,
#     cellboxes_to_boxes,
#     get_bboxes,
#     plot_image,
#     save_checkpoint,
#     load_checkpoint,
# )
# from loss import YoloLoss
# import os
# from tqdm import tqdm
# # 자동 혼합 정밀도(AMP)를 위한 도구 import
# from torch.cuda.amp import GradScaler, autocast

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# learning_rate = 2e-5
# # 배치 사이즈를 4로 수정 (메모리 부족 문제 해결을 위한 최우선 조치)
# batch_size = 4
# weight_decay = 0
# num_epochs = 100
# num_workers = 2
# pin_memory = True
# load_model = False
# load_model_file = "overfit.pth.tar"
# img_dir = "Ecoli"
# label_dir = "Ecoli"

# # Transforms
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, bboxes):
#         for t in self.transforms:
#             img, bboxes = t(img), bboxes
#         return img, bboxes

# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler): # scaler 인자 추가
#     model.train()
#     loop = tqdm(data_loader, leave=True)
#     mean_loss = []

#     for batch_idx, (x, y) in enumerate(loop):
#         x, y = x.to(device), y.to(device)

#         # autocast 컨텍스트 내에서 forward pass 실행
#         with autocast():
#             out = model(x)
#             loss = YoloLoss()(out, y)
        
#         mean_loss.append(loss.item())
        
#         # 기존 옵티마이저 스텝을 scaler를 사용하도록 수정
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update progress bar
#         loop.set_postfix(loss=loss.item())

#     print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

# def main():
#     model = YOLOv1(split_size=7, num_boxes=2, num_classes=1).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
#     if load_model:
#         load_checkpoint(torch.load(load_model_file), model, optimizer)

#     train_dataset = EcoliDataset(
#         csv_file=os.path.join(img_dir, "train.csv"),
#         transform=transform,
#         img_dir=img_dir,
#         label_dir=label_dir,
#     )

#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#         drop_last=False,
#     )
    
#     # GradScaler 객체 생성
#     scaler = GradScaler()

#     for epoch in range(num_epochs):
#         print(f"--- [EPOCH: {epoch+1} / {num_epochs}] ---")
        
#         # train_one_epoch 함수에 scaler 전달
#         train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        
#         if (epoch + 1) % 10 == 0:
#             checkpoint = {
#                 "state_dict": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#             }
#             save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
#             print("==> Checkpoint saved")

# if __name__ == "__main__":
#     main()


        
# 기대 데이터 확대 배수

# E = 4 × (1 + mixup)
# → mixup 0.15일 때 E ≈ 4.6 배.
# 예) 원본 10 000장 ⇒ epoch당 약 46 000 장이 모델에 노출됩니다. (flip·색상 변형은 픽셀만 변형하므로 ‘장 수’에 추가 계산하지 않음.)




# 1. 증강을 “모두 새로운 장수”로 간주할 때의 산식 — 논리
# 증강기법	동작 방식	우리가 새 장수로 더칠 때의 가산 비율
# Mosaic (mosaic=1.0)	한 학습 샘플을 만들 때 원본 4 장을 붙여 1 장을 생성	4배
# MixUp (mixup=p)	p 확률로 두 번째 Mosaic 이미지를 블렌딩	1 + p
# Horizontal Flip (fliplr=q)	q 확률로 좌우 반전	1 + q
# Rotation
# (Ultralytics YOLO의 degrees≠0)	회전은 항상 적용되어 각 샘플이 임의 각도로 돌려짐	1 + 1 ≈ 2
# (회전有·無 를 별개 장수로 본다면)

# 따라서 “각 기법을 독립적으로 겹쳐 쓴다”는 가정 아래,
# 총 확대 배수 E 는
 
# 2. 예시 — 기본 권장 하이퍼파라미터
# 파라미터	값	설명
# mosaic	1.0	항상 사용
# mixup	0.15	15 % 확률
# fliplr	0.50	50 % 확률
# degrees	±10°	임의 회전(=100 % 확률)
  
# =4×(1+0.15)×(1+0.50)×2
# =4×1.15×1.5×2
# =13.8
# ​
 
# 즉, 원본 10 000장이 있다면 한 epoch 동안

# “이미지”
# 10000×13.8=138000“이미지”
# 가 서로 다른 픽셀 구성을 갖는 형태로 모델에 노출됩니다.