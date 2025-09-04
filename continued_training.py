from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 이전에 학습한 모델 가중치 불러오기 (best.pt 또는 last.pt)
    model = YOLO('C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11_det/runs/train/gb4_exp13/weights')

    # 추가 학습 (continue training)
    model.train(
        data='C:/Users/cho-j/OneDrive/바탕 화면/data2nd/data.yaml',
        epochs=20,              # 추가로 학습할 epoch 수 (기존 학습과 합쳐지지 않음!)
        batch=8,
        imgsz=1024,
        lr0=1e-4,               # 보통 추가 학습 시 learning rate 낮추는 게 좋다
        device=0,
        project='C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_2025/runs',
        name='exp_continued_v11',
        task='detection',
        resume=False            # resume은 체크포인트 자동 이어붙일 때만 True로 (여기선 False)
    )

