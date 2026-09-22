import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ['WANDB_MODE'] = 'disabled'

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11fuse.yaml')

    model.train(data=r'ultralytics/cfg/datasets/data.yaml',
                imgsz=640,
                epochs=1000,
                batch=64,
                workers=8,
                device='0',
                # resume = True
                name = 'final',
                )
