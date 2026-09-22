import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'runs/detect/final/weights/best.pt')
    models = model.model.yaml
    print(models)

    results = model.val(data=r'ultralytics/cfg/datasets/data.yaml',
              imgsz=640,
              device=0,
              batch = 64,
              # save_json=True,
              )
