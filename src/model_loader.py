# 棉云哨 - 模型更新模块
# 用于在 app.py 中加载训练好的模型

import os
from ultralytics import YOLO

# 模型路径（训练后会更新这里）
MODEL_PATH = 'runs/detect/whitefly_detection/weights/best.pt'
DEFAULT_MODEL = 'yolo11n.pt'  # 备用：通用模型

def load_model():
    """加载训练好的模型，如果不存在则使用默认模型"""
    if os.path.exists(MODEL_PATH):
        print(f"加载训练模型: {MODEL_PATH}")
        return YOLO(MODEL_PATH)
    else:
        print("训练模型不存在，使用默认通用模型")
        return YOLO(DEFAULT_MODEL)

def detect_whitefly(image_path, conf_threshold=0.4):
    """检测黄板照片中的烟粉虱"""
    model = load_model()
    results = model(image_path, conf=conf_threshold, verbose=False)
    
    count = len(results[0].boxes) if results[0].boxes is not None else 0
    return results, count
