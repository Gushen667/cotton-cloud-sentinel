# ============================================================
# 棉云哨 - 真实数据推理模块
# 农户上传黄板照片 → YOLO模型识别烟粉虱 → 返回数量、位置、
# 图像质量可靠度、识别置信度
# ============================================================

import os
import hashlib
import tempfile

import numpy as np
from pathlib import Path

# cv2 可选降级（Streamlit Cloud 等无 GUI 环境安装失败时不影响核心识别）
try:
    import cv2
except Exception:
    cv2 = None

# 模型路径：训练完成后这里会有 whitefly_real 的 best.pt
# 按优先级查找
MODEL_CANDIDATES = [
    "runs/detect/runs/train/whitefly_real-2/weights/best.pt",
    "runs/train/whitefly_real-2/weights/best.pt",
    "runs/detect/runs/train/whitefly_real/weights/best.pt",
    "runs/train/whitefly_real/weights/best.pt",
    "runs/detect/whitefly_real-2/weights/best.pt",
    "runs/detect/whitefly_real/weights/best.pt",
]

def find_model():
    """找到最新的训练好的模型"""
    for m in MODEL_CANDIDATES:
        if os.path.exists(m):
            return m
    # 找 runs/detect/runs/train 和 runs/train 下最新的 whitefly_real-*
    for base in [Path("runs/detect/runs/train"), Path("runs/train")]:
        if base.exists():
            matches = sorted(base.glob("whitefly_real*/weights/best.pt"))
            if matches:
                return str(matches[-1])
    return None

_MODEL = None
_MODEL_PATH = None

def get_model():
    """懒加载 YOLO 模型（缓存）"""
    global _MODEL, _MODEL_PATH
    path = find_model()
    if path is None:
        return None, None
    if _MODEL is None or path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(path)
        _MODEL_PATH = path
    return _MODEL, path


# cv2 可选降级（见文件顶部 try/except import cv2）

def _imread_unicode(path):
    """cv2.imread 在 Windows 上读不了含中文/非ASCII的路径，改走内存字节流。"""
    if cv2 is None:
        return None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, img):
    """与 _imread_unicode 对应：imencode 后写字节，避免 imwrite 的路径限制。"""
    if cv2 is None or img is None:
        return False
    ext = os.path.splitext(str(path))[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _overlay_path(image_path):
    """叠加图统一写到系统临时目录，避免污染数据集目录。"""
    key = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"wf_overlay_{key}.jpg")

def check_image_quality(image_path):
    """
    图像质量检查（防止漏检/坏图误判为"低风险"）
    返回 dict:
      sharpness: 拉普拉斯方差（模糊度，黄板照片约3-38正常）
      brightness: 平均亮度（0-255，10-240 正常）
      yellow_ratio: 黄板黄色区域占比（0-1，>0.10 说明拍的是黄板）
      ok: 综合是否可信
      reason: 不通过的原因
    """
    if cv2 is None:
        # 无 cv2 时降级：不做质量校验，直接标记为可信（不影响核心识别）
        return {"sharpness": 0, "brightness": 0, "yellow_ratio": 0.5,
                "ok": True, "reason": "质量检测不可用（无cv2），按可信处理"}
    # 排除标注图混入
    if "._det" in str(image_path) or ".overlay" in str(image_path):
        image_path = str(image_path).replace("._det", "").replace(".overlay", "")
    img = _imread_unicode(image_path)
    if img is None:
        return {"sharpness": 0, "brightness": 0, "yellow_ratio": 0,
                "ok": False, "reason": "图片无法读取"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 黄色区间 H:15-35
    mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    yellow_ratio = float(mask.sum() / 255 / mask.size)

    ok = True
    reason = ""
    # 黄板照片特征：表面偏软焦（虫体微小、粘性涂层），清晰度中位数仅约14
    # 因此阈值放宽：sharpness < 3 才判"模糊"（排除严重失焦/运动模糊）
    if sharpness < 3:
        ok = False
        reason = f"图像严重失焦（清晰度 {sharpness:.0f}，需 >3）"
    elif brightness < 15 or brightness > 245:
        ok = False
        reason = f"曝光异常（亮度 {brightness:.0f}）"
    elif yellow_ratio < 0.10:
        ok = False
        reason = f"画面中黄板占比过低（{yellow_ratio*100:.0f}%），可能不是黄板照片"

    return {"sharpness": sharpness, "brightness": brightness,
            "yellow_ratio": yellow_ratio, "ok": ok, "reason": reason}


def detect(image_path, conf=0.25):
    """
    对一张黄板照片运行检测。
    返回 dict:
      found: 是否找到训练好的模型
      count: 检出的烟粉虱数量
      boxes: [(x1,y1,x2,y2,conf), ...]
      quality: 图像质量检查结果
      confidence: 平均识别置信度（0-1，无检出为0）
      confidence_level: 高/中/低
      reliable: 结果是否可信（模型可用 + 图像质量合格）
    """
    quality = check_image_quality(image_path)

    model, path = get_model()
    if model is None:
        return {"found": False, "count": 0, "boxes": [], "quality": quality,
                "confidence": 0.0, "confidence_level": "低", "reliable": False}

    results = model(image_path, conf=conf, verbose=False)
    r = results[0]

    boxes = []
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cf = float(b.conf[0])
        boxes.append((x1, y1, x2, y2, cf))

    count = len(boxes)
    confidence = float(np.mean([b[4] for b in boxes])) if boxes else 0.0
    if count > 0 and confidence >= 0.5:
        clevel = "高"
    elif count > 0 and confidence >= 0.3:
        clevel = "中"
    else:
        clevel = "低"

    reliable = quality["ok"] and (count > 0 or quality["yellow_ratio"] > 0.15)
    # 核心防错逻辑：模型未检出(0只)且质量存疑 → 结果不可靠，不能直接判低风险
    if count == 0 and quality["yellow_ratio"] < 0.25:
        reliable = False

    return {
        "found": True,
        "count": count,
        "boxes": boxes,
        "quality": quality,
        "confidence": confidence,
        "confidence_level": clevel,
        "reliable": reliable,
    }


def draw_detection_overlay(image_path, boxes, conf=0.25, out_path=None):
    """
    在原始图片上画检测框（红框+置信度），叠加标注虫量。
    没有检出时显示"未检出目标"水印。
    """
    if cv2 is None:
        # 无 cv2 时无法画框，返回 None（app 端已有"识别失败或图片无法读取"兜底提示）
        return None
    img = _imread_unicode(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    for (x1, y1, x2, y2, cf) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"WF {cf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if not boxes:
        cv2.putText(img, "No target detected", (w // 8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    if out_path is None:
        out_path = _overlay_path(image_path)
    _imwrite_unicode(out_path, img)
    return out_path
