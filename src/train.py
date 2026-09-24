# ============================================================
# 棉云哨 - YOLOv8 训练脚本
# 用法：python src/train.py
# ============================================================

from ultralytics import YOLO
import os

def main():
    print("=" * 50)
    print("棉云哨 - YOLOv8 模型训练")
    print("=" * 50)
    
    # 加载预训练模型
    model = YOLO('yolo11n.pt')
    
    # 训练配置
    results = model.train(
        data='data.yaml',           # 数据集配置
        epochs=50,                  # 训练轮数（比赛演示够用了）
        imgsz=640,                  # 图像尺寸
        batch=16,                   # 批次大小（根据你的GPU调整）
        device=0,                   # GPU 0（用你的GPU）
        name='whitefly_detection',  # 实验名称
        patience=10,                # 早停（验证集不再提升时停止）
        optimizer='SGD',            # 优化器
        lr0=0.01,                   # 初始学习率
        weight_decay=0.0005,
        mosaic=0.9,                 # 数据增强（mosaic）
        mixup=0.1,                  # 数据增强（mixup）
        save=True,                  # 保存模型
        plots=True,                 # 生成训练曲线图
        val=True,                   # 每轮验证
    )
    
    # 评估模型
    metrics = model.val()
    print(f"\n验证结果:")
    print(f"  mAP@50:  {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  精确率:  {metrics.box.mp:.4f}")
    print(f"  召回率:  {metrics.box.mr:.4f}")
    
    # 导出 ONNX 模型（用于部署）
    print("\n导出 ONNX 模型...")
    model.export(format='onnx', simplify=True, opset=11)
    print("✓ 导出完成：runs/detect/whitefly_detection/exported/whitefly_detection.onnx")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"模型保存在: runs/detect/whitefly_detection/weights/best.pt")
    print("=" * 50)

if __name__ == "__main__":
    main()
