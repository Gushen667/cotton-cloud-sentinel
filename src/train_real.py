# ============================================================
# 棉云哨 - YOLOv8 训练脚本（真实数据版）
# 用法：python src/train_real.py
# ============================================================

from ultralytics import YOLO
import os

def main():
    print("=" * 60)
    print("棉云哨 - YOLOv8 模型训练（真实数据集）")
    print("=" * 60)
    
    # 检查是否已有模型
    model_path = 'yolo11n.pt'
    
    # 如果网络不通，尝试使用已缓存的模型
    if not os.path.exists(model_path):
        print("⚠ 预训练模型不存在，尝试下载...")
        print("  如果下载失败，将使用生成的模拟数据继续演示")
        
        # 尝试从国内镜像下载（如果可用）
        try:
            import urllib.request
            mirror_urls = [
                'https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt',
                'https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt',
            ]
            for url in mirror_urls:
                try:
                    print(f"  尝试镜像: {url}")
                    urllib.request.urlretrieve(url, model_path)
                    print(f"  ✓ 下载成功")
                    break
                except:
                    continue
        except:
            pass
        
        if not os.path.exists(model_path):
            print("\n✗ 无法下载预训练模型，使用模拟数据训练演示模型")
            print("  建议：找老师要网络代理或VPN")
            return
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 训练
    print("\n开始训练...")
    results = model.train(
        data='data.yaml',
        epochs=30,                  # 减少轮数（CPU训练慢）
        imgsz=640,                  # 图像尺寸（压缩高分辨率图像）
        batch=4,                    # 批次大小（CPU内存限制）
        device='cpu',               # 使用CPU（PyTorch没有CUDA）
        name='whitefly_real',       # 实验名称
        patience=5,                 # 早停
        save=True,
        plots=True,
        val=True,
        project='runs/train',
    )
    
    # 评估
    metrics = model.val()
    print(f"\n{'='*60}")
    print("训练完成！验证结果：")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  精确率:    {metrics.box.mp:.4f}")
    print(f"  召回率:    {metrics.box.mr:.4f}")
    print(f"{'='*60}")
    
    # 导出
    print("\n导出 ONNX 模型...")
    model.export(format='onnx', simplify=True, opset=11)
    print(f"✓ 导出完成")
    
    print(f"\n模型保存在: runs/train/whitefly_real/weights/best.pt")
    print("下一步：更新 app_v2.py 使用新模型")

if __name__ == "__main__":
    main()
