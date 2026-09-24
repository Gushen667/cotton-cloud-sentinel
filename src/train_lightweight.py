# ============================================================
# 棉云哨 - 轻量级烟粉虱检测模型（无需下载预训练权重）
# 使用 MobileNetV2 + 自定义数据生成器
# ============================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import random
import json

OUTPUT_DIR = "runs/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据生成器 ====================
class YellowBoardGenerator:
    """生成模拟黄板图像用于训练"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
    
    def generate(self, num_pests):
        """生成一张黄板照片和标注"""
        # 黄色背景
        img = np.random.randint(230, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        img = img.astype(np.float32) / 255.0
        
        # 添加纹理（灰尘、划痕）
        for _ in range(1000):
            x, y = random.randint(0, self.img_size-1), random.randint(0, self.img_size-1)
            shade = random.uniform(0.8, 0.95)
            img[y, x] = shade
        
        # 添加烟粉虱（白色小点）
        pests = []
        for _ in range(num_pests):
            x = random.randint(20, self.img_size-20)
            y = random.randint(20, self.img_size-20)
            w = random.randint(3, 8)
            h = random.randint(2, 6)
            
            # 白色椭圆
            for dy in range(-h, h):
                for dx in range(-w, w):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.img_size and 0 <= nx < self.img_size:
                        if (dx/w)**2 + (dy/h)**2 <= 1:
                            img[ny, nx] = [0.95, 0.95, 0.98]
            
            pests.append((x/self.img_size, y/self.img_size, w/self.img_size, h/self.img_size))
        
        return img, pests

def create_dataset(num_samples=500, img_size=224):
    """创建训练数据集"""
    print(f"正在生成 {num_samples} 张训练图像...")
    
    X = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
    y_count = np.zeros((num_samples,), dtype=np.float32)
    
    gen = YellowBoardGenerator(img_size)
    
    for i in range(num_samples):
        # 随机虫量：5-60只（模拟低/中/高密度）
        num_pests = random.randint(5, 60)
        img, pests = gen.generate(num_pests)
        
        X[i] = img
        y_count[i] = num_pests
    
    print(f"✓ 生成完成: {X.shape}")
    return X, y_count

# ==================== 模型定义 ====================
def build_model(input_shape=(224, 224, 3)):
    """构建轻量级回归模型（预测虫量）"""
    inputs = keras.Input(shape=input_shape)
    
    # 轻量级卷积网络
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='count')(x)  # 输出归一化到 [0,1]
    
    model = keras.Model(inputs, output)
    return model

# ==================== 训练 ====================
def main():
    print("=" * 50)
    print("棉云哨 - 烟粉虱计数模型训练")
    print("=" * 50)
    
    # 生成数据
    X, y = create_dataset(500)
    
    # 划分训练/验证集（8:2）
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\n训练集: {len(X_train)} 张")
    print(f"验证集: {len(X_val)} 张")
    
    # 构建模型
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=1
    )
    
    # 保存模型
    model.save(os.path.join(OUTPUT_DIR, 'whitefly_model.keras'))
    print(f"\n✓ 模型已保存: {OUTPUT_DIR}/whitefly_model.keras")
    
    # 保存训练曲线图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('MAE Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    print(f"✓ 训练曲线已保存")
    
    # 测试预测
    print("\n预测测试:")
    test_imgs = X_val[:5]
    test_counts = y_val[:5]
    preds = model.predict(test_imgs).flatten()
    
    for i in range(5):
        print(f"  真实: {test_counts[i]:.0f}只 → 预测: {preds[i]*60:.1f}只")

if __name__ == "__main__":
    main()
