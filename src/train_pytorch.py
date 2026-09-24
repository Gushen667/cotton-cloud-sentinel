# ============================================================
# 棉云哨 - PyTorch 轻量级烟粉虱计数模型
# 无需下载预训练权重，本地训练即可
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image

OUTPUT_DIR = "runs/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ==================== 数据集 ====================
class WhiteflyDataset(Dataset):
    """黄板烟粉虱数据集"""
    
    def __init__(self, num_samples=500, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        
        # 生成数据
        self.X = np.zeros((num_samples, img_size, img_size, 3), dtype=np.float32)
        self.y = np.zeros((num_samples,), dtype=np.float32)
        
        print(f"正在生成 {num_samples} 张训练图像...")
        for i in range(num_samples):
            # 随机虫量
            num_pests = random.randint(5, 60)
            
            # 黄色背景
            img = np.random.uniform(0.85, 0.95, (img_size, img_size, 3)).astype(np.float32)
            
            # 添加纹理
            for _ in range(500):
                x, y = random.randint(0, img_size-1), random.randint(0, img_size-1)
                shade = random.uniform(0.7, 0.9)
                img[y, x] = shade
            
            # 添加烟粉虱（白色小点）
            for _ in range(num_pests):
                cx = random.randint(10, img_size-10)
                cy = random.randint(10, img_size-10)
                r = random.randint(2, 5)
                
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < img_size and 0 <= nx < img_size:
                            if dx**2 + dy**2 <= r**2:
                                img[ny, nx] = [0.95, 0.95, 0.98]
            
            self.X[i] = img
            self.y[i] = num_pests
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.X[idx].transpose(2, 0, 1))  # CHW格式
        count = torch.tensor(self.y[idx], dtype=torch.float32)
        return img, count

# ==================== 模型 ====================
class WhiteflyCounter(nn.Module):
    """轻量级烟粉虱计数网络"""
    
    def __init__(self, max_count=60):
        super().__init__()
        self.max_count = max_count
        
        self.features = nn.Sequential(
            # 卷积块1
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 卷积块2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 卷积块3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x * self.max_count  # 还原到实际虫量

# ==================== 训练 ====================
def train():
    print("=" * 50)
    print("棉云哨 - PyTorch 模型训练")
    print("=" * 50)
    
    # 数据
    dataset = WhiteflyDataset(500)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"\n训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    
    # 模型
    model = WhiteflyCounter().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for imgs, counts in train_loader:
            imgs, counts = imgs.to(DEVICE), counts.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, counts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, counts in val_loader:
                imgs, counts = imgs.to(DEVICE), counts.to(DEVICE)
                outputs = model(imgs).squeeze()
                val_loss += criterion(outputs, counts).item()
        
        if epoch % 5 == 0 or epoch == 29:
            print(f"Epoch {epoch+1}/30 | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'whitefly_counter.pt'))
    print(f"\n✓ 模型已保存: {OUTPUT_DIR}/whitefly_counter.pt")
    
    # 测试
    print("\n预测测试:")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            idx = random.randint(0, len(val_dataset)-1)
            img, count = val_dataset[idx]
            img = img.unsqueeze(0).to(DEVICE)
            pred = model(img).item()
            print(f"  真实: {count.item():.0f}只 → 预测: {pred:.1f}只 (误差: {abs(pred-count.item()):.1f})")

if __name__ == "__main__":
    train()
