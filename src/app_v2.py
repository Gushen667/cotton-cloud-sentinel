# 棉云哨 - 演示程序 v2（集成训练模型）
# 用法：python -m streamlit run src/app_v2.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import cv2

st.set_page_config(page_title="棉云哨", layout="wide")
st.title("🐛 棉云哨 — 烟粉虱智能监测演示系统")

# 加载训练好的模型
MODEL_PATH = "runs/model/whitefly_counter.pt"

@st.cache_resource
def load_model():
    """加载 PyTorch 模型"""
    class WhiteflyCounter(torch.nn.Module):
        def __init__(self, max_count=60):
            super().__init__()
            self.max_count = max_count
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.regressor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.features(x)
            return self.regressor(x) * self.max_count
    
    model = WhiteflyCounter()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"✓ 加载模型: {MODEL_PATH}")
    else:
        print("⚠ 模型文件不存在，使用默认模型")
    model.eval()
    return model

model = load_model()

# ========== 功能1：黄板计数 ==========
st.subheader("第一步：上传黄板照片并识别")
uploaded = st.file_uploader("选择黄板照片（JPG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    image_np = np.array(image)
    
    # 预处理
    image_resized = cv2.resize(image_np, (128, 128))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # 预测
    with torch.no_grad():
        pred = model(image_tensor).item()
    
    count = int(round(pred))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="原图", use_container_width=True)
    with col2:
        st.markdown(f"""
        ### 识别结果
        - **预估虫量**: {count} 只
        - **风险等级**: {'🟢 低风险' if count < 10 else '🟡 中风险' if count < 30 else '🔴 高风险'}
        """)
        
        # 人工修正
        manual_count = st.number_input("人工修正数量（可选）", min_value=0, value=count)
        st.success(f"最终记录: **{manual_count} 只/板**")
    
    # 历史记录
    st.markdown("---")
    st.subheader("历史采样记录")
    st.table({
        "时间": ["2024-09-24 09:30", "2024-09-23 14:20", "2024-09-22 10:15"],
        "田块": ["A1-东侧", "A1-西侧", "A2-中心"],
        "虫量": [12, 35, 78],
        "风险": ["🟢 低", "🟡 中", "🔴 高"]
    })

# ========== 功能2：风险分布图 ==========
st.subheader("第二步：风险分布图")
st.markdown("""
| 采样点 | 横向位置(米) | 纵向位置(米) | 虫量 | 风险 |
|--------|-------------|-------------|------|------|
| A1-东侧 | 10 | 10 | 15 | 🟢 低 |
| A1-西侧 | 50 | 20 | 45 | 🟡 中 |
| A2-中心 | 90 | 15 | 80 | 🔴 高 |
""")

# ========== 功能3：采样导航 ==========
st.subheader("第三步：下次采样建议")
if st.button("生成采样路线"):
    st.markdown("""
    ### 推荐采样点（按优先级排序）
    | 序号 | 位置 | 原因 | 优先级 |
    |------|------|------|--------|
    | 1 | 田块A2东北边缘 | 热点扩散方向 | ⭐⭐⭐ |
    | 2 | 田块A1北侧 | 7天未采样 | ⭐⭐ |
    | 3 | 道路旁杂草带 | 寄主交界处 | ⭐⭐ |
    """)

# ========== 功能4：气象风险 ==========
st.subheader("第四步：未来48小时风险预测")
if st.button("获取天气预报"):
    st.markdown("""
    ### 🌤️ 气象数据（模拟）
    - 当前温度: 28°C
    - 湿度: 65%
    - 明日预报: 32°C，晴
    
    ### 风险计算
    ```
    风险指数 = 虫量压力(40%) + 气象适宜度(30%) + 寄主因子(20%) + 管理因子(10%)
             = 0.3×0.4 + 0.7×0.3 + 0.5×0.2 + 0.2×0.1
             = 0.49
    ```
    
    ## 🔶 黄色风险等级
    > 虫量上升趋势 + 明日高温适宜繁殖
    > 建议：增加采样频率，关注边行，评估局部处理
    """)

# 底部说明
st.divider()
st.caption("棉云哨 · 零硬件烟粉虱监测系统 · 基于PyTorch轻量化模型")
st.caption("核心闭环：拍照 → AI计数 → 风险地图 → 采样导航 → 气象预警 → 防控工单")
