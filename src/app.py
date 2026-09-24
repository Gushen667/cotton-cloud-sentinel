# 棉云哨 - 烟粉虱智能监测演示系统
# 使用方法：在 D:\AI测试\whitefly 目录下执行：
#   python -m streamlit run src/app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import folium
from folium.plugins import HeatMap
import os

# 页面设置
st.set_page_config(page_title="棉云哨", layout="wide")
st.title("🐛 棉云哨 — 烟粉虱智能监测演示")
st.caption("上传黄板照片 → 自动计数 → 生成风险地图")

# ============ 页面1：黄板计数 ============
st.subheader("第一步：上传黄板照片")
uploaded = st.file_uploader("选择黄板照片（JPG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded:
    with open("temp_board.jpg", "wb") as f:
        f.write(uploaded.getvalue())
    
    st.write("正在识别...")
    
    # 加载YOLO模型（第一次会自动下载通用模型）
    with st.spinner("加载AI模型中..."):
        model = YOLO('yolo11n.pt')  # 通用目标检测模型
    
    results = model("temp_board.jpg", conf=0.4, verbose=False)
    
    # 获取检测结果
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0
    
    # 显示原图和结果
    col1, col2 = st.columns(2)
    with col1:
        st.write("**原图**")
        st.image("temp_board.jpg", use_container_width=True)
    with col2:
        # 绘制检测框
        annotated = results[0].plot()
        st.write(f"**识别结果（共 {count} 只）**")
        st.image(annotated, use_container_width=True)
    
    # 风险判断
    st.info(f"检测到 **{count}** 只烟粉虱\n> 建议：{'低风险，继续监测' if count < 10 else '中风险，增加采样' if count < 30 else '高风险，立即联系植保员'}")
    
    # 允许人工修正
    manual_count = st.number_input("人工修正数量（可选）", min_value=0, value=count)
    st.write(f"最终记录：**{manual_count} 只/板**")

# ============ 页面2：风险热点图（纯图表，不依赖外部地图） ============
st.subheader("第二步：风险热点分布图")

# 模拟数据：x=田块横坐标(米), y=纵坐标(米), val=虫量, 名称
demo_points = [
    (10, 10, 15, "田块A1"),
    (50, 20, 45, "田块A2"),
    (90, 15, 80, "田块A3"),
]

if st.button("显示风险分布"):
    import altair as alt
    import pandas as pd
    df = pd.DataFrame(demo_points, columns=["x", "y", "虫量", "田块"])
    color_map = {0: "green", 1: "orange", 2: "red"}
    df["颜色"] = ["red" if v >= 50 else "orange" if v >= 20 else "green" for v in df["虫量"]]
    
    chart = (
        alt.Chart(df)
        .mark_circle(size=200)
        .encode(
            x=alt.X("x:Q", title="横向位置 (米)", scale=alt.Scale(domain=[0, 100], nice=True)),
            y=alt.Y("y:Q", title="纵向位置 (米)", scale=alt.Scale(domain=[0, 60], nice=True)),
            color=alt.Color("颜色:N", scale=alt.Scale(domain=["green","orange","red"], range=["green","orange","red"]), title="风险等级"),
            tooltip=["田块", "虫量"]
        )
        .properties(width=500, height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("横轴/纵轴 = 田块采样点位置，颜色 = 风险等级（绿低 / 橙中 / 红高）")

# ============ 页面3：采样导航 ============
st.subheader("第三步：推荐下一次采样点")

if st.button("生成采样建议"):
    st.markdown("""
    | 推荐点 | 位置 | 原因 | 优先级 |
    |--------|------|------|--------|
    | 采样点1 | 田块A3东边缘 | 热点边缘，风险最高 | ⭐⭐⭐ |
    | 采样点2 | 田块A1北边 | 7天未采样，数据缺失 | ⭐⭐ |
    | 采样点3 | 道路旁杂草带 | 寄主交界处，易迁入 | ⭐⭐ |
    """)

# ============ 页面4：风险预警 ============
st.subheader("第四步：未来48小时风险")

if st.button("计算风险等级"):
    risk = st.progress(0.65, text="正在计算...")
    time.sleep(1)
    st.success("## 🔶 黄色风险等级\n虫量上升趋势 + 明日高温（32℃）适宜繁殖\n建议：增加采样频率，关注边行，评估局部处理")

# 页面底部
st.divider()
st.caption("棉云哨 · 零硬件烟粉虱监测系统 · 比赛演示版")
st.caption("核心闭环：拍照 → AI计数 → 热点地图 → 采样导航 → 风险预警 → 防控工单")
