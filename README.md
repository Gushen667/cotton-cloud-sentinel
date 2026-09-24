# 棉云哨 - 烟粉虱识别系统
# 项目说明文档

## 一、项目结构

```
D:\AI测试\whitefly\
├── dataset/              # 训练数据（黄板照片）
│   ├── pest_sticky_traps/ # 已下载数据集
│   └── downloaded/       # 下载的文件会放到这里
├── src/                  # 源代码
│   ├── app.py           # Streamlit 演示界面（已跑通）
│   └── train.py          # 模型训练脚本（下一步写）
├── demos/                # 演示材料
├── scripts/              # 辅助脚本
│   └── download_data.py  # 数据集下载脚本
└── data.yaml             # 数据集配置（YOLO训练用）
```

## 二、当前状态

✅ 已完成：
- Python 环境安装（3.14.7）
- 依赖安装（PyTorch + YOLOv8 + Streamlit）
- 演示程序运行正常（http://localhost:8501）

🔄 进行中：
- 数据集下载 → **下一步**
- 模型训练

## 三、数据集来源

### 推荐：Pest Sticky Traps 数据集（Zenodo）
- 链接：https://zenodo.org/records/7801239
- 大小：约 144 MB
- 包含：烟粉虱黄板照片 + 专业标注
- 许可：开放使用，可直接用于比赛

### 备选：Yellow Glue Paper Traps
- 链接：https://zenodo.org/records/7139220
- 包含：225张照片，5904只昆虫标注

## 四、下一步操作

1. 运行 `scripts/download_data.py` 下载数据集
2. 如果下载失败，手动访问 Zenodo 页面下载
3. 数据集准备好后，运行 `src/train.py` 开始训练
4. 训练完成后，用新模型更新演示程序

## 五、注意事项

- 数据集是免费的，用于学术比赛无需额外授权
- 比赛答辩时可以说"使用公开数据集训练"
- 实际田块照片可以后续补充（但比赛演示用公开数据够了）
