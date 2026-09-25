# ============================================================
# 棉云哨 - 完整监测系统（最终版 v2）
# 农户上传黄板照片 → 真实YOLO模型识别 → 真实气象数据 → 应对建议
#
# 运行：
#   cd D:\AI测试\whitefly
#   python -m streamlit run src/app_final.py
# ============================================================

import streamlit as st
import io
import os
import sys
import time
import hashlib
import tempfile
from datetime import datetime
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.detect import detect, find_model, draw_detection_overlay
from src.weather_risk import get_weather, assess_risk, next_sampling_hint, wind_summary

st.set_page_config(page_title="棉云哨", layout="wide", page_icon="🐛")

# ============ 移动端适配 CSS ============
st.markdown("""
<style>
/* 手机端缩小标题 */
@media (max-width: 768px) {
    .stApp h1 { font-size: 1.6rem; }
    .stApp h2 { font-size: 1.25rem; }
    .stApp h3 { font-size: 1.1rem; }
}
/* 结论卡片 */
.conclusion-card {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.conclusion-card .cc-item {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    background: #fafafa;
}
.conclusion-card .cc-item .cc-value { font-size: 1.6rem; font-weight: 700; }
.conclusion-card .cc-item .cc-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
@media (max-width: 768px) {
    .conclusion-card { grid-template-columns: repeat(2, 1fr); }
}
/* 工单纵向卡片（移动端） */
@media (max-width: 768px) {
    .stDataFrame { overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)

MODEL_READY = find_model() is not None

# ============ 识别/画框缓存与临时文件管理 ============
# 上传图和叠加图都放系统临时目录，不污染项目根目录和数据集目录
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "mianyunshao_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def cleanup_old_uploads(max_age_hours=48):
    """清掉过期上传临时图，防止无限堆积。"""
    try:
        now = time.time()
        for f in os.listdir(UPLOAD_DIR):
            p = os.path.join(UPLOAD_DIR, f)
            if now - os.path.getmtime(p) > max_age_hours * 3600:
                os.remove(p)
    except Exception:
        pass


cleanup_old_uploads()


@st.cache_data(show_spinner="正在识别虫量（首次约需十几秒）...")
def run_detection(image_path, mtime, conf=0.25):
    """YOLO 识别结果按 图片路径+mtime 缓存：调参数（尺寸/天数/地区）不会重跑推理。"""
    return detect(image_path, conf=conf)


@st.cache_data(show_spinner=False)
def build_overlay(image_path, mtime, boxes):
    """检测框叠加图同样缓存，避免重复画框写盘。"""
    return draw_detection_overlay(image_path, boxes)

# ============ 会话管理函数（必须在侧栏前定义）============
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

def load_sessions():
    if not os.path.exists(HISTORY_FILE):
        return {"sessions": []}
    import json
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sessions": []}

def save_sessions(data):
    import json
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def new_session_id():
    import time
    return f"S_{time.time_ns()}"


def session_key(name):
    """让上传控件和表单状态跟随当前会话隔离。"""
    return f"{name}_{st.session_state.current_session}"


def start_new_session():
    """创建空会话并清空当前工作区，页面随后完整重跑。"""
    new_id = new_session_id()
    create_session(new_id)
    st.session_state.current_session = new_id
    st.session_state.pop("uploaded_example", None)
    return new_id

if "current_session" not in st.session_state:
    st.session_state.current_session = new_session_id()

def session_label(s):
    cur = " ★当前" if s["id"] == st.session_state.current_session else ""
    return f'{s.get("name", "未命名")}（{len(s.get("records", []))}条）{cur}'

def delete_session(sid):
    data = load_sessions()
    data["sessions"] = [s for s in data["sessions"] if s.get("id") != sid]
    save_sessions(data)

def rename_session(sid, new_name):
    data = load_sessions()
    for s in data["sessions"]:
        if s.get("id") == sid:
            s["name"] = new_name
            break
    save_sessions(data)

def clear_all_sessions():
    save_sessions({"sessions": []})

def create_session(sid):
    data = load_sessions()
    if not any(s.get("id") == sid for s in data["sessions"]):
        data["sessions"].append({
            "id": sid,
            "name": f"会话 {len(data['sessions'])+1}",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "records": [],
        })
        save_sessions(data)

def record_to_current_session(record):
    data = load_sessions()
    sessions = data["sessions"]
    target = None
    for s in sessions:
        if s.get("id") == st.session_state.current_session:
            target = s
            break
    if target is None:
        target = {"id": st.session_state.current_session,
                  "name": f"会话 {len(sessions)+1}",
                  "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                  "records": []}
        sessions.append(target)
    target["records"].append(record)
    if len(target["records"]) > 100:
        target["records"] = target["records"][-100:]
    data["sessions"] = sessions
    save_sessions(data)


# ============ 侧栏：会话管理（必须在标题前执行，否则切换不生效）============
with st.sidebar:
    st.header("📋 监测会话")
    if st.button("＋ 新建会话", use_container_width=True):
        start_new_session()
        st.rerun()

    data_all = load_sessions()
    sessions = data_all["sessions"]

    if sessions:
        labels_list = [session_label(s) for s in sessions]
        cur_idx = next((i for i, s in enumerate(sessions)
                        if s["id"] == st.session_state.current_session),
                       len(sessions) - 1)
        chosen = st.selectbox("监测会话", labels_list, index=cur_idx)
        chosen_idx = labels_list.index(chosen)
        target = sessions[chosen_idx]

        # 切换会话：直接改 current_session 并 rerun（选下拉框就切，不用点按钮）
        if target["id"] != st.session_state.current_session:
            st.session_state.current_session = target["id"]
            st.rerun()

        # 重命名
        st.markdown("---")
        new_name = st.text_input("会话名称（可改）", value=target.get("name", "未命名"),
                                 key=f"rename_{target['id']}")
        if st.button("保存名称") and new_name.strip() and new_name.strip() != target.get("name"):
            rename_session(target["id"], new_name.strip())
            st.rerun()

        # 删除
        if st.button("🗑 删除该会话", use_container_width=False):
            delete_session(target["id"])
            if target["id"] == st.session_state.current_session:
                start_new_session()
            st.rerun()
    else:
        st.caption("暂无会话。点上方按钮新建。")

    # 清空全部
    st.markdown("---")
    if st.button("清空全部会话", use_container_width=True):
        clear_all_sessions()
        start_new_session()
        st.rerun()


st.title("🐛 棉云哨")
st.caption("烟粉虱智能监测与防控决策系统 · 黄板拍照识别 · 气象联动 · 零硬件方案")

if MODEL_READY:
    st.success("✅ 识别服务正常（YOLO模型已加载，真实训练于本地黄板数据）")
else:
    st.warning("⏳ 识别模型尚未就绪，气象评估功能可正常使用。")

# ============ 示例照片（一键演示） ============
DATASET_IMG = os.path.join(BASE_DIR, "dataset/yellow_sticky_traps/images")
EXAMPLES = {
    "低虫量示例（约12只）": "1187.jpg",
    "中虫量示例（约44只）": "1004.jpg",
    "高虫量示例（约228只）": "1252.jpg",
}
EXISTING_EXAMPLES = {k: v for k, v in EXAMPLES.items()
                     if os.path.exists(os.path.join(DATASET_IMG, v))}

# ============ 气象数据 ============
@st.cache_data(ttl=3600, show_spinner="正在获取真实气象数据...")
def load_weather(lat, lon):
    try:
        return get_weather(lat=lat, lon=lon), True
    except Exception as e:
        return str(e), False


def get_photo_time(image_path, source_name):
    """
    拍摄时间：
    - 示例照片 → 读照片 EXIF 里的实际拍摄时间（别人的照片，不能显示"现在"）
    - 用户上传 → 当前时间
    读不到 EXIF 时用文件名/目录信息兜底，实在没有就标"原图时间未知"
    """
    if source_name != "示例照片":
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    # 尝试读 EXIF
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(image_path) as im:
            exif = im.getexif()
            for tag, val in exif.items():
                if TAGS.get(tag) == "DateTime":
                    return str(val)
    except Exception:
        pass
    # 兜底：用文件修改时间（原始数据集的时间）
    try:
        mtime = os.path.getmtime(image_path)
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") + "（原图记录时间）"
    except Exception:
        return "原图时间未知"


def run_pipeline(image_path, source_name):
    """完整流程：识别 → 气象 → 风险 → 输出，返回各部分数据"""
    import numpy as np
    sid = st.session_state.current_session
    image_key = os.path.basename(image_path)
    # --- 1. AI 识别（带缓存：同图不重复推理） ---
    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        mtime = 0.0
    detection = run_detection(image_path, mtime, 0.25)
    overlay = None
    if detection["boxes"]:
        overlay = build_overlay(image_path, mtime, detection["boxes"])
    elif detection["found"]:
        overlay = build_overlay(image_path, mtime, [])  # 显示"未检出目标"

    # --- 2. 气象 + 地区（用 session_state 记住上次选择） ---
    preset_coords = {
        "阿拉尔（默认）": (40.87, 81.28),
        "喀什": (39.47, 75.99),
        "阿克苏": (41.17, 80.26),
        "和田": (37.12, 79.92),
    }

    # --- 3. 标准化虫量字段 ---
    field_name = st.text_input(
        "田块名称", value="演示田块",
        help="将写入防控工单",
        key=session_key("field_name"),
    )
    trap_size = st.selectbox(
        "黄板尺寸",
        ["15×15cm", "20×20cm", "30×40cm", "其他"],
        key=session_key("trap_size"),
    )
    hang_days = st.number_input(
        "挂板天数（放置到拍摄）",
        min_value=1,
        value=3,
        key=session_key("hang_days"),
    )
    shot_time = st.text_input(
        "拍摄时间",
        value=get_photo_time(image_path, source_name),
        key=f"{session_key('shot_time')}_{image_key}",
    )
    last_treat = st.number_input(
        "距上次防治天数",
        min_value=0,
        value=14,
        step=1,
        key=session_key("last_treat"),
    )

    # 人工修正（保留AI检出值作为默认）
    detected_count = detection["count"]
    manual = st.number_input(
        "人工复核虫量（可与AI对比）", min_value=0, value=detected_count,
        help="AI漏检/误检时可手动修正，工单以复核值为准",
        key=f"{session_key('manual_count')}_{image_key}",
    )
    final_count = manual

    # --- 4. 气象 ---
    region_choice = st.selectbox(
        "监测地区（气象数据所在地）",
        list(preset_coords.keys()) + ["自定义坐标"],
        key=session_key("region_choice"),
    )
    if region_choice == "自定义坐标":
        c1, c2 = st.columns(2)
        custom_lat = c1.number_input(
            "纬度（南疆棉区约 37-41）",
            min_value=35.0,
            max_value=45.0,
            value=40.5,
            key=session_key("custom_lat"),
        )
        custom_lon = c2.number_input(
            "经度（南疆约 75-85）",
            min_value=70.0,
            max_value=90.0,
            value=80.0,
            key=session_key("custom_lon"),
        )
        lat, lon = custom_lat, custom_lon
    else:
        lat, lon = preset_coords[region_choice]

    wdata, wok = load_weather(lat, lon)
    weather_lines = []
    if wok:
        temps = wdata.get("hourly", {}).get("temperature_2m", [])
        rhums = wdata.get("hourly", {}).get("relative_humidity_2m", [])
        ws = wind_summary(wdata)
        today_temp = f"{np.mean(temps[:24]):.1f}℃" if temps else "-"
        today_rhum = f"{np.mean(rhums[:24]):.0f}%" if rhums else "-"
        today_wind_avg = f"{ws['avg']:.1f}" if ws else "-"
        today_wind_peak = f"{ws['peak']:.1f}" if ws else "-"
        weather_lines = [
            f"**{region_choice} 未来24小时气象（真实数据 · Open-Meteo API · 坐标 {lat},{lon}）：**",
            f"- 平均气温：{today_temp}",
            f"- 平均湿度：{today_rhum}",
            f"- 风速：24h平均 {today_wind_avg} m/s，峰值（含阵风） {today_wind_peak} m/s",
            "",
            "> 💡 风速 >5 m/s 抑制烟粉虱迁飞扩散；大风（>8 m/s）影响采样与施药作业窗口",
        ]
    else:
        st.error(f"气象数据获取失败：{wdata}")
        wdata = None

    # --- 5. 风险评估（含可信度判断） ---
    q = detection["quality"]
    reliable = detection["reliable"]
    q_reason = ""
    if not q["ok"]:
        q_reason = q["reason"]
    risk = assess_risk(
        final_count, weather=wdata, last_treatment_days=last_treat,
        reliable=reliable, quality_reason=q_reason
    )

    # --- 6. 结论卡片（首屏四大数字） ---
    emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢", "unknown": "⚪"}[risk["level"]]
    level_name = {"red": "高风险", "yellow": "中等风险", "green": "低风险",
                  "unknown": "暂无法评估"}[risk["level"]]
    next_action = {
        "red": "立即处置", "yellow": "加密监测",
        "green": "维持监测", "unknown": "人工复核"
    }[risk["level"]]

    st.markdown(f"""
    <div class="conclusion-card">
        <div class="cc-item"><div class="cc-value">{final_count}</div>
        <div class="cc-label">当前虫量（只/板/{hang_days}天）</div></div>
        <div class="cc-item"><div class="cc-value">{detection['confidence_level']}</div>
        <div class="cc-label">识别可信度</div></div>
        <div class="cc-item"><div class="cc-value">{emoji}</div>
        <div class="cc-label">{level_name}</div></div>
        <div class="cc-item"><div class="cc-value">{next_action}</div>
        <div class="cc-label">下一步行动</div></div>
    </div>
    """, unsafe_allow_html=True)

    # --- 7. 识别结果图（原图 + 叠加检测框） ---
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**原图**（{source_name}）")
        st.image(image_path, use_container_width=True)
    with col2:
        st.write("**AI识别结果**（检测框 + 置信度）")
        if overlay:
            st.image(overlay, use_container_width=True)
        else:
            st.info("识别失败或图片无法读取")

    # 图像质量说明
    if not q["ok"]:
        st.warning(f"⚠️ 图像质量提示：{q['reason']}（清晰度 {q['sharpness']:.0f} / 亮度 {q['brightness']:.0f} / 黄板占比 {q['yellow_ratio']*100:.0f}%）")
        if final_count == 0:
            st.error("模型未检出目标 且 图像质量不可靠 → **暂无法判定低风险**，请人工复核后填入真实虫量。")

    # 可信度详情
    st.caption(f"识别置信度：{detection['confidence']:.2f}（{detection['confidence_level']}） · "
               f"图像清晰度 {q['sharpness']:.0f} · 亮度 {q['brightness']:.0f} · 黄板占比 {q['yellow_ratio']*100:.0f}%")

    # --- 8. 气象展示 ---
    st.subheader("气象数据与风险评估")
    for line in weather_lines:
        st.markdown(line)

    risk_detail_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢", "unknown": "⚪"}[risk["level"]]
    # 不可信时风险指数没有意义，显示 "--" 而不是误导性的数字
    score_txt = f"{risk['score']:.2f}" if risk["level"] != "unknown" else "--"
    st.markdown(f"### {risk_detail_emoji} {level_name}（风险指数 {score_txt}）")
    st.write(risk["advice"])
    if risk.get("quality_reason"):
        st.caption(f"可信度说明：{risk['quality_reason']}")

    with st.expander("风险指数构成（透明可解释）"):
        for k, v in risk["breakdown"].items():
            st.write(f"- {k}: {v}")

    # --- 9. 采样导航 ---
    st.subheader("下次采样建议")
    weather_score = risk["weather_score"]
    for h in next_sampling_hint(final_count, weather_score):
        st.write(f"- {h}")

    # --- 10. 防控工单（移动端友好：卡片式） ---
    st.subheader("防控工单")
    trap_area = {"15×15cm": "15×15cm", "20×20cm": "20×20cm",
                 "30×40cm": "30×40cm", "其他": "未指定"}[trap_size]
    st.markdown(f"""
    **工单摘要（{risk['timestamp']}）**

    - 田块：{field_name}
    - 黄板尺寸：{trap_area}
    - 挂板天数：{hang_days} 天
    - 拍摄时间：{shot_time}
    - 当前虫量：**{final_count} 头/板/{hang_days}天**（AI检出 {detected_count} 只）
    - 风险等级：{level_name}
    - 距上次防治：{last_treat} 天
    - 建议措施：{risk['advice']}
    - 复核时间：处理后 3-7 天
    """)
    if st.button("生成工单记录", key=session_key("generate_order")):
        if source_name != "示例照片":
            record_to_current_session({
                "time": risk["timestamp"],
                "count": final_count,
                "level": risk["level"],
                "hang_days": hang_days,
                "ai_count": detected_count,
                "confidence": detection["confidence_level"],
                "region": region_choice,
                "advice": risk["advice"],
            })
            st.success("工单已生成并记录。处理后 3-7 天系统提醒复核。")
            st.caption("✅ 本次识别已存档至当前会话")
        else:
            st.info("本次为示例照片，仅供演示，不生成工单记录。上传自己的照片后即可存档。")
    elif source_name == "示例照片":
        st.caption("本次为示例照片，仅供演示，不计入历史趋势。")
    # 当前会话趋势
    cur = None
    for s in load_sessions()["sessions"]:
        if s["id"] == st.session_state.current_session:
            cur = s
    if cur and len(cur["records"]) >= 2:
        st.subheader(f"历史趋势（当前会话：{len(cur['records'])} 条记录，已保存至 history.json）")
        import pandas as pd
        df = pd.DataFrame(cur["records"])
        st.line_chart(df.set_index("time")["count"])
        last_growth = df["count"].iloc[-1] - df["count"].iloc[-2]
        st.caption(f"最近两次增长 {last_growth:+d} 只/板/{hang_days}天")


# ============ 主界面 ============
st.subheader("第一步：上传黄板照片 或 一键示例")

sid = st.session_state.current_session
example_key = session_key("uploaded_example")
uploader_key = session_key("uploaded_file")

# 一键示例按钮
ex_cols = st.columns(len(EXISTING_EXAMPLES))
for i, (label, fname) in enumerate(EXISTING_EXAMPLES.items()):
    if ex_cols[i].button(label, key=session_key(f"example_{i}")):
        example_path = os.path.join(DATASET_IMG, fname)
        st.session_state.pop(uploader_key, None)
        st.session_state[example_key] = example_path
        st.rerun()

uploaded = st.file_uploader("或上传自己的黄板照片（JPG/PNG，建议黄板占画面80%）",
                            type=["jpg", "jpeg", "png"],
                            key=uploader_key)

active_img = None
active_src = ""

if uploaded:
    img_bytes = uploaded.getvalue()
    digest = hashlib.sha1(img_bytes).hexdigest()[:12]
    tmp_path = os.path.join(UPLOAD_DIR, f"upload_{sid}_{digest}.jpg")
    with open(tmp_path, "wb") as f:
        f.write(img_bytes)
    st.session_state[example_key] = None
    active_img = tmp_path
    active_src = f"上传：{uploaded.name}"
elif st.session_state.get(example_key):
    active_img = st.session_state[example_key]
    active_src = "示例照片"

if active_img:
    st.divider()
    run_pipeline(active_img, active_src)

st.divider()
st.caption("棉云哨 · 零硬件烟粉虱监测系统 · AI识别基于本地真实黄板数据训练 · 气象数据来自 Open-Meteo 公开API")
