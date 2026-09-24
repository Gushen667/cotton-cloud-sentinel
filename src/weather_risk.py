# ============================================================
# 棉云哨 - 气象风险模块（真实数据）
# 调用 Open-Meteo 免费API获取真实天气数据，
# 结合黄板虫量给出风险等级和应对建议
# ============================================================

import requests
from datetime import datetime

# 新疆阿拉尔市典型坐标（南疆棉区）
DEFAULT_COORDS = (40.87, 81.28)  # 阿拉尔市


def get_weather(lat=None, lon=None):
    """获取真实天气（Open-Meteo 免费API，无需密钥）"""
    lat = lat or DEFAULT_COORDS[0]
    lon = lon or DEFAULT_COORDS[1]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
        ],
        "forecast_days": 2,
        "timezone": "Asia/Shanghai",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def wind_summary(weather):
    """
    风速口径修正：分别返回 24h平均风速 和 24h最大风速（含阵风）
    """
    if not weather:
        return None
    hourly = weather.get("hourly", {})
    winds = hourly.get("wind_speed_10m", [])[:24]
    gusts = hourly.get("wind_gusts_10m", [])[:24]
    if not winds:
        return None
    avg = sum(winds) / len(winds)
    peak = max(max(winds), max(gusts) if gusts else 0)
    return {"avg": avg, "peak": peak}


def assess_risk(pest_count, weather=None, crop_stage=None, last_treatment_days=30,
                reliable=True, quality_reason=""):
    """
    风险评估（计划书第七节的公式）：
    风险指数 = 虫量压力×40% + 气象适宜度×30% + 寄主与生育期×20% + 管理因素×10%

    pest_count: 黄板上检出的烟粉虱数量
    weather: get_weather() 的返回值
    crop_stage: 0-1（生育期因子，1=最适生育期，None=默认0.5）
    last_treatment_days: 距上次防治的天数（越长风险越高）

    返回 dict: level(绿/黄/红), score(0-1), detail(说明), actions(建议措施)
    """
    # --- 虫量压力（40%）---
    # 经验阈值：板日虫量 <10 低, 10-30 中, >30 高
    pressure = min(pest_count / 50.0, 1.0)

    # --- 气象适宜度（30%）---
    # 烟粉虱适宜温度21-33℃，高温促进繁殖
    weather_score = 0.5
    if weather:
        temps = weather.get("hourly", {}).get("temperature_2m", [])
        rhums = weather.get("hourly", {}).get("relative_humidity_2m", [])
        if temps:
            avg_temp = sum(temps[:24]) / len(temps[:24])  # 未来24小时平均
            # 21-33℃之间越接近28℃越适宜
            if avg_temp < 21:
                weather_score = max(0.0, (avg_temp - 12) / 9.0 * 0.4)
            elif avg_temp > 33:
                weather_score = max(0.4, 1.0 - (avg_temp - 33) / 10.0)
            else:
                weather_score = 0.4 + 0.6 * (1 - abs(avg_temp - 28) / 7.0)

    # --- 寄主与生育期（20%）---
    crop_factor = 0.5 if crop_stage is None else crop_stage

    # --- 管理因素（10%）---
    # 距上次防治时间越长，虫害反弹风险越高（>14天视为风险高）
    mgmt = min(last_treatment_days / 30.0, 1.0)

    score = (
        pressure * 0.4
        + weather_score * 0.3
        + crop_factor * 0.2
        + mgmt * 0.1
    )
    score = round(min(score, 1.0), 3)

    # 分级
    if not reliable:
        level, advice = "unknown", "暂无法评估：图像质量/识别可信度不足，请人工复核虫量"
    elif score >= 0.6:
        level, advice = "red", "高风险，建议立即联系植保员确认并准备处置"
    elif score >= 0.35:
        level, advice = "yellow", "增加采样频率，关注边行和热点区域，评估局部处理"
    else:
        level, advice = "green", "维持日常监测，暂不需要全田处理"

    return {
        "level": level,
        "score": score,
        "advice": advice,
        "reliable": reliable,
        "quality_reason": quality_reason,
        "breakdown": {
            "虫量压力(40%)": round(pressure, 3),
            "气象适宜度(30%)": round(weather_score, 3),
            "寄主生育期(20%)": round(crop_factor, 3),
            "管理因素(10%)": round(mgmt, 3),
        },
        "pest_count": pest_count,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def next_sampling_hint(count, weather_score=0.5):
    """
    主动采样导航（计划书功能6的简化版）：
    根据当前虫量和气象给出下一次采样方向建议
    """
    hints = []
    if count >= 30:
        hints.append("当前板虫量高，建议扩大采样范围至田块四角 + 中心共5点")
        hints.append("优先沿主导风向的田边方向布点（迁入风险区）")
    elif count >= 10:
        hints.append("中密度，建议维持现有采样点并增加热点周边2个新点")
    else:
        hints.append("低密度，可延长采样间隔至3-5天")

    if weather_score > 0.7:
        hints.append("未来气象条件适宜繁殖，建议把采样间隔缩短至1天")

    return hints
