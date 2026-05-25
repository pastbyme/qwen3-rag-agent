"""
评估结果可视化模块
生成准确率柱状图、响应时间柱状图、雷达图、综合仪表盘
"""
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# 字体设置
_FONT_CANDIDATES = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                    "Noto Sans CJK SC", "DejaVu Sans"]
_FONT_NAME = "DejaVu Sans"
for font_name in _FONT_CANDIDATES:
    try:
        font_path = fm.findfont(font_name, fallback_to_default=False)
        if font_path:
            _FONT_NAME = font_name
            break
    except Exception:
        continue
plt.rcParams["font.sans-serif"] = [_FONT_NAME, "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# ======================================


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _get_attr(obj, attr: str, default=None):
    """兼容 dict / 对象 / dataclass 的属性获取"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    val = getattr(obj, attr, None)
    if val is not None:
        return val
    if hasattr(obj, "__dataclass_fields__") and attr in obj.__dataclass_fields__:
        return getattr(obj, attr, default)
    return default


def _inspect_object(obj, label="result"):
    """调试：打印对象类型和属性（WARNING 级别确保总是显示）"""
    attrs = []
    if isinstance(obj, dict):
        attrs = list(obj.keys())
    else:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
    logger.warning(f"[{label}] 类型={type(obj).__name__}, 属性={attrs}")


def generate_charts_from_result(result, prefix: str = "./results/eval_report"):
    """
    从评估结果生成所有图表
    """
    _inspect_object(result, "EvaluationResult")

    # 从 detail 中精准提取基于分类的正确结构
    detail = _get_attr(result, "detail", {})
    acc_by_type = detail.get("accuracy_by_type", {})
    time_by_type = detail.get("response_time_by_type", {})

    if not acc_by_type:
        logger.warning("未找到分类数据，请检查数据源格式。")
        return

    categories = list(acc_by_type.keys())
    accuracy_data = []
    time_data = []

    for cat in categories:
        # metrics.py 保存的字典格式为 {"mean": val, "std": val, "count": val}
        acc = acc_by_type[cat].get("mean", 0) * 100
        t = time_by_type.get(cat, {}).get("mean", 0)
        accuracy_data.append(acc)
        time_data.append(t)


    # 生成图表
    _plot_accuracy(categories, accuracy_data, f"{prefix}_accuracy")
    _plot_response_time(categories, time_data, f"{prefix}_time")
    _plot_radar(categories, accuracy_data, time_data, f"{prefix}_radar")
    _plot_dashboard(result, categories, accuracy_data, time_data, f"{prefix}_dashboard")


def _plot_accuracy(categories: List[str], accuracy_data: List[float],
                   filepath: str):
    """准确率柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50"] * len(categories)
    bars = ax.bar(categories, accuracy_data, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, accuracy_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title("各问题类型准确率", fontsize=14, fontweight="bold")
    avg = np.mean(accuracy_data)
    ax.axhline(y=avg, color="red", linestyle="--", linewidth=1,
               label=f"平均: {avg:.1f}%")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _ensure_dir(filepath)
    plt.tight_layout()
    plt.savefig(f"{filepath}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[准确率柱状图] 已保存 -> {filepath}.png")


def _plot_response_time(categories: List[str], time_data: List[float],
                        filepath: str):
    """响应时间柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF9800"] * len(categories)
    bars = ax.bar(categories, time_data, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, time_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("响应时间 (秒)", fontsize=12)
    ax.set_xlabel("问题类型", fontsize=12)
    ax.set_title("各问题类型响应时间对比", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _ensure_dir(filepath)
    plt.tight_layout()
    plt.savefig(f"{filepath}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[响应时间柱状图] 已保存 -> {filepath}.png")


def _plot_radar(categories: List[str], accuracy_data: List[float],
                time_data: List[float], filepath: str):
    """雷达图"""
    max_time = max(time_data) if time_data else 1
    time_norm = [(1 - t / max_time) * 100 for t in time_data]
    labels = categories
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    acc_plot = accuracy_data + accuracy_data[:1]
    ax.plot(angles, acc_plot, "o-", linewidth=2, label="准确率", color="#4CAF50")
    ax.fill(angles, acc_plot, alpha=0.15, color="#4CAF50")
    tm_plot = time_norm + time_norm[:1]
    ax.plot(angles, tm_plot, "o-", linewidth=2, label="响应速度 (归一化)", color="#FF9800")
    ax.fill(angles, tm_plot, alpha=0.15, color="#FF9800")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("综合能力雷达图", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=10)
    ax.grid(True)
    _ensure_dir(filepath)
    plt.tight_layout()
    plt.savefig(f"{filepath}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[雷达图] 已保存 -> {filepath}.png")


def _plot_dashboard(result, categories: List[str],
                    accuracy_data: List[float], time_data: List[float],
                    filepath: str):
    """综合仪表盘"""
    # 使用 EvaluationResult 类中真正定义的属性名称
    overall_accuracy = _get_attr(result, "accuracy", 0) * 100
    avg_response_time = _get_attr(result, "avg_response_time", 0)
    avg_confidence = _get_attr(result, "avg_confidence", 0) * 100
    tool_accuracy = _get_attr(result, "tool_calling_accuracy", 0) * 100

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.25)

    metrics = [
        ("总体准确率", f"{overall_accuracy:.1f}%", "#4CAF50"),
        ("平均响应时间", f"{avg_response_time:.1f}s", "#FF9800"),
        ("平均置信度", f"{avg_confidence:.1f}%", "#2196F3"),
        ("工具调用准确率", f"{tool_accuracy:.1f}%", "#9C27B0"),
    ]

    for idx, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                   facecolor=color, alpha=0.1,
                                   edgecolor=color, linewidth=2))
        ax.text(0.5, 0.65, value, ha="center", va="center",
                fontsize=28, fontweight="bold", color=color)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=13, color="gray")

    # 下排左侧: 准确率
    ax = fig.add_subplot(gs[1, :2])
    colors_acc = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax.bar(categories, accuracy_data, color=colors_acc, width=0.6,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, accuracy_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("各问题类型准确率", fontsize=13, fontweight="bold")
    avg = np.mean(accuracy_data)
    ax.axhline(y=avg, color="red", linestyle="--", linewidth=1,
               label=f"平均: {avg:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 下排右侧: 响应时间
    ax = fig.add_subplot(gs[1, 2:])
    colors_time = plt.cm.plasma(np.linspace(0.3, 0.9, len(categories)))
    bars = ax.bar(categories, time_data, color=colors_time, width=0.6,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, time_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("响应时间 (秒)", fontsize=11)
    ax.set_xlabel("问题类型", fontsize=11)
    ax.set_title("各问题类型响应时间对比", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _ensure_dir(filepath)
    plt.tight_layout()
    plt.savefig(f"{filepath}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[综合仪表盘] 已保存 -> {filepath}.png")