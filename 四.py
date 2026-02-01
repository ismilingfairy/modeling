import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体 (防止乱码，如果在纯英文环境请注释掉)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_outputs():
    # --- 模拟数据 ---
    strategies = ['策略A: 固定票价', '策略B: 分档定价', '策略C: H2D2动态定价']
    revenue = [14000000, 18500000, 22800000]  # 模拟的赛季总收入 ($)
    attendance_rate = [82.5, 88.0, 98.5]  # 平均上座率 (%)

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 柱状图 (收入)
    x = np.arange(len(strategies))
    width = 0.4
    bars = ax1.bar(x, revenue, width, color=['#95a5a6', '#3498db', '#e74c3c'], alpha=0.9, label='总收入 ($)')

    # 设置左轴
    ax1.set_ylabel('赛季总收入 (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('产出物 3: 三种定价策略的赛季总收益与上座率对比', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=11)
    ax1.set_ylim(0, 26000000)

    # 标注收入金额
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 500000,
                 f'${height / 1000000:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 折线图 (上座率)
    ax2 = ax1.twinx()
    line = ax2.plot(x, attendance_rate, color='#2ecc71', marker='o', linewidth=3, markersize=8, label='平均上座率 (%)')
    ax2.set_ylabel('平均上座率 (%)', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2.set_ylim(50, 110)

    # 标注上座率
    for i, v in enumerate(attendance_rate):
        ax2.text(i, v + 3, f'{v}%', ha='center', va='bottom', color='#2ecc71', fontweight='bold')

    # 图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_outputs()