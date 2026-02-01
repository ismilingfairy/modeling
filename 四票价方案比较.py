import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_pricing_metrics():
    """
    数据计算逻辑保持不变
    """
    # 基础参数
    arena_capacity = 18000
    revenue_share_ratio = 0.8
    game_levels = {"热门": 10, "常规": 20, "冷门": 10}
    levels = ["热门", "常规", "冷门"]
    total_games = sum(game_levels.values())

    # ====================== 策略A ======================
    a_fixed_price = 80
    a_estimate_rate = {"热门": 0.7, "常规": 0.7, "冷门": 0.7}
    a_actual_rate = {"热门": 0.9, "常规": 0.7, "冷门": 0.5}
    a_total_revenue, a_avg_attendance = 0, 0
    for level, game_count in game_levels.items():
        single_rev = a_fixed_price * arena_capacity * a_actual_rate[level]
        a_total_revenue += single_rev * game_count * revenue_share_ratio
        a_avg_attendance += a_actual_rate[level] * game_count
    a_avg_attendance_rate = (a_avg_attendance / total_games) * 100
    a_avg_price = a_fixed_price

    # ====================== 策略B ======================
    b_level_price = {"热门": 120, "常规": 85, "冷门": 50}
    b_estimate_rate = {"热门": 0.9, "常规": 0.7, "冷门": 0.5}
    b_actual_rate = {"热门": 0.95, "常规": 0.75, "冷门": 0.55}
    b_total_revenue, b_avg_attendance = 0, 0
    for level, game_count in game_levels.items():
        single_rev = b_level_price[level] * arena_capacity * b_actual_rate[level]
        b_total_revenue += single_rev * game_count * revenue_share_ratio
        b_avg_attendance += b_actual_rate[level] * game_count
    b_avg_attendance_rate = (b_avg_attendance / total_games) * 100
    b_avg_price = (b_level_price["热门"] * 10 + b_level_price["常规"] * 20 + b_level_price["冷门"] * 10) / total_games

    # ====================== 策略C ======================
    c_base_price = {"热门": 120, "常规": 85, "冷门": 50}
    c_discount_rate = 0.7
    c_discount_price = {
        "热门": round(c_base_price["热门"] * c_discount_rate, 1),
        "常规": round(c_base_price["常规"] * c_discount_rate, 1),
        "冷门": round(c_base_price["冷门"] * c_discount_rate, 1)
    }
    c_estimate_rate = {"热门": 0.9, "常规": 0.7, "冷门": 0.5}
    c_base_rate = b_actual_rate
    c_actual_rate = {"热门": 0.98, "常规": 0.9, "冷门": 0.85}
    c_discount_rate_sale = {level: c_actual_rate[level] - c_base_rate[level] for level in levels}

    c_total_revenue, c_avg_attendance = 0, 0
    for level, game_count in game_levels.items():
        base_rev = c_base_price[level] * arena_capacity * c_base_rate[level]
        discount_rev = c_discount_price[level] * arena_capacity * c_discount_rate_sale[level]
        single_rev = base_rev + discount_rev
        c_total_revenue += single_rev * game_count * revenue_share_ratio
        c_avg_attendance += c_actual_rate[level] * game_count
    c_avg_attendance_rate = (c_avg_attendance / total_games) * 100

    c_level_avg_price = {}
    for level in levels:
        total_rate = c_actual_rate[level]
        if total_rate == 0:
            c_level_avg_price[level] = 0
        else:
            c_level_avg_price[level] = round(
                (c_base_price[level] * c_base_rate[level] + c_discount_price[level] * c_discount_rate_sale[level])
                / total_rate, 1)

    c_total_tickets = 0
    c_total_price = 0
    for level in levels:
        base_tickets = c_base_rate[level] * arena_capacity * game_levels[level]
        discount_tickets = c_discount_rate_sale[level] * arena_capacity * game_levels[level]
        c_total_tickets += base_tickets + discount_tickets
        c_total_price += (base_tickets * c_base_price[level]) + (discount_tickets * c_discount_price[level])
    c_avg_price = round(c_total_price / c_total_tickets, 1)

    # 整理数据
    strategies = ['策略A: 固定金额', '策略B: 分层定价', '策略C: 动态调价']
    total_revenue = [a_total_revenue, b_total_revenue, c_total_revenue]
    total_attendance_rate = [a_avg_attendance_rate, b_avg_attendance_rate, c_avg_attendance_rate]
    avg_price_list = [a_avg_price, b_avg_price, c_avg_price]

    estimate_rates = {
        "策略A: 固定金额": [a_estimate_rate[level] for level in levels],
        "策略B: 分层定价": [b_estimate_rate[level] for level in levels],
        "策略C: 动态调价": [c_estimate_rate[level] for level in levels]
    }
    actual_rates = {
        "策略A: 固定金额": [a_actual_rate[level] for level in levels],
        "策略B: 分层定价": [b_actual_rate[level] for level in levels],
        "策略C: 动态调价": [c_actual_rate[level] for level in levels]
    }
    c_base_rates = [c_base_rate[level] for level in levels]
    c_discount_rates = [c_discount_rate_sale[level] for level in levels]
    price_data = {
        "策略A: 固定金额": [a_fixed_price for _ in levels],
        "策略B: 分层定价": [b_level_price[level] for level in levels],
        "策略C: 动态调价-原价": [c_base_price[level] for level in levels],
        "策略C: 动态调价-折扣价": [c_discount_price[level] for level in levels],
        "策略C: 各层级平均票价": [c_level_avg_price[level] for level in levels]
    }

    return (strategies, total_revenue, total_attendance_rate, levels, estimate_rates,
            actual_rates, price_data, avg_price_list, c_base_rates, c_discount_rates)


def generate_outputs():
    (strategies, total_revenue, total_attendance_rate, levels, estimate_rates,
     actual_rates, price_data, avg_price_list, c_base_rates, c_discount_rates) = calculate_pricing_metrics()

    # 创建图表，增加整体高度以便容纳更多信息
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), gridspec_kw={'height_ratios': [2, 3]})
    fig.suptitle('三种售票机制对比（含原价/折扣拆分+层级平均票价）', fontsize=20, fontweight='bold', y=0.98)

    # ====================== 上区域：总收入(柱) + 上座率(线) + 均价(线) ======================
    x = np.arange(len(strategies))
    width = 0.35

    # 1. 左轴：赛季总收入 (柱状图)
    bars_rev = ax1.bar(
        x, total_revenue, width,
        color=['#7f8c8d', '#2980b9', '#e74c3c'],
        alpha=0.8, label='赛季总收入（分成后）'
    )

    # 【核心调整1】：显著提高左轴上限，将柱子“压”到下方，为上方的折线腾出空间，防止遮挡
    # 之前是 * 1.2，现在改为 * 1.6
    ax1.set_ylim(0, max(total_revenue) * 1.6)

    # 柱状图注释
    for bar in bars_rev:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height + 100000,
            f'${height / 1000000:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    # 2. 右轴1：总体上座率 (绿色折线)
    ax1_twin1 = ax1.twinx()  # twinx默认就在右侧
    line_att = ax1_twin1.plot(
        x, total_attendance_rate,
        color='#27ae60', marker='o', linewidth=4, markersize=10, label='总体上座率'
    )

    # 设置右轴1 (绿色) 的范围，使其线条处于上方区域
    ax1_twin1.set_ylim(0, 115)

    # 上座率注释 (位置微调)
    for i, v in enumerate(total_attendance_rate):
        ax1_twin1.text(
            i, v + 2, f'{v:.1f}%',
            ha='center', va='bottom', color='#27ae60', fontsize=11, fontweight='bold'
        )

    # 3. 右轴2：平均票价 (橙色折线)
    ax1_twin2 = ax1.twinx()  # 再次twinx，依然在右侧
    # 【核心调整2】：将右轴2向右偏移，避免与右轴1重叠
    ax1_twin2.spines['right'].set_position(('outward', 60))

    line_price = ax1_twin2.plot(
        x, avg_price_list,
        color='#f39c12', marker='s', linewidth=4, markersize=10, label='策略整体平均票价'
    )

    # 设置右轴2 (橙色) 的范围，使其线条处于上方区域
    ax1_twin2.set_ylim(0, max(avg_price_list) * 1.4)

    # 平均票价注释
    for i, v in enumerate(avg_price_list):
        ax1_twin2.text(
            i, v + 2, f'${v:.1f}',
            ha='center', va='bottom', color='#f39c12', fontsize=11, fontweight='bold'
        )

    # === 上区域样式设置 ===
    # 左轴设置
    ax1.set_ylabel('赛季总收入 (USD)', fontsize=12, fontweight='bold')

    # 右轴1 (绿色) 设置 - 明确指定分度值在右侧
    ax1_twin1.set_ylabel('总体上座率 (%)', fontsize=12, fontweight='bold', color='#27ae60')
    ax1_twin1.tick_params(axis='y', labelcolor='#27ae60', right=True, labelright=True, left=False, labelleft=False)

    # 右轴2 (橙色) 设置
    ax1_twin2.set_ylabel('策略整体平均票价 (USD)', fontsize=12, fontweight='bold', color='#f39c12')
    ax1_twin2.tick_params(axis='y', labelcolor='#f39c12')

    # X轴设置
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=11)

    # 网格线 (仅横向)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin1.get_legend_handles_labels()
    lines3, labels3 = ax1_twin2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=10)

    # ====================== 下区域：分层售票率 (保持黑色注释优化) ======================
    x_level = np.arange(len(levels))
    bar_width = 0.15
    offsets = [-0.25, 0, 0.25]
    colors_estimate = ['#95a5a6', '#3498db', '#e74c3c']
    colors_actual = ['#5d6d7e', '#21618c', '#c0392b']
    color_c_base = '#8e44ad'
    color_c_discount = '#1abc9c'

    # 策略A/B
    for i, strategy in enumerate(strategies[:2]):
        # 预估
        ax2.bar(
            x_level + offsets[i] - bar_width / 2, estimate_rates[strategy],
            bar_width, color=colors_estimate[i], alpha=0.8, label=f'{strategy} - 预估'
        )
        # 实际
        ax2.bar(
            x_level + offsets[i] + bar_width / 2, actual_rates[strategy],
            bar_width, color=colors_actual[i], alpha=0.8, label=f'{strategy} - 实际'
        )

    # 策略C
    c_idx = 2
    # 预估
    ax2.bar(
        x_level + offsets[c_idx] - bar_width / 2, estimate_rates[strategies[c_idx]],
        bar_width, color=colors_estimate[c_idx], alpha=0.8, label=f'{strategies[c_idx]} - 预估'
    )
    # 实际 (堆叠)
    ax2.bar(
        x_level + offsets[c_idx] + bar_width / 2, c_base_rates,
        bar_width, color=color_c_base, alpha=0.9, label=f'{strategies[c_idx]} - 原价出售'
    )
    ax2.bar(
        x_level + offsets[c_idx] + bar_width / 2, c_discount_rates,
        bar_width, color=color_c_discount, alpha=0.9, label=f'{strategies[c_idx]} - 折扣出售',
        bottom=c_base_rates
    )

    # 策略A/B 注释
    for i, strategy in enumerate(strategies[:2]):
        for j, level in enumerate(levels):
            est_val = estimate_rates[strategy][j]
            ax2.text(j + offsets[i] - bar_width / 2, est_val + 0.02, f'{est_val * 100:.0f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

            act_val = actual_rates[strategy][j]
            ax2.text(j + offsets[i] + bar_width / 2, act_val + 0.02, f'{act_val * 100:.0f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

            price_val = price_data[strategy][j]
            ax2.text(j + offsets[i], -0.1, f'票价${price_val}', ha='center', va='top', fontsize=7, color='#666666',
                     fontweight='bold')

    # 策略C 注释 (黑色字体)
    for j, level in enumerate(levels):
        est_val = estimate_rates[strategies[c_idx]][j]
        ax2.text(j + offsets[c_idx] - bar_width / 2, est_val + 0.02, f'{est_val * 100:.0f}%', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

        base_val = c_base_rates[j]
        base_price = price_data["策略C: 动态调价-原价"][j]
        ax2.text(j + offsets[c_idx] + bar_width / 2, base_val / 2, f'{base_val * 100:.0f}%\n原价${base_price}',
                 ha='center', va='center', fontsize=7, color='black', fontweight='bold')

        discount_val = c_discount_rates[j]
        if discount_val > 0:
            discount_price = price_data["策略C: 动态调价-折扣价"][j]
            ax2.text(j + offsets[c_idx] + bar_width / 2, base_val + discount_val / 2,
                     f'{discount_val * 100:.0f}%\n折扣${discount_price}', ha='center', va='center', fontsize=7,
                     color='black', fontweight='bold')

        c_level_avg_price = price_data["策略C: 各层级平均票价"][j]
        ax2.text(j + offsets[c_idx], -0.1, f'层级均价${c_level_avg_price}', ha='center', va='top', fontsize=7,
                 color='#333333', fontweight='bold')

    # 下区域样式
    ax2.set_ylabel('售票率（预估/实际）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('比赛层级', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_level)
    ax2.set_xticklabels(levels, fontsize=11)
    ax2.set_ylim(-0.15, 1.15)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')
    ax2.text(0.02, -0.12,
             '注：策略C柱子分为「原价出售（紫）」和「折扣出售（青）」，数值为售票率+对应票价；下方为各层级平均票价（原价+折扣加权）',
             transform=ax2.transAxes, fontsize=9, color='#666666')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    generate_outputs()
