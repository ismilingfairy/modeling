import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font settings (Switching to a standard font for English)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def calculate_pricing_metrics():
    """
    Data Calculation Logic:
    - Split Strategy C into Base Price / Discounted Sales Rates
    - Calculate average price for each strategy (New: Tier-specific avg price for Strategy C)
    - Calibration: Hot games discount is calculated as 30% off (0.7), not manually set to 120
    """
    # Basic Parameters
    arena_capacity = 18000
    revenue_share_ratio = 0.8
    # Game Tiers (Used for weighted average calculations)
    game_levels = {"Hot": 10, "Regular": 20, "Cold": 10}
    levels = ["Hot", "Regular", "Cold"]
    total_games = sum(game_levels.values())

    # ====================== Strategy A ======================
    a_fixed_price = 80
    a_estimate_rate = {"Hot": 0.7, "Regular": 0.7, "Cold": 0.7}
    a_actual_rate = {"Hot": 0.9, "Regular": 0.7, "Cold": 0.5}
    a_total_revenue, a_avg_attendance = 0, 0
    for level, game_count in game_levels.items():
        single_rev = a_fixed_price * arena_capacity * a_actual_rate[level]
        a_total_revenue += single_rev * game_count * revenue_share_ratio
        a_avg_attendance += a_actual_rate[level] * game_count
    a_avg_attendance_rate = (a_avg_attendance / total_games) * 100
    a_avg_price = a_fixed_price

    # ====================== Strategy B ======================
    b_level_price = {"Hot": 120, "Regular": 85, "Cold": 50}
    b_estimate_rate = {"Hot": 0.9, "Regular": 0.7, "Cold": 0.5}
    b_actual_rate = {"Hot": 0.95, "Regular": 0.75, "Cold": 0.55}
    b_total_revenue, b_avg_attendance = 0, 0
    for level, game_count in game_levels.items():
        single_rev = b_level_price[level] * arena_capacity * b_actual_rate[level]
        b_total_revenue += single_rev * game_count * revenue_share_ratio
        b_avg_attendance += b_actual_rate[level] * game_count
    b_avg_attendance_rate = (b_avg_attendance / total_games) * 100
    b_avg_price = (b_level_price["Hot"] * 10 + b_level_price["Regular"] * 20 + b_level_price["Cold"] * 10) / total_games

    # ====================== Strategy C ======================
    c_base_price = {"Hot": 120, "Regular": 85, "Cold": 50}
    c_discount_rate = 0.7
    c_discount_price = {
        "Hot": round(c_base_price["Hot"] * c_discount_rate, 1),
        "Regular": round(c_base_price["Regular"] * c_discount_rate, 1),
        "Cold": round(c_base_price["Cold"] * c_discount_rate, 1)
    }
    c_estimate_rate = {"Hot": 0.9, "Regular": 0.7, "Cold": 0.5}
    c_base_rate = b_actual_rate
    c_actual_rate = {"Hot": 0.98, "Regular": 0.9, "Cold": 0.85}
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

    # Organize Data
    strategies = ['Strategy A: Fixed Price', 'Strategy B: Tiered Pricing', 'Strategy C: Dynamic Pricing']
    total_revenue = [a_total_revenue, b_total_revenue, c_total_revenue]
    total_attendance_rate = [a_avg_attendance_rate, b_avg_attendance_rate, c_avg_attendance_rate]
    avg_price_list = [a_avg_price, b_avg_price, c_avg_price]

    estimate_rates = {
        "Strategy A: Fixed Price": [a_estimate_rate[level] for level in levels],
        "Strategy B: Tiered Pricing": [b_estimate_rate[level] for level in levels],
        "Strategy C: Dynamic Pricing": [c_estimate_rate[level] for level in levels]
    }
    actual_rates = {
        "Strategy A: Fixed Price": [a_actual_rate[level] for level in levels],
        "Strategy B: Tiered Pricing": [b_actual_rate[level] for level in levels],
        "Strategy C: Dynamic Pricing": [c_actual_rate[level] for level in levels]
    }
    c_base_rates = [c_base_rate[level] for level in levels]
    c_discount_rates = [c_discount_rate_sale[level] for level in levels]
    price_data = {
        "Strategy A: Fixed Price": [a_fixed_price for _ in levels],
        "Strategy B: Tiered Pricing": [b_level_price[level] for level in levels],
        "Strategy C: Dynamic Pricing - Base": [c_base_price[level] for level in levels],
        "Strategy C: Dynamic Pricing - Disc.": [c_discount_price[level] for level in levels],
        "Strategy C: Tier Avg Price": [c_level_avg_price[level] for level in levels]
    }

    return (strategies, total_revenue, total_attendance_rate, levels, estimate_rates,
            actual_rates, price_data, avg_price_list, c_base_rates, c_discount_rates)


def generate_outputs():
    (strategies, total_revenue, total_attendance_rate, levels, estimate_rates,
     actual_rates, price_data, avg_price_list, c_base_rates, c_discount_rates) = calculate_pricing_metrics()

    # Create figure, increase height for better spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), gridspec_kw={'height_ratios': [2, 3]})
    fig.suptitle('Comparison of 3 Ticketing Mechanisms (Split Base/Discount + Tier Avg Price)', fontsize=20,
                 fontweight='bold', y=0.98)

    # ====================== Top Area: Revenue(Bar) + Attendance(Line) + Avg Price(Line) ======================
    x = np.arange(len(strategies))
    width = 0.35

    # 1. Left Axis: Total Season Revenue (Bar Chart)
    bars_rev = ax1.bar(
        x, total_revenue, width,
        color=['#7f8c8d', '#2980b9', '#e74c3c'],
        alpha=0.8, label='Total Season Revenue (After Split)'
    )

    # Core Adjustment 1: Significantly increase left y-axis limit to push bars down,
    # clearing space for lines above to avoid overlap.
    ax1.set_ylim(0, max(total_revenue) * 1.6)

    # Bar Annotations
    for bar in bars_rev:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height + 100000,
            f'${height / 1000000:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    # 2. Right Axis 1: Overall Attendance Rate (Green Line)
    ax1_twin1 = ax1.twinx()  # Default position is right
    line_att = ax1_twin1.plot(
        x, total_attendance_rate,
        color='#27ae60', marker='o', linewidth=4, markersize=10, label='Overall Attendance Rate'
    )

    # Set Right Axis 1 (Green) limits to keep line in the upper area
    ax1_twin1.set_ylim(0, 115)

    # Attendance Annotations (Fine-tuned position)
    for i, v in enumerate(total_attendance_rate):
        ax1_twin1.text(
            i, v + 2, f'{v:.1f}%',
            ha='center', va='bottom', color='#27ae60', fontsize=11, fontweight='bold'
        )

    # 3. Right Axis 2: Avg Price (Orange Line)
    ax1_twin2 = ax1.twinx()  # Twinx again, still on the right
    # Core Adjustment 2: Offset Right Axis 2 further right to avoid overlap with Axis 1
    ax1_twin2.spines['right'].set_position(('outward', 60))

    line_price = ax1_twin2.plot(
        x, avg_price_list,
        color='#f39c12', marker='s', linewidth=4, markersize=10, label='Overall Avg Price'
    )

    # Set Right Axis 2 (Orange) limits to keep line in the upper area
    ax1_twin2.set_ylim(0, max(avg_price_list) * 1.4)

    # Avg Price Annotations
    for i, v in enumerate(avg_price_list):
        ax1_twin2.text(
            i, v + 2, f'${v:.1f}',
            ha='center', va='bottom', color='#f39c12', fontsize=11, fontweight='bold'
        )

    # === Top Area Styling ===
    # Left Axis Settings
    ax1.set_ylabel('Total Season Revenue (USD)', fontsize=12, fontweight='bold')

    # Right Axis 1 (Green) Settings - Explicitly put ticks on the right
    ax1_twin1.set_ylabel('Overall Attendance Rate (%)', fontsize=12, fontweight='bold', color='#27ae60')
    ax1_twin1.tick_params(axis='y', labelcolor='#27ae60', right=True, labelright=True, left=False, labelleft=False)

    # Right Axis 2 (Orange) Settings
    ax1_twin2.set_ylabel('Overall Avg Price (USD)', fontsize=12, fontweight='bold', color='#f39c12')
    ax1_twin2.tick_params(axis='y', labelcolor='#f39c12')

    # X-Axis Settings
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=11)

    # Grid (Horizontal only)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')

    # Merge Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin1.get_legend_handles_labels()
    lines3, labels3 = ax1_twin2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=10)

    # ====================== Bottom Area: Tiered Sales Rates (Black Text Optimized) ======================
    x_level = np.arange(len(levels))
    bar_width = 0.15
    offsets = [-0.25, 0, 0.25]
    colors_estimate = ['#95a5a6', '#3498db', '#e74c3c']
    colors_actual = ['#5d6d7e', '#21618c', '#c0392b']
    color_c_base = '#8e44ad'
    color_c_discount = '#1abc9c'

    # Strategies A/B
    for i, strategy in enumerate(strategies[:2]):
        # Estimated
        ax2.bar(
            x_level + offsets[i] - bar_width / 2, estimate_rates[strategy],
            bar_width, color=colors_estimate[i], alpha=0.8, label=f'{strategy} - Est.'
        )
        # Actual
        ax2.bar(
            x_level + offsets[i] + bar_width / 2, actual_rates[strategy],
            bar_width, color=colors_actual[i], alpha=0.8, label=f'{strategy} - Act.'
        )

    # Strategy C
    c_idx = 2
    # Estimated
    ax2.bar(
        x_level + offsets[c_idx] - bar_width / 2, estimate_rates[strategies[c_idx]],
        bar_width, color=colors_estimate[c_idx], alpha=0.8, label=f'{strategies[c_idx]} - Est.'
    )
    # Actual (Stacked)
    ax2.bar(
        x_level + offsets[c_idx] + bar_width / 2, c_base_rates,
        bar_width, color=color_c_base, alpha=0.9, label=f'{strategies[c_idx]} - Base Price'
    )
    ax2.bar(
        x_level + offsets[c_idx] + bar_width / 2, c_discount_rates,
        bar_width, color=color_c_discount, alpha=0.9, label=f'{strategies[c_idx]} - Discounted',
        bottom=c_base_rates
    )

    # Strategies A/B Annotations
    for i, strategy in enumerate(strategies[:2]):
        for j, level in enumerate(levels):
            est_val = estimate_rates[strategy][j]
            ax2.text(j + offsets[i] - bar_width / 2, est_val + 0.02, f'{est_val * 100:.0f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

            act_val = actual_rates[strategy][j]
            ax2.text(j + offsets[i] + bar_width / 2, act_val + 0.02, f'{act_val * 100:.0f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

            price_val = price_data[strategy][j]
            ax2.text(j + offsets[i], -0.1, f'Price ${price_val}', ha='center', va='top', fontsize=7, color='#666666',
                     fontweight='bold')

    # Strategy C Annotations (Black Text)
    for j, level in enumerate(levels):
        est_val = estimate_rates[strategies[c_idx]][j]
        ax2.text(j + offsets[c_idx] - bar_width / 2, est_val + 0.02, f'{est_val * 100:.0f}%', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

        base_val = c_base_rates[j]
        base_price = price_data["Strategy C: Dynamic Pricing - Base"][j]
        ax2.text(j + offsets[c_idx] + bar_width / 2, base_val / 2, f'{base_val * 100:.0f}%\nBase ${base_price}',
                 ha='center', va='center', fontsize=7, color='black', fontweight='bold')

        discount_val = c_discount_rates[j]
        if discount_val > 0:
            discount_price = price_data["Strategy C: Dynamic Pricing - Disc."][j]
            ax2.text(j + offsets[c_idx] + bar_width / 2, base_val + discount_val / 2,
                     f'{discount_val * 100:.0f}%\nDisc. ${discount_price}', ha='center', va='center', fontsize=7,
                     color='black', fontweight='bold')

        c_level_avg_price = price_data["Strategy C: Tier Avg Price"][j]
        ax2.text(j + offsets[c_idx], -0.1, f'Tier Avg ${c_level_avg_price}', ha='center', va='top', fontsize=7,
                 color='#333333', fontweight='bold')

    # Bottom Area Styling
    ax2.set_ylabel('Ticket Sales Rate (Est./Act.)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Game Tier', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_level)
    ax2.set_xticklabels(levels, fontsize=11)
    ax2.set_ylim(-0.15, 1.15)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')

    # MODIFIED: Moved note position further down (-0.20) to avoid overlap with labels
    ax2.text(0.02, -0.20,
             'Note: Strategy C bars split into "Base Price (Purple)" and "Discounted (Teal)", showing rate + price; Bottom values are tier weighted avg prices.',
             transform=ax2.transAxes, fontsize=9, color='#666666')

    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    # MODIFIED: Increased bottom margin (0.05 -> 0.08) to accommodate the lower note
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    generate_outputs()
