import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 适配中英文显示
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 数据准备 (基于之前的模型计算结果)
# ==========================================

# 狂热队配色
FEVER_BLUE = '#002D62'
FEVER_RED = '#C8102E'
FEVER_GOLD = '#FFCD00'
FEVER_GRAY = '#A2AAAD'

# 模块一数据：球员价值构成
# Base_Eff = Base * Stability (有效基础分)
# Synergy = Total_Fuzzy - Base_Eff (协同效应增益)
player_data = {
    'Name': ['Caitlin Clark', 'Aliyah Boston', 'Kelsey Mitchell', 'NaLyssa Smith',
             'Erica Wheeler', 'K.L. Samuelson', 'Lexie Hull', 'Temi Fagbenle',
             'Kristy Wallace', 'Grace Berger', 'Victaria Saxton'],
    'Fuzzy_Val': [106.83, 90.06, 69.39, 63.71, 40.16, 40.80, 36.00, 35.00, 29.75, 24.00, 13.50],
    'Base_Raw': [85, 80, 75, 70, 55, 48, 40, 50, 35, 30, 15],
    'Stability': [0.95, 0.95, 0.90, 0.85, 0.80, 0.85, 0.90, 0.70, 0.85, 0.80, 0.90]
}
df_players = pd.DataFrame(player_data)
# 计算成分
df_players['Base_Eff'] = df_players['Base_Raw'] * df_players['Stability']
df_players['Synergy'] = df_players['Fuzzy_Val'] - df_players['Base_Eff']
# 修正排序：按总价值降序
df_players = df_players.sort_values(by='Fuzzy_Val', ascending=True).reset_index(drop=True)

# 模块三数据：城市影响分析
city_data = {
    'City': ['Golden State (SF)', 'Portland', 'Toronto', 'Philadelphia', 'Nashville'],
    'Distance_km': [3200, 3150, 780, 920, 280],
    'Net_Finance': [0.67, 0.60, 0.04, 0.12, -1.18],  # 财务净收益 (分红 - 蚕食)
    'Competitive': [2.78, 2.60, 1.81, 1.63, -3.02],  # 竞技/战略影响 (折算后的分数)
    'Total_Score': [3.45, 3.20, 1.85, 1.75, -4.20]
}
df_cities = pd.DataFrame(city_data)


# ==========================================
# 1. 模块一可视化：球员价值与保护线 (Stacked Bar)
# ==========================================

def plot_player_valuation():
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(df_players))

    # 绘制堆叠条形图
    # 1. 有效基础能力 (Base * Stability)
    p1 = ax.barh(y_pos, df_players['Base_Eff'], color=FEVER_BLUE,
                 edgecolor='white', label='Effective Base Skill')

    # 2. 协同效应/化学反应 (Synergy)
    # 绿色表示正向协同，红色表示负向协同
    colors_syn = [FEVER_GOLD if x >= 0 else FEVER_RED for x in df_players['Synergy']]
    p2 = ax.barh(y_pos, df_players['Synergy'], left=df_players['Base_Eff'],
                 color=colors_syn, alpha=0.9, edgecolor='white', label='Synergy/Chemistry Boost')

    # 添加保护截止线 (Top 6 保护)
    K = 6
    cut_line = len(df_players) - K - 0.5
    ax.axhline(y=cut_line, color=FEVER_RED, linestyle='--', linewidth=2)
    ax.text(115, cut_line, '  Protection Cut-off (Top 6)', color=FEVER_RED, va='center', fontweight='bold')

    # 标签与装饰
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_players['Name'], fontsize=12)
    ax.set_xlabel('Fuzzy Shapley Value (Total Contribution)', fontsize=12, fontweight='bold')
    ax.set_title('Module 1: Player Valuation Breakdown (Fuzzy Shapley Model)', fontsize=15, fontweight='bold', pad=20)

    # 在条形图末尾标注总分
    for i, v in enumerate(df_players['Fuzzy_Val']):
        ax.text(v + 1, i, f"{v:.1f}", va='center', fontweight='bold', color='#333')

    # 定制图例
    ax.legend(loc='lower right')

    # 添加注释
    ax.text(80, 0.5, "Exposed Zone", fontsize=20, color='gray', alpha=0.2, ha='center')
    ax.text(80, 8.5, "Protected Zone", fontsize=20, color=FEVER_BLUE, alpha=0.1, ha='center')

    plt.tight_layout()
    plt.show()


# ==========================================
# 2. 模块三可视化：选址影响矩阵 (Scatter + Breakdown)
# ==========================================

def plot_city_impact():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # --- 子图 1: 距离-影响 散点图 (The Gravity Model View) ---
    ax1 = fig.add_subplot(gs[0, :])

    # 绘制散点
    sc = ax1.scatter(df_cities['Distance_km'], df_cities['Total_Score'],
                     s=1500, alpha=0.9,
                     c=df_cities['Total_Score'], cmap='RdYlGn', edgecolors='black')

    # 装饰
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Distance from Indianapolis (km)', fontsize=11)
    ax1.set_ylabel('Total Impact Score', fontsize=11)
    ax1.set_title('City Expansion Impact: Distance vs. Benefit', fontsize=14, fontweight='bold')

    # 标注城市名
    for i, row in df_cities.iterrows():
        ax1.text(row['Distance_km'], row['Total_Score'], row['City'],
                 ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # 添加区域标注
    ax1.text(200, -3, "Cannibalization Zone\n(High Harm)", color='red', fontsize=12, alpha=0.6)
    ax1.text(3000, 2, "Safe Growth Zone\n(High Benefit)", color='green', fontsize=12, alpha=0.6)

    # --- 子图 2: 影响因子分解 (Financial vs. Competitive) ---
    ax2 = fig.add_subplot(gs[1, :])

    # 数据准备
    x = np.arange(len(df_cities))
    width = 0.35

    # 绘制分组柱状图
    rects1 = ax2.bar(x - width / 2, df_cities['Net_Finance'], width, label='Financial Impact (Net)', color=FEVER_GOLD)
    rects2 = ax2.bar(x + width / 2, df_cities['Competitive'], width, label='Competitive/Strategic Impact',
                     color=FEVER_BLUE)

    # 装饰
    ax2.set_ylabel('Component Score')
    ax2.set_title('Impact Component Analysis by City', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_cities['City'])
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.legend()

    # 标注数值
    ax2.bar_label(rects1, padding=3, fmt='%.2f')
    ax2.bar_label(rects2, padding=3, fmt='%.2f')

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 执行绘图
# ==========================================

print("正在生成模块一图表：球员夏普利估值...")
plot_player_valuation()

print("正在生成模块三图表：城市选址影响分析...")
plot_city_impact()

