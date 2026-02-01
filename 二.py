import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
# 印第安纳狂热队配色 (Navy, Red, Gold)
FEVER_COLORS = ["#041E42", "#C8102E", "#FFB81C"]


# ==========================================
# 1. 数据生成 (模拟 H2D2 模型输出)
# ==========================================

def generate_visualization_data():
    # --- 模拟一：选秀与自由球员池 (Synergy Analysis) ---
    # 生成 50 名球员数据
    np.random.seed(42)
    n_players = 50
    usage = np.random.normal(20, 8, n_players)  # 球权使用率
    raw_talent = np.random.normal(75, 10, n_players)  # 原始天赋
    defense = np.random.normal(60, 15, n_players)

    # H2D2 核心公式模拟: Synergy = Talent - lambda * (Usage - 20) + Defense_Bonus
    # 核心逻辑：Usage 越高，协同分越低（除非你是核心）
    synergy = raw_talent - 1.2 * (usage - 18) + 0.3 * defense

    df_players = pd.DataFrame({
        "Usage_Rate": usage,
        "Synergy_Score": synergy,
        "Raw_Talent": raw_talent,
        "Type": ["Role Player"] * n_players
    })

    # 添加特殊点
    cc_data = pd.DataFrame({"Usage_Rate": [35], "Synergy_Score": [99], "Raw_Talent": [99], "Type": ["Core (Clark)"]})
    target_data = pd.DataFrame(
        {"Usage_Rate": [12], "Synergy_Score": [88], "Raw_Talent": [78], "Type": ["Target (3-and-D)"]})
    bust_data = pd.DataFrame(
        {"Usage_Rate": [32], "Synergy_Score": [40], "Raw_Talent": [85], "Type": ["Trap (High Usage)"]})

    df_scatter = pd.concat([df_players, cc_data, target_data, bust_data], ignore_index=True)

    # --- 模拟二：选秀决策对比 (Bar Chart) ---
    draft_comparison = pd.DataFrame({
        "Player": ["Scorer A (High Usage)", "Wing B (3-and-D)", "Big C (Defense)"],
        "Traditional_Scout_Score": [92, 84, 78],  # 传统球探看重个人得分能力
        "H2D2_Model_Score": [45, 90, 82]  # 模型看重适配度
    })

    # --- 模拟三：留队概率曲线 (Logistic Curve) ---
    x_exposure = np.linspace(0, 15, 100)  # 品牌曝光价值 ($M)
    # Logit: P = 1 / (1 + e^-(beta0 + beta1*x))
    # 假设基础薪资满意度为 -2 (即如果没有额外曝光，只有 12% 概率留队)
    y_prob = 1 / (1 + np.exp(-(-2.0 + 0.5 * x_exposure)))

    return df_scatter, draft_comparison, (x_exposure, y_prob)


df_scatter, df_draft, (x_retention, y_retention) = generate_visualization_data()

# ==========================================
# 2. 绘图逻辑
# ==========================================

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig)

# --- 图表 A: 协同适配度矩阵 (The "No-Ball-Hog" Zone) ---
ax1 = fig.add_subplot(gs[0, :])  # 占据第一行整行

# 绘制散点
sns.scatterplot(
    data=df_scatter[df_scatter["Type"] == "Role Player"],
    x="Usage_Rate", y="Synergy_Score",
    color="grey", alpha=0.4, s=60, ax=ax1, label="Generic Players"
)

# 突出显示特殊点
special_points = df_scatter[df_scatter["Type"] != "Role Player"]
colors = {"Core (Clark)": FEVER_COLORS[0], "Target (3-and-D)": "green", "Trap (High Usage)": FEVER_COLORS[1]}

for idx, row in special_points.iterrows():
    ax1.scatter(row["Usage_Rate"], row["Synergy_Score"], color=colors[row["Type"]], s=200, edgecolors='black',
                label=row["Type"])
    # 添加注释
    ax1.text(row["Usage_Rate"] + 0.5, row["Synergy_Score"] + 2, row["Type"], fontsize=11, fontweight='bold',
             color=colors[row["Type"]])

# 添加趋势线和区域
ax1.axvline(x=25, color='red', linestyle='--', alpha=0.3)
ax1.text(26, 90, "Danger Zone: High Usage\n(Reduces Synergy)", color='red', fontsize=10)
ax1.axvspan(0, 18, color='green', alpha=0.05)
ax1.text(2, 90, "Target Zone: Low Usage\n(Fits with Clark)", color='green', fontsize=10)

ax1.set_title("H2D2 Model: Player Synergy vs. Usage Rate Evaluation", fontsize=16, fontweight='bold')
ax1.set_xlabel("Usage Rate (%) - Ball Dominance", fontsize=12)
ax1.set_ylabel("H2D2 Synergy Score (0-100)", fontsize=12)
ax1.legend(loc='lower left')

# --- 图表 B: 选秀评价差异 (Traditional vs H2D2) ---
ax2 = fig.add_subplot(gs[1, 0])

# 数据转换以便绘图
df_melt = df_draft.melt(id_vars="Player", var_name="Metric", value_name="Score")

sns.barplot(data=df_melt, x="Player", y="Score", hue="Metric", palette=[FEVER_COLORS[0], FEVER_COLORS[2]], ax=ax2)

ax2.set_title("Draft Strategy: Raw Talent vs. System Fit", fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_ylabel("Score")
ax2.legend(title="Evaluation Method")

# 添加标注
for i, p in enumerate(ax2.patches):
    if p.get_height() > 0:
        ax2.annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9), textcoords='offset points')

# --- 图表 C: 核心留队概率模型 (The "Clark Effect") ---
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(x_retention, y_retention, color=FEVER_COLORS[0], linewidth=3, label="Retention Probability Curve")
ax3.fill_between(x_retention, 0, y_retention, color=FEVER_COLORS[0], alpha=0.1)

# 标记特定点
# 点 1: 普通球队给出的报价
ax3.scatter([2], [0.26], color='grey', s=100, zorder=5)
ax3.annotate("Offer from\nAverage Team\n(Low Exposure)", (2.2, 0.20))

# 点 2: 狂热队的报价 (含 Caitlin Clark 曝光加成)
cc_exposure_val = 10
cc_prob = 1 / (1 + np.exp(-(-2.0 + 0.5 * cc_exposure_val)))
ax3.scatter([cc_exposure_val], [cc_prob], color=FEVER_COLORS[1], s=150, zorder=5, edgecolors='black')
ax3.annotate(f"Indiana Fever Offer\n(+${cc_exposure_val}M Brand Value)\nProb: {cc_prob:.0%}",
             (cc_exposure_val - 4, cc_prob - 0.15), fontweight='bold')

ax3.set_title("Retention Probability: Salary + Brand Exposure Utility", fontsize=14, fontweight='bold')
ax3.set_xlabel("Projected Brand/Endorsement Value ($ Million)", fontsize=12)
ax3.set_ylabel("Probability of Re-signing", fontsize=12)
ax3.set_ylim(0, 1.1)

# 调整布局
plt.tight_layout()
plt.show()