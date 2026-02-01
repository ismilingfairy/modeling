import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格 (符合学术论文标准)
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ==========================================
# Visualization 1: The "Clark Effect" (Micro Layer)
# 展示 Clark 在场/不在场时的收入阶跃 (Step Function)
# ==========================================

def plot_revenue_step_function():
    # 数据模拟
    games = np.arange(1, 41)  # WNBA 赛季 40 场比赛

    # 基础收入 (无 Clark): 假设场均 4000 人 * $30 + 少量周边
    base_revenue = np.random.normal(150000, 20000, 40)

    # Clark 收入 (有 Clark): 假设场均 17000 人 * $50 + 大量周边
    # 这是一个非线性的阶跃
    clark_premium = 850000  # 额外的票房和周边溢价
    clark_revenue = base_revenue + clark_premium + np.random.normal(50000, 50000, 40)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(games, base_revenue / 1000, label='Baseline Scenario (No Star)',
             color='grey', linestyle='--', alpha=0.7, linewidth=2)
    plt.plot(games, clark_revenue / 1000, label='Caitlin Clark Scenario',
             color='#C8102E', linewidth=3)  # 狂热队红色

    # 填充区域表示 "Surplus Value"
    plt.fill_between(games, base_revenue / 1000, clark_revenue / 1000,
                     color='#FFCD00', alpha=0.2, label='Consumer Surplus (The "Clark Premium")')  # 狂热队黄色

    # 标注
    plt.annotate('Non-Linear Step Jump\nDue to Star Power', xy=(10, 800), xytext=(15, 600),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    plt.title('Micro Layer: Non-Linear Revenue Impact of Caitlin Clark', fontsize=16, fontweight='bold')
    plt.xlabel('Season Games (t)', fontsize=14)
    plt.ylabel('Match Day Revenue ($1,000s)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.ylim(0, 1500)

    plt.tight_layout()
    plt.show()


# ==========================================
# Visualization 2: System Dynamics Simulation (Macro Layer)
# 展示投资策略对 2027 年签约概率的影响
# ==========================================

def plot_system_dynamics_forecast():
    # 时间轴: 2024 - 2029
    years = np.linspace(2024, 2029, 100)

    # 定义 Sigmoid 函数 (签约概率 P_sign)
    def sigmoid(x, k=0.5, x0=60):
        return 1 / (1 + np.exp(-k * (x - x0)))

    # 场景 1: 低投资 (Low Investment) - 老板拿走利润
    # 基础设施吸引力 (InfraAttr) 缓慢增长甚至因折旧持平
    infra_low = 40 + 2 * (years - 2024)

    # 场景 2: 高投资 (High Reinvestment) - 利润投入设施
    # 基础设施吸引力呈指数或快速线性增长
    infra_high = 40 + 10 * (years - 2024) + 1.5 * (years - 2024) ** 2

    # 计算对应的 P_sign
    p_sign_low = sigmoid(infra_low)
    p_sign_high = sigmoid(infra_high)

    # 绘图 (双子图)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 子图 1: 基础设施评分变化
    ax1.plot(years, infra_high, color='#002D62', linewidth=3,
             label='Strategy A: High Reinvestment (Infrastructure)')  # 狂热队深蓝
    ax1.plot(years, infra_low, color='grey', linewidth=2, linestyle='--', label='Strategy B: Profit Taking')
    ax1.set_ylabel('Infrastructure Score (A)', fontsize=12)
    ax1.set_title('Macro Layer: Strategic Investment Simulation', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 子图 2: 签约概率 P_sign
    ax2.plot(years, p_sign_high, color='#C8102E', linewidth=3, label='Prob. of Signing Top FA')
    ax2.plot(years, p_sign_low, color='grey', linewidth=2, linestyle='--')

    # 关键时间点标注 (2027 CBA/Free Agency)
    target_year = 2027
    idx = np.abs(years - target_year).argmin()

    # 标注 Gap
    ax2.annotate(f'Crucial Gap in 2027\n$\Delta P \approx {p_sign_high[idx] - p_sign_low[idx]:.2f}$',
                 xy=(2027, p_sign_high[idx]), xytext=(2025, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    # 绘制垂直线
    ax2.axvline(x=2027, color='black', linestyle=':', alpha=0.8)
    ax2.text(2027.1, 0.1, '2027: New CBA / Clark Extension Window', rotation=90, fontsize=10)

    ax2.set_ylabel('Probability of Signing Free Agents ($P_{sign}$)', fontsize=12)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 运行绘图函数
if __name__ == "__main__":
    plot_revenue_step_function()
    plot_system_dynamics_forecast()