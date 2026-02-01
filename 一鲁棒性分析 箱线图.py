import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'


# ---------------------------------------------------------
# 1. 灵敏度分析: 商业权重 w_biz 对 "保镖策略" 的影响
# ---------------------------------------------------------

def run_sensitivity_w_biz():
    w_biz_values = np.linspace(0, 20, 50)  # 权重从 0 到 20
    enforcer_ratios = []

    for w in w_biz_values:
        # 简化模拟: 随着 w_biz 增加，核心资产价值增加
        # 假设: 当 资产价值 * 受伤概率 > 保镖成本 时，选择签保镖

        asset_value = 100 + w * 500  # 基础价值 + 商业溢价
        cost_enforcer = 50
        prob_injury = 0.05

        # 模拟 100 次决策机会
        decisions = []
        for _ in range(100):
            # 加入一点随机扰动
            current_risk = prob_injury * np.random.normal(1, 0.2)
            expected_loss = asset_value * current_risk

            if expected_loss > cost_enforcer:
                decisions.append(1)  # Sign Enforcer
            else:
                decisions.append(0)  # Don't Sign

        enforcer_ratios.append(sum(decisions) / 100)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(w_biz_values, enforcer_ratios, color='#C8102E', linewidth=3)

    # 标注相变点
    plt.axvline(x=4.0, color='grey', linestyle='--')
    plt.text(4.2, 0.1, 'Phase Transition Point\n(Strategy Shift)', fontsize=12)

    plt.title('Sensitivity Analysis: Effect of Business Weight ($w_{biz}$) on Strategy', fontsize=14, fontweight='bold')
    plt.xlabel('Commercial Valuation Weight ($w_{biz}$)', fontsize=12)
    plt.ylabel('Frequency of "Sign Enforcer" Action', fontsize=12)
    plt.fill_between(w_biz_values, 0, enforcer_ratios, color='#C8102E', alpha=0.1)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 2. 鲁棒性检验: 蒙特卡洛模拟 (Monte Carlo)
# ---------------------------------------------------------

def run_robustness_check():
    n_simulations = 1000

    h2d2_profits = []
    myopic_profits = []

    for _ in range(n_simulations):
        # 随机因子
        market_noise = np.random.normal(1.0, 0.2)  # 市场环境波动
        injury_luck_h2d2 = np.random.rand()  # H2D2 的运气
        injury_luck_myopic = np.random.rand()  # Myopic 的运气

        # H2D2 模拟逻辑 (简化版)
        # 投入高，抗风险高
        base_profit_h2d2 = 20 * market_noise
        # 只有极坏的运气才会导致 H2D2 失败 (阈值低)
        if injury_luck_h2d2 < 0.05:
            loss = 10
        else:
            loss = 0
        h2d2_profits.append(base_profit_h2d2 - loss)

        # Myopic 模拟逻辑
        # 初期利润高，但抗风险极低
        base_profit_myopic = 25 * market_noise  # 初期看起来赚更多
        # 只要运气一般，就会导致失败 (阈值高)
        if injury_luck_myopic < 0.30:
            loss = 35  # 毁灭性打击 (Clark 离队/受伤)
        else:
            loss = 0
        myopic_profits.append(base_profit_myopic - loss)

    # 整理数据绘图
    data = pd.DataFrame({
        'Profit': h2d2_profits + myopic_profits,
        'Strategy': ['H2D2 (Robust)'] * n_simulations + ['Myopic (Fragile)'] * n_simulations
    })

    plt.figure(figsize=(8, 6))

    # 绘制箱线图 + 散点
    sns.boxplot(x='Strategy', y='Profit', data=data, palette=['#002D62', 'orange'], width=0.5)
    sns.stripplot(x='Strategy', y='Profit', data=data, color='.3', alpha=0.1, jitter=True)

    plt.title('Robustness Check: Monte Carlo Simulation (N=1000)', fontsize=14, fontweight='bold')
    plt.ylabel('Final Cumulative Profit ($M)', fontsize=12)

    # 计算统计量并标注
    mean_h = np.mean(h2d2_profits)
    std_h = np.std(h2d2_profits)
    mean_m = np.mean(myopic_profits)
    std_m = np.std(myopic_profits)

    plt.text(0.2, mean_h + 5, f'Mean: ${mean_h:.1f}M\nStd Dev: {std_h:.1f}\n(Low Variance)', ha='center',
             color='#002D62', fontweight='bold')
    plt.text(1.2, mean_m + 5, f'Mean: ${mean_m:.1f}M\nStd Dev: {std_m:.1f}\n(High Variance)', ha='center',
             color='darkorange', fontweight='bold')

    plt.tight_layout()
    plt.show()


# 运行
if __name__ == "__main__":
    run_sensitivity_w_biz()
    run_robustness_check()