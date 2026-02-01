import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以保证结果可复现
np.random.seed(42)
sns.set_theme(style="whitegrid")


class H2D2_Model:
    def __init__(self, strategy_type='H2D2'):
        """
        初始化模型参数
        strategy_type: 'H2D2' (长期优化) vs 'Myopic' (短视/利润优先)
        """
        self.strategy = strategy_type
        self.time_steps = 6 * 20  # 6个赛季，每赛季20个决策周期(周)

        # --- Layer 1: Macro Parameters ---
        self.cash_balance = 5.0  # 初始现金 (Million $)
        self.infrastructure = 50.0  # 初始设施评分 (0-100)
        self.brand_equity = 50.0
        self.cumulative_profit = 0.0
        self.debt = 0.0

        # --- Layer 2: Meso Parameters ---
        self.salary_cap = 1.5  # WNBA硬工资帽 (Million $)
        self.current_payroll = 1.3
        self.roster_slots = 12
        self.clark_health = 100.0  # 核心健康值
        self.wins = 0

        # --- Layer 3: Micro (Shapley) ---
        # 核心在场带来的基础周收入 (门票+周边)
        self.base_revenue_per_week = 0.8  # Million $ (有Clark)
        self.no_clark_revenue = 0.15  # Million $ (无Clark)

        # 记录轨迹
        self.history = []

    def micro_layer_valuation(self, health):
        """
        Layer 3: 模糊夏普利值估算
        根据健康状况计算核心球员的即时商业价值
        """
        # 简单的模糊逻辑：如果健康<70，商业价值呈指数下降（不上场就没有票房）
        effectiveness = 1.0 if health > 80 else (health / 80.0) ** 2
        val_biz = (self.base_revenue_per_week - self.no_clark_revenue) * effectiveness
        return val_biz

    def meso_layer_decision(self, t, opponent_intensity):
        """
        Layer 2: 赛季内动态操作 (DRL Policy Simulation)
        输出: x_{i,t} (阵容决策)
        """
        # 状态观测
        risk_threshold = 0.05 if self.strategy == 'H2D2' else 0.20
        budget_space = self.salary_cap - self.current_payroll

        action_x = "Hold"  # 默认不动
        cost_impact = 0.0

        # 决策逻辑 (模拟 DRL 训练后的最优策略)
        # 场景：如果对手凶狠且Clark有风险，H2D2策略会签“保镖”，Myopic策略会省钱或签得分手

        injury_prob = 0.05 * (opponent_intensity / 50.0)  # 基础受伤率

        if self.strategy == 'H2D2':
            # H2D2 策略：不惜一切代价保护资产
            if opponent_intensity > 70 and budget_space > 0.05:
                action_x = "Sign_Enforcer"
                cost_impact = 0.05  # 签个蓝领
                injury_prob *= 0.2  # 受伤风险大幅降低
            elif opponent_intensity > 80:
                action_x = "Rest_Star"  # 轮休核心
                injury_prob = 0.0
        else:
            # Myopic 策略：省钱，或者只看眼前胜率
            if budget_space > 0.1:
                action_x = "Sign_Scorer"  # 签得分手增加胜率，但不降低风险
                cost_impact = 0.1
                # injury_prob 保持高位

        return action_x, cost_impact, injury_prob

    def macro_layer_finance(self, t, weekly_net_income):
        """
        Layer 1: 年度财务与战略 (System Dynamics)
        输出: Delta E (再投资), Delta D (债务)
        """
        delta_D = 0  # 假设不举债
        delta_E = 0  # 再投资额

        # 赛季末结算 (假设每20周是一个赛季末)
        is_season_end = (t % 20 == 19)

        if is_season_end:
            season_profit = self.cash_balance  # 简化：假设现金池即为当季可支配利润

            if self.strategy == 'H2D2':
                # H2D2: 80% 利润再投资于设施 (为了2027年续约)
                reinvest_rate = 0.8
                delta_E = season_profit * reinvest_rate
                self.infrastructure += delta_E * 5  # 设施评分提升
                self.cash_balance -= delta_E  # 现金转为固定资产
            else:
                # Myopic: 老板分红，只留 10% 维护
                reinvest_rate = 0.1
                delta_E = season_profit * reinvest_rate
                self.infrastructure -= 2.0  # 设施自然折旧
                self.infrastructure += delta_E * 2
                self.cash_balance -= (season_profit * 0.9)  # 老板拿走

        return delta_D, delta_E

    def step(self, t):
        # 1. 环境随机性 (对手强度)
        opponent_intensity = np.random.normal(50, 20)
        opponent_intensity = np.clip(opponent_intensity, 20, 100)

        # 2. 运行 Layer 2 (Meso) 决策
        action_x, cost_x, prob_injury = self.meso_layer_decision(t, opponent_intensity)

        # 3. 模拟比赛与风险
        # 判定是否受伤
        is_injured = np.random.rand() < prob_injury
        if is_injured:
            self.clark_health -= 30  # 重伤
        else:
            self.clark_health = min(100, self.clark_health + 1)  # 恢复

        # 4. 运行 Layer 3 (Micro) 估值与收入
        # 关键逻辑：如果Clark健康<50，收入断崖式下跌
        current_revenue = self.micro_layer_valuation(self.clark_health)

        # 2027年 (t=60周左右) 续约危机模拟
        # 如果设施太差，Clark 离队，收入永久归零
        if t == 60:
            retention_prob = 1 / (1 + np.exp(-0.1 * (self.infrastructure - 70)))
            if np.random.rand() > retention_prob:
                self.base_revenue_per_week = self.no_clark_revenue  # Clark 离队
                print(f"[{self.strategy}] CATASTROPHE: Clark left due to poor infrastructure!")

        # 5. 财务计算
        weekly_cost = (self.current_payroll / 20.0) + cost_x + 0.1  # 运营成本
        net_income = current_revenue - weekly_cost

        self.cash_balance += net_income
        self.cumulative_profit += net_income

        # 6. 运行 Layer 1 (Macro) 决策
        delta_D, delta_E = self.macro_layer_finance(t, net_income)

        # 7. 动态定价策略 (pi_g)
        pi_g = 1.0
        if opponent_intensity > 80 and self.clark_health > 80:
            pi_g = 1.5  # 强强对话且球星健康，票价上浮50%

        # 记录数据
        risk_metric = prob_injury * (self.base_revenue_per_week * 10)  # 风险 = 概率 * 潜在长期损失

        state_vector = {
            'time': t,
            'strategy': self.strategy,
            'action_x': action_x,  # 决策: 阵容
            'delta_E': delta_E,  # 决策: 再投资
            'pi_g': pi_g,  # 决策: 定价
            'cumulative_profit': self.cumulative_profit,
            'risk_metric': risk_metric,
            'clark_health': self.clark_health,
            'infrastructure': self.infrastructure,
            'revenue': current_revenue
        }
        self.history.append(state_vector)

    def run(self):
        for t in range(self.time_steps):
            self.step(t)
        return pd.DataFrame(self.history)


# --- 主程序：对比两种策略 ---

# 运行 H2D2 策略
model_h2d2 = H2D2_Model(strategy_type='H2D2')
df_h2d2 = model_h2d2.run()

# 运行 短视 策略
model_myopic = H2D2_Model(strategy_type='Myopic')
df_myopic = model_myopic.run()

# --- 结果可视化 ---

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# 1. 预期利润轨迹 (Profit Trajectory)
axes[0].plot(df_h2d2['time'], df_h2d2['cumulative_profit'], label='H2D2 Strategy (Long-term)', color='blue',
             linewidth=2)
axes[0].plot(df_myopic['time'], df_myopic['cumulative_profit'], label='Myopic Strategy (Short-term)', color='grey',
             linestyle='--', linewidth=2)
axes[0].set_ylabel('Cumulative Profit ($M)')
axes[0].set_title('Output 2: Profit Trajectory {$\Pi^*_t$}', fontweight='bold')
axes[0].legend()
axes[0].axvline(x=60, color='red', linestyle=':', alpha=0.5, label='2027 Contract Year')

# 2. 风险指标 (Risk Metric)
# 平滑处理以便观察趋势
axes[1].plot(df_h2d2['time'], df_h2d2['risk_metric'].rolling(5).mean(), label='H2D2 Risk (Managed)', color='green')
axes[1].plot(df_myopic['time'], df_myopic['risk_metric'].rolling(5).mean(), label='Myopic Risk (Unmanaged)',
             color='orange')
axes[1].set_ylabel('Risk Value ($M exposure)')
axes[1].set_title('Output 3: Risk Metric $Risk_t$ (VaR)', fontweight='bold')
axes[1].legend()

# 3. 最优决策可视化 (Action Snapshot)
# 我们将 action_x 转换为数值以便绘图: Hold=0, Sign_Enforcer=1, Sign_Scorer=-1, Rest=2
action_map = {'Hold': 0, 'Sign_Enforcer': 1, 'Sign_Scorer': -1, 'Rest_Star': 2}
df_h2d2['action_val'] = df_h2d2['action_x'].map(action_map)

axes[2].scatter(df_h2d2['time'], df_h2d2['action_val'], alpha=0.6, s=20, color='purple', label='H2D2 Actions')
axes[2].set_yticks([-1, 0, 1, 2])
axes[2].set_yticklabels(['Sign Scorer', 'Hold', 'Sign Enforcer', 'Rest Star'])
axes[2].set_ylabel('Optimal Decision $S^*_t$')
axes[2].set_title('Output 1: Optimal Decision Vector Snapshot', fontweight='bold')
axes[2].set_xlabel('Simulation Weeks (t)')

plt.tight_layout()
plt.show()

# --- 输出具体的决策向量样本 (Text Output) ---
print("=== Sample Optimal Decision Vectors S*_t (H2D2 Strategy) ===")
sample_indices = [10, 30, 60]  # 选取第10周, 30周, 60周查看
cols = ['time', 'action_x', 'delta_E', 'pi_g']
print(df_h2d2.loc[sample_indices, cols])

print("\n=== Financial Summary (End of Simulation) ===")
print(f"H2D2 Final Profit: ${df_h2d2['cumulative_profit'].iloc[-1]:.2f} M")
print(f"Myopic Final Profit: ${df_myopic['cumulative_profit'].iloc[-1]:.2f} M")