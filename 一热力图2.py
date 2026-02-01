import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os  # 新增：用于文件夹创建


# ===============================
# 1. 核心类定义（完全不变）
# ===============================

class Player:
    def __init__(self, name, off, deff, fame, salary):
        self.name = name
        self.off = off
        self.deff = deff
        self.fame = fame
        self.salary = salary
        self.health = 100
        self.shapley_val = 0.0

    @property
    def ability(self):
        return (self.off + self.deff) / 2
    
    def __repr__(self):
        return f"{self.name} ({self.ability:.0f})"

class Team:
    def __init__(self, roster, brand, config):
        self.roster = roster
        self.brand = brand
        self.config = config
        self.wins = 0
        self.gate_revenue = 0.0

    def play_game(self):
        # 挑选健康球员
        healthy = [p for p in self.roster if p.health > 60]
        if not healthy: healthy = self.roster[:]

        w_win = self.config['W1_WIN']

        # 核心逻辑：根据策略权重决定上场名单
        rotation = sorted(
            healthy,
            key=lambda p: w_win * p.ability + (1 - w_win) * p.fame,
            reverse=True
        )[:8]

        # 竞技胜率 (Logistic)
        strength = np.mean([p.ability for p in rotation])
        opp = np.random.normal(75, 5) # 对手强度波动
        win_prob = 1 / (1 + np.exp(-(strength - opp) / 6))

        if random.random() < win_prob:
            self.wins += 1

        # ===== 修正后的票价弹性（指数型）=====
        # P=45 -> ratio=1.0 -> elasticity=1.0
        # P=60 -> ratio=1.33 -> elasticity=exp(-0.5) ≈ 0.6
        base_price = 45
        price_ratio = self.config['TICKET_PRICE'] / base_price
        elasticity = np.exp(-1.5 * (price_ratio - 1))

        # 基础吸引力
        has_star = any(p.fame >= 90 for p in rotation)
        brand_factor = self.brand / 300
        raw_attend = 0.55 + brand_factor + (0.25 if has_star else 0)

        # 最终上座率
        attend_rate = min(
            self.config['MAX_ATTEND_RATE'],
            raw_attend * elasticity
        )
        attend_rate = max(0.15, attend_rate) # 保底

        attendance = self.config['ARENA_CAPACITY'] * attend_rate
        self.gate_revenue += attendance * self.config['TICKET_PRICE']

        # 磨损
        for p in self.roster:
            if p in rotation:
                p.health -= np.random.uniform(0.5, 1.5)
            else:
                p.health = min(100, p.health + 3.0)

# ===============================
# 2. Shapley Value（修正逻辑，完全不变）
# ===============================

def characteristic_value(players, w1_win, w2_profit):
    if not players: return 0.0

    # 【修正1】使用 Mean 而非 L2/N，避免人数越多评分越低的问题
    strength = np.mean([p.ability for p in players])
    
    # 竞技价值 (0-100)
    win_val = 100 / (1 + np.exp(-(strength - 70) / 8))

    # 商业价值
    fame_sum = sum(p.fame for p in players)
    has_megastar = any(p.fame >= 95 for p in players)
    superstar_cnt = sum(1 for p in players if p.fame >= 80)
    
    # 商业乘数效应
    multiplier = 1.0 + (0.5 if has_megastar else 0) + (0.3 if superstar_cnt >= 2 else 0)
    profit_val = fame_sum * multiplier

    # 加权组合
    return w1_win * win_val + w2_profit * profit_val

def compute_shapley(roster, w1_win, w2_profit, samples=50): 
    # samples 降为 50 以提升灵敏度分析速度，结果差异不大
    shapley = {p.name: 0.0 for p in roster}

    for _ in range(samples):
        perm = roster[:]
        random.shuffle(perm)
        coalition = []
        prev_val = 0.0

        for p in perm:
            coalition.append(p)
            new_val = characteristic_value(coalition, w1_win, w2_profit)
            shapley[p.name] += new_val - prev_val
            prev_val = new_val

    for p in roster:
        p.shapley_val = shapley[p.name] / samples

def enforce_salary_cap(roster, cap, w1_win, w2_profit):
    compute_shapley(roster, w1_win, w2_profit)

    # 【修正2】量级平衡
    # salary 单位是 k (e.g. 76k)，shapley 是 0-100
    # ratio = shapley / (salary/1000) -> 范围在 0.5 ~ 2.0 之间
    # 加上 star_bonus (e.g. 1.0) 就处于同一量级了
    
    def evaluation_score(p):
        cost_efficiency = p.shapley_val / (p.salary / 1000 + 1e-5) # 避免除0
        star_bonus = 2.0 if p.fame >= 90 else 0.0 # 巨星保护分
        return cost_efficiency + star_bonus

    roster.sort(key=evaluation_score, reverse=False) # 升序，先切分低的

    total_salary = sum(p.salary for p in roster)
    # 保持至少 10 人
    while total_salary > cap and len(roster) > 10:
        cut = roster.pop(0) # 移除得分最低的
        total_salary -= cut.salary

    return roster

# ===============================
# 3. 仿真引擎（完全不变）
# ===============================

def run_single_simulation(config):
    # 初始名单 (硬编码以便重现)
    roster = [
        Player("Caitlin Clark", 96, 70, 99, 76_000), # 高能低薪
        Player("Aliyah Boston", 88, 85, 78, 78_000),
        Player("Vet Star", 80, 72, 70, 190_000),     # 低能高薪
        Player("Role B", 78, 75, 20, 160_000),
        Player("Role C", 76, 74, 15, 150_000),
    ] + [Player(f"Bench{i}", 65, 65, 5, 80_000) for i in range(7)]

    cap = 1_500_000
    brand = 60
    
    total_profit = 0
    total_wins = 0

    for year in range(config['YEARS']):
        team = Team(roster, brand, config)

        for p in roster: p.health = 100

        for _ in range(config['GAMES']):
            team.play_game()

        gate = team.gate_revenue

        # 媒体收入公式：品牌基础 + 策略修正 (赢球策略通常带来稍好的媒体合同)
        media_base = 8_000_000 + 25_000_000 * (brand / (brand + 150))
        media = media_base * (0.8 + 0.2 * config['W1_WIN']) 
        
        merch = 20_000 * sum(p.fame for p in roster)

        revenue = gate + media + merch
        salary_cost = sum(p.salary for p in roster)
        profit = revenue - salary_cost - config['FIXED_COST']

        total_profit += profit
        total_wins += team.wins

        # 品牌迭代 (对数增长，指数衰减)
        star_power = sum(p.fame for p in roster)
        dB = (
            config['ALPHA_WIN'] * np.log(1 + team.wins)
            + config['ALPHA_STAR'] * np.log(1 + star_power)
        ) * (1 - brand / config['MAX_BRAND'])

        brand = max(0, brand + dB - config['DELTA'] * brand)

        # 休赛期管理
        cap *= 1.08
        roster = enforce_salary_cap(roster, cap, config['W1_WIN'], 1 - config['W1_WIN'])

        while len(roster) < 12:
            roster.append(Player(f"Rookie_{year}_{len(roster)}", 70, 70, 30, 70_000))

    return {
        "Avg_Profit_M": total_profit / config['YEARS'] / 1e6,
        "Avg_Wins": total_wins / config['YEARS']
    }

# ===============================
# 4. 灵敏度分析（完全不变）
# ===============================

def run_sensitivity_analysis():
    base_config = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'MAX_ATTEND_RATE': 0.98,
        'FIXED_COST': 25_000_000,
        'MAX_BRAND': 400,
        'ALPHA_WIN': 1.2,
        'ALPHA_STAR': 0.15, # 稍微调高星味权重
        'DELTA': 0.10,
        'YEARS': 5
    }

    # 参数扫描空间（原参数，未修改）
    prices = np.linspace(30, 80, 6)     # 票价 30-80
    win_weights = np.linspace(0.1, 0.9, 5) # 策略权重

    results = []
    
    print(f"开始模拟: {len(prices) * len(win_weights)} 个组合...")

    for price, w_win in product(prices, win_weights):
        config = base_config.copy()
        config['TICKET_PRICE'] = price
        config['W1_WIN'] = w_win

        profits, wins = [], []
        # 运行多次取平均以平滑随机性
        for _ in range(6): 
            res = run_single_simulation(config)
            profits.append(res['Avg_Profit_M'])
            wins.append(res['Avg_Wins'])

        avg_profit = np.mean(profits)
        avg_wins = np.mean(wins)
        
        results.append({
            "Ticket_Price": price,
            "Strategy_Win_Weight": w_win,
            "Avg_Profit_M": avg_profit,
            "Avg_Wins": avg_wins
        })
        # print(f"P:{price:.0f} W:{w_win:.1f} -> ${avg_profit:.2f}M") # 调试用

    return pd.DataFrame(results)

# ===============================
# 5. 新增：保存热力图函数
# ===============================
def save_heatmap(df, year, save_dir):
    # 生成透视表
    pivot = df.pivot(
        index="Strategy_Win_Weight",
        columns="Ticket_Price",
        values="Avg_Profit_M"
    )
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", cbar_kws={'label': 'Avg Profit ($M)'})
    plt.title(f"Year {year} - WNBA Simulation: Profit Optimization\n(Ticket Price vs. Strategy Focus)")
    plt.ylabel("Strategy Weight (Winning > Fame)")
    plt.xlabel("Ticket Price ($)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"year_{year}_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布，释放内存
    print(f"Year {year} 热力图已保存至: {save_path}")

# ===============================
# 6. 主执行逻辑（新增5年循环+趋势图）
# ===============================

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # 1. 创建保存热力图的文件夹
    heatmap_dir = "wnba_heatmaps"
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
        print(f"创建文件夹: {heatmap_dir}")

    # 2. 初始化最优解收集列表
    best_results = []

    # 3. 5年循环执行仿真
    for current_year in range(1, 6):
        print(f"\n========== 开始第 {current_year} 年仿真 ==========")
        # 执行灵敏度分析
        df_year = run_sensitivity_analysis()
        
        # 找到当年最优解
        best_config = df_year.loc[df_year['Avg_Profit_M'].idxmax()].copy()
        best_config['Year'] = current_year  # 新增年份列
        best_results.append(best_config)
        
        # 保存当年热力图
        save_heatmap(df_year, current_year, heatmap_dir)
        
        # 打印当年最优解
        print(f"\n=== 第 {current_year} 年最优策略组合 ===")
        print(best_config)

    # 4. 生成最优解工作表并保存为CSV
    best_df = pd.DataFrame(best_results)
    best_df = best_df[['Year', 'Ticket_Price', 'Strategy_Win_Weight', 'Avg_Profit_M', 'Avg_Wins']]
    best_df.to_csv("wnba_best_strategies.csv", index=False)
    print("\n最优解工作表已保存为: wnba_best_strategies.csv")

    # 5. 绘制最优票价折线图（横坐标趋势）
    plt.figure(figsize=(10, 5))
    plt.plot(best_df['Year'], best_df['Ticket_Price'], marker='o', linewidth=2, color='blue', label='Optimal Ticket Price')
    plt.title("5-Year Trend of Optimal Ticket Price")
    plt.xlabel("Year")
    plt.ylabel("Optimal Ticket Price ($)")
    plt.xticks(range(1, 6))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_ticket_price_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("最优票价趋势图已保存为: optimal_ticket_price_trend.png")

    # 6. 绘制最优策略权重折线图（纵坐标趋势）
    plt.figure(figsize=(10, 5))
    plt.plot(best_df['Year'], best_df['Strategy_Win_Weight'], marker='s', linewidth=2, color='red', label='Optimal Strategy Weight (Win)')
    plt.title("5-Year Trend of Optimal Strategy Weight (Winning > Fame)")
    plt.xlabel("Year")
    plt.ylabel("Optimal Strategy Weight")
    plt.xticks(range(1, 6))
    plt.ylim(0, 1)  # 权重范围0-1
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_strategy_weight_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("最优策略权重趋势图已保存为: optimal_strategy_weight_trend.png")

    print("\n========== 所有任务完成 ==========")
    print(f"1. 5年热力图保存在: {heatmap_dir} 文件夹")
    print("2. 最优解工作表: wnba_best_strategies.csv")
    print("3. 最优票价趋势图: optimal_ticket_price_trend.png")
    print("4. 最优策略权重趋势图: optimal_strategy_weight_trend.png")
