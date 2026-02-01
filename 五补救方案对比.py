# -*- coding: utf-8 -*-
"""
WNBA球队十年收益预测（Clark受伤后补救方案对比版）
核心修改：
1. 分阶段策略：前3年Clark正常受伤，第4年开始实施补救措施
2. 对比三种方案：原始方案、医疗团队（降伤病率）、双子星替补
3. 生成统一的利润+品牌值对比折线图
4. 修复matplotlib低版本兼容性问题（移除pad参数）
"""

import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product


# ===============================
# 1. 球员类：支持分阶段调整Clark伤病率+双子星标记
# ===============================
class Player:
    def __init__(self, name, off, deff, fame, salary, age):
        self.name = name
        self.off = off
        self.deff = deff
        self.fame = fame
        self.salary = salary
        self.age = age
        self.health = 100
        self.shapley_val = 0.0

        # 核心伤病属性
        self.is_core = name in ["Caitlin Clark", "Aliyah Boston", "Kelsey Mitchell", "Twin Star"]
        self.is_clark = name == "Caitlin Clark"
        self.is_twin_star = name == "Twin Star"  # 双子星标记
        self.is_injured = False
        self.injury_level = 0
        self.injury_impact = 0.0

        # 分阶段调整Clark伤病率
        self.base_clark_injury_prob = 0.30  # 初始伤病率
        self.current_clark_injury_prob = 0.30  # 实时伤病率（可动态调整）

        # 新增：记录个人伤病史
        self.injury_history = []  # 格式：[(年份, 伤病等级, 影响系数), ...]

    @property
    def ability(self):
        """强化Clark伤病对能力的影响，降低板凳球员影响"""
        base_ability = (self.off + self.deff) / 2
        if self.is_injured:
            # 影响系数分层：Clark(2.5倍) > 其他核心(2倍) > 板凳(0.8倍)
            if self.is_clark:
                impact = self.injury_impact * 2.5
            elif self.is_core:
                impact = self.injury_impact * 2
            else:
                impact = self.injury_impact * 0.8  # 板凳球员伤病影响降低
            base_ability *= (1 - impact)
        return base_ability

    def evolve(self):
        """调整能力演化：巅峰期27-32岁，32岁后才下滑"""
        if self.age < 27:  # 上升期
            g = np.random.uniform(-1, 3)  # 小幅波动
            self.off = min(99, max(40, self.off + g))
            self.deff = min(99, max(40, self.deff + g))
            self.fame = min(99, max(10, self.fame + np.random.uniform(-0.5, 1)))
        elif self.age <= 32:  # 巅峰期（27-32岁）
            self.off += np.random.uniform(-1, 1)  # 小幅波动，保证巅峰期稳定
            self.deff += np.random.uniform(-1, 1)
        else:  # 32岁后下滑
            d = np.random.uniform(1, 4)  # 下滑幅度略降
            self.off = max(40, self.off - d)
            self.deff = max(40, self.deff - d)
            self.fame = max(10, self.fame - np.random.uniform(1, 2))
        self.age += 1

    def generate_season_injury(self, year):
        """分层伤病概率+限制重度伤病比例"""
        self.is_injured = False
        self.injury_level = 0
        self.injury_impact = 0.0

        # 分层伤病概率：使用实时伤病率
        if self.is_clark:
            injury_prob = self.current_clark_injury_prob  # 动态调整的Clark伤病率
        elif self.is_core:
            injury_prob = 0.25
        else:
            injury_prob = 0.05  # 板凳球员低概率

        if random.random() < injury_prob:
            self.is_injured = True
            # 伤病等级权重：Clark更易重度伤病，板凳几乎无重度
            if self.is_clark:
                self.injury_level = random.choices([1, 2, 3], weights=[0.1, 0.2, 0.7])[0]
            elif self.is_core:
                self.injury_level = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
            else:
                self.injury_level = random.choices([1, 2, 3], weights=[0.8, 0.2, 0.0])[0]

            if self.injury_level == 1:
                self.injury_impact = 0.1
            elif self.injury_level == 2:
                self.injury_impact = 0.3
            elif self.injury_level == 3:
                self.injury_impact = 1.0

        # 记录个人伤病史
        self.injury_history.append((year, self.injury_level, self.injury_impact))

    def update_clark_injury_prob(self, new_prob):
        """动态更新Clark的伤病率（用于实施医疗团队方案）"""
        if self.is_clark:
            self.current_clark_injury_prob = new_prob


# ===============================
# 2. 球队类：新增双子星补偿逻辑
# ===============================
class Team:
    def __init__(self, roster, brand, config, use_twin_star=False):
        self.roster = roster
        self.brand = brand
        self.config = config
        self.use_twin_star = use_twin_star  # 是否启用双子星补偿

    def play_season(self, year):
        self.wins = 0
        self.gate_revenue = 0.0
        self.media_revenue = 0.0
        self.merch_revenue = 0.0

        # 生成伤病
        for p in self.roster:
            p.generate_season_injury(year)
            if p.injury_level == 3:
                p.health = 0
            else:
                p.health = 100

        # 限制伤病人数，避免球队瘫痪
        core_injured = [p for p in self.roster if p.is_core and p.is_injured]
        bench_injured = [p for p in self.roster if not p.is_core and p.is_injured]

        if len(core_injured) > 2:
            for p in random.sample(core_injured[2:], len(core_injured) - 2):
                p.is_injured = False
                p.injury_level = 0
                p.injury_impact = 0.0
                p.injury_history[-1] = (year, 0, 0.0)
        if len(core_injured) + len(bench_injured) > 5:
            excess = len(core_injured) + len(bench_injured) - 5
            for p in random.sample(bench_injured, min(excess, len(bench_injured))):
                p.is_injured = False
                p.injury_level = 0
                p.injury_impact = 0.0
                p.injury_history[-1] = (year, 0, 0.0)

        # 重新统计伤病情况
        self.core_injured_severe = sum(1 for p in self.roster if p.is_core and p.injury_level >= 2)
        self.core_injured_any = sum(1 for p in self.roster if p.is_core and p.is_injured)
        self.clark_injured = 1 if any(p.is_clark and p.is_injured for p in self.roster) else 0
        self.twin_star_available = 1 if any(p.is_twin_star and not p.is_injured for p in self.roster) else 0

        # 模拟比赛
        for _ in range(self.config['GAMES']):
            self._play_one_game()

        # 媒体收入：Clark伤病惩罚 + 双子星补偿
        base_media = 10_000_000
        brand_bonus = 30_000_000 * self.brand / (self.brand + 200)
        media_penalty = 0.2 * self.core_injured_severe
        if self.clark_injured:
            media_penalty += 0.15  # Clark伤病额外罚15%
            # 双子星补偿：抵消50%Clark伤病惩罚
            if self.use_twin_star and self.twin_star_available:
                media_penalty -= 0.075
        self.media_revenue = (base_media + brand_bonus) * (1 - media_penalty)

        # 周边收入：Clark伤病惩罚 + 双子星补偿
        total_fame = sum(p.fame for p in self.roster if not (p.is_injured and p.injury_level == 3))
        core_merch_penalty = 0.25 * self.core_injured_any
        bench_merch_penalty = 0.05 * len([p for p in self.roster if not p.is_core and p.is_injured])
        if self.clark_injured:
            core_merch_penalty += 0.2  # Clark伤病额外罚20%
            # 双子星补偿：抵消60%Clark伤病惩罚
            if self.use_twin_star and self.twin_star_available:
                core_merch_penalty -= 0.12
        self.merch_revenue = 22_000 * total_fame * (1 - core_merch_penalty - bench_merch_penalty)

    def _play_one_game(self):
        """比赛模拟逻辑（保持不变）"""
        healthy = [p for p in self.roster if p.health > 60 and p.injury_level != 3] or self.roster[:]

        # 双子星优先补位Clark
        w = self.config['W1_WIN']
        if self.clark_injured and self.use_twin_star and self.twin_star_available:
            rotation = sorted(healthy, key=lambda p:
            (1.2 if p.is_twin_star else 1) * (w * p.ability + (1 - w) * p.fame),
                              reverse=True)[:8]
        else:
            rotation = sorted(healthy, key=lambda p: w * p.ability + (1 - w) * p.fame, reverse=True)[:8]

        strength = np.mean([p.ability for p in rotation])
        opp_strength = np.random.normal(77, 7)
        win_prob = 1 / (1 + np.exp(-(strength - opp_strength) / 6))
        if random.random() < win_prob:
            self.wins += 1

        # 门票收入计算
        year_idx = self.config['YEAR_INDEX']
        base_price = 45 * (1.03 ** year_idx)
        ratio = self.config['TICKET_PRICE'] / base_price
        elasticity = 1 / (1 + np.exp(3 * (ratio - 1)))

        has_healthy_core = any(p.is_core and not p.is_injured for p in rotation)
        core_injured_penalty = 0.25 * self.core_injured_severe
        # 双子星补偿门票收入
        if self.clark_injured and self.use_twin_star and self.twin_star_available:
            core_injured_penalty -= 0.1

        attend_rate = min(0.99, max(0.2,
                                    (0.52 + self.brand / 420 + (
                                        0.18 if has_healthy_core else 0)) * elasticity - core_injured_penalty))
        attend_rate *= np.random.uniform(0.95, 1.05)
        attendance = self.config['ARENA_CAPACITY'] * attend_rate

        max_price = base_price * 1.28
        ticket_price = min(self.config['TICKET_PRICE'], max_price)
        self.gate_revenue += attendance * ticket_price

        # 更新健康值
        for p in self.roster:
            if p.injury_level >= 2:
                if p in rotation:
                    p.health -= np.random.uniform(1.2, 3.5)
                else:
                    p.health = min(100, p.health + 1.2)
            else:
                if p in rotation:
                    p.health -= np.random.uniform(0.6, 1.8)
                else:
                    p.health = min(100, p.health + 2.8)


# ===============================
# 3. 基础工具函数（保持不变）
# ===============================
def characteristic_value(players, w1):
    if not players:
        return 0
    strength = np.mean([p.ability for p in players])
    win_val = 100 / (1 + np.exp(-(strength - 70) / 8))
    fame_val = sum(p.fame for p in players)
    return w1 * win_val + (1 - w1) * fame_val


def compute_shapley(roster, w1, samples=30):
    shap = {p.name: 0 for p in roster}
    for _ in range(samples):
        perm = copy.deepcopy(roster)
        random.shuffle(perm)
        coalition, prev_val = [], 0
        for p in perm:
            coalition.append(p)
            curr_val = characteristic_value(coalition, w1)
            shap[p.name] += curr_val - prev_val
            prev_val = curr_val
    for p in roster:
        p.shapley_val = shap[p.name] / samples


def manage_roster_salary(roster, cap):
    roster = copy.deepcopy(roster)
    roster.sort(key=lambda p: p.shapley_val / (p.salary / 1000 + 1e-6) + (5 if p.fame > 90 else 0))
    while sum(p.salary for p in roster) > cap and len(roster) > 10:
        roster.pop(0)
    return roster


def find_optimal_params(year, roster, brand, cap, config):
    compute_shapley(roster, 0.5)
    base_p = 40 * (1.03 ** year)
    prices = np.linspace(base_p * 0.9, base_p * 1.3, 5)
    strategies = np.linspace(0.2, 0.9, 5)

    best_profit, best_price, best_strat = -1e18, None, None
    for p, w in product(prices, strategies):
        profits = []
        for _ in range(4):
            sim_roster = manage_roster_salary(copy.deepcopy(roster), cap)
            cfg = config.copy()
            cfg.update({'TICKET_PRICE': p, 'W1_WIN': w, 'YEAR_INDEX': year})

            use_twin_star = any(p.is_twin_star for p in sim_roster)
            tm = Team(sim_roster, brand, cfg, use_twin_star=use_twin_star)
            tm.play_season(2023 + year + 1)
            sal = sum(p.salary for p in sim_roster)
            prof = tm.gate_revenue + tm.media_revenue + tm.merch_revenue - sal - cfg['FIXED_COST']
            prof *= np.random.uniform(0.95, 1.05)
            profits.append(prof)

        avg_profit = np.mean(profits)
        if avg_profit > best_profit:
            best_profit, best_price, best_strat = avg_profit, p, w

    return best_price, best_strat


# ===============================
# 4. 分阶段预测函数（核心修改）
# ===============================
def run_forecast_phased(strategy_type="original"):
    """
    分阶段预测函数：前3年Clark正常受伤，第4年开始实施补救措施
    :param strategy_type: 策略类型 - original(原始) / medical(医疗团队) / twin_star(双子星)
    :return: 预测结果DataFrame
    """
    # 初始阵容
    roster = [
                 Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                 Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                 Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                 Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                 Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                 Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
             ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    # 双子星策略：第4年新增（先受伤后补人）
    add_twin_star = False
    if strategy_type == "twin_star":
        add_twin_star = True

    config = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 25_000_000,
    }

    brand = 80
    cap = 1_500_000
    history = []
    prev_core_severe_injuries = 0
    clark_player = next(p for p in roster if p.is_clark)  # 获取Clark对象

    for y in range(1, 11):
        current_year = 2023 + y
        # 分阶段实施补救措施（第4年开始）
        if y == 4:
            if strategy_type == "medical":
                # 方案1：降低Clark伤病率至10%
                clark_player.update_clark_injury_prob(0.10)
                print(f"第{current_year}年：实施医疗团队方案，Clark伤病率从30%降至10%")
            elif strategy_type == "twin_star":
                # 方案2：新增双子星球员
                twin_star = Player("Twin Star", 92, 70, 90, 85_000, 23)
                roster.append(twin_star)
                print(f"第{current_year}年：实施双子星方案，新增能力接近Clark的替补球员")

        # 最优策略搜索
        price, strat = find_optimal_params(y - 1, roster, brand, cap, config)
        config.update({'TICKET_PRICE': price, 'W1_WIN': strat, 'YEAR_INDEX': y - 1})

        compute_shapley(roster, strat)
        roster = manage_roster_salary(roster, cap)

        # 运行赛季
        team = Team(roster, brand, config, use_twin_star=add_twin_star)
        team.play_season(current_year)
        revenue = team.gate_revenue + team.media_revenue + team.merch_revenue
        salary = sum(p.salary for p in roster)
        profit = revenue - salary - config['FIXED_COST']
        profit *= np.random.uniform(0.9, 1.1)

        # 品牌值计算
        brand += 1.2 * np.log(1 + team.wins) + 0.1 * np.log(1 + sum(p.fame for p in roster))
        injury_improvement = max(0, prev_core_severe_injuries - team.core_injured_severe)
        injury_recovery_bonus = injury_improvement * 0.028 * brand
        brand += injury_recovery_bonus

        # 动态衰减率（含双子星补偿）
        win_bonus = 0.018 * team.wins
        injury_penalty = 0.048 * team.core_injured_severe
        if team.clark_injured:
            injury_penalty += 0.02
            # 双子星补偿品牌衰减
            if add_twin_star and team.twin_star_available:
                injury_penalty -= 0.01

        decay_rate = 0.9 - injury_penalty + win_bonus
        decay_rate = np.clip(decay_rate, 0.85, 0.95)
        brand *= decay_rate
        brand *= np.random.uniform(0.98, 1.02)

        prev_core_severe_injuries = team.core_injured_severe

        # 记录年度数据
        history.append({
            "Year": current_year,
            "Profit_M": profit / 1e6,
            "Brand": brand,
            "Wins": team.wins,
            "Clark_Injured": team.clark_injured,
            "Strategy_Type": strategy_type
        })

        # 球员演化
        for p in roster:
            p.evolve()

    return pd.DataFrame(history)


# ===============================
# 5. 对比可视化函数（修复pad参数问题）
# ===============================
def plot_phased_comparison(df_original, df_medical, df_twin_star):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 合并数据用于绘图
    df_original['方案'] = '原始方案（无补救）'
    df_medical['方案'] = '方案1（医疗团队）'
    df_twin_star['方案'] = '方案2（双子星）'
    df_combined = pd.concat([df_original, df_medical, df_twin_star])

    # 创建2个子图：利润对比 + 品牌值对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # 子图1：年度利润对比（重点标注第4年补救节点）
    sns.lineplot(data=df_combined, x="Year", y="Profit_M", hue="方案", style="方案",
                 markers=['o', 's', '^'], linewidth=3, markersize=10, ax=ax1,
                 palette=['#e74c3c', '#2ecc71', '#3498db'])

    # 标注补救措施实施节点（第4年，2027年）
    ax1.axvline(x=2027, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='补救措施实施节点（2027年）')
    ax1.text(2027, df_combined['Profit_M'].max() * 0.9, '  开始实施补救措施',
             fontsize=11, color='black', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    ax1.set_ylabel("年度利润（百万美元）", fontsize=14)
    ax1.set_title("Clark受伤后补救方案 - 年度利润对比", fontsize=16)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12, loc='upper left')

    # 子图2：品牌值对比
    sns.lineplot(data=df_combined, x="Year", y="Brand", hue="方案", style="方案",
                 markers=['o', 's', '^'], linewidth=3, markersize=10, ax=ax2,
                 palette=['#e74c3c', '#2ecc71', '#3498db'])

    # 标注补救措施实施节点
    ax2.axvline(x=2027, color='gray', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_xlabel("年份", fontsize=14)
    ax2.set_ylabel("品牌值", fontsize=14)
    ax2.set_title("Clark受伤后补救方案 - 品牌值对比", fontsize=16)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=12, loc='upper left')

    # 总标题：移除pad参数，改用y参数调整位置（兼容低版本）
    fig.suptitle("WNBA球队Clark受伤后补救方案对比分析（2024-2033）", fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig("phased_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 6. 统计对比函数
# ===============================
def print_phased_stats(df_original, df_medical, df_twin_star):
    """输出分阶段统计对比"""
    print("=" * 120)
    print("Clark受伤后补救方案 - 核心统计对比")
    print("=" * 120)

    # 分阶段计算（前3年 vs 后7年）
    def get_phase_stats(df, name):
        df_phase1 = df[df['Year'].isin([2024, 2025, 2026])]  # 前3年（2024、2025、2026）
        df_phase2 = df[df['Year'] > 2026]  # 后7年（2027-2033）

        stats = {
            "方案名称": name,
            "前3年平均利润": df_phase1['Profit_M'].mean(),
            "后7年平均利润": df_phase2['Profit_M'].mean(),
            "后7年利润提升": (df_phase2['Profit_M'].mean() - df_phase1['Profit_M'].mean()) / df_phase1[
                'Profit_M'].mean() * 100,
            "前3年平均品牌值": df_phase1['Brand'].mean(),
            "后7年平均品牌值": df_phase2['Brand'].mean(),
            "后7年品牌值提升": (df_phase2['Brand'].mean() - df_phase1['Brand'].mean()) / df_phase1[
                'Brand'].mean() * 100,
            "Clark总伤病次数": df['Clark_Injured'].sum()
        }
        return stats

    # 计算各方案统计
    stats_original = get_phase_stats(df_original, "原始方案（无补救）")
    stats_medical = get_phase_stats(df_medical, "方案1（医疗团队）")
    stats_twin_star = get_phase_stats(df_twin_star, "方案2（双子星）")

    # 输出表格
    headers = ["方案", "前3年平均利润(百万$)", "后7年平均利润(百万$)", "后7年利润提升(%)",
               "前3年平均品牌值", "后7年平均品牌值", "后7年品牌值提升(%)", "Clark总伤病次数"]
    header_line = "| " + " | ".join([f"{h:<20}" for h in headers]) + " |"
    print(header_line)
    print("|" + "-" * (len(header_line) - 2) + "|")

    for stats in [stats_original, stats_medical, stats_twin_star]:
        row_line = "| " + " | ".join([
            f"{stats['方案名称']:<20}",
            f"{stats['前3年平均利润']:<20.2f}",
            f"{stats['后7年平均利润']:<20.2f}",
            f"{stats['后7年利润提升']:<20.1f}",
            f"{stats['前3年平均品牌值']:<20.1f}",
            f"{stats['后7年平均品牌值']:<20.1f}",
            f"{stats['后7年品牌值提升']:<20.1f}",
            f"{stats['Clark总伤病次数']:<20d}"
        ]) + " |"
        print(row_line)

    print("=" * 120)


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    # 固定随机种子（结果可复现）
    np.random.seed(42)
    random.seed(42)

    # 运行三种分阶段方案
    print("=== 运行原始方案（无补救措施）===")
    df_original = run_forecast_phased(strategy_type="original")

    print("\n=== 运行方案1（医疗团队：第4年降Clark伤病率）===")
    df_medical = run_forecast_phased(strategy_type="medical")

    print("\n=== 运行方案2（双子星：第4年新增替补）===")
    df_twin_star = run_forecast_phased(strategy_type="twin_star")

    # 生成对比可视化
    plot_phased_comparison(df_original, df_medical, df_twin_star)

    # 输出分阶段统计
    print_phased_stats(df_original, df_medical, df_twin_star)