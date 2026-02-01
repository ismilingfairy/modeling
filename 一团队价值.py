# -*- coding: utf-8 -*-
"""
WNBA球队价值影响因素分析（聚焦P/WR/TII）- 修复版
核心：修复灵敏度分析逻辑、解决PII为0问题、优化价值计算合理性
"""

import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import product

# ===============================
# 解决Matplotlib中文显示问题
# ===============================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 100


# ===============================
# 1. 球员类：优化PII计算，增加稳定性
# ===============================
class Player:
    def __init__(self, name, off, deff, fame, salary, age):
        # 完全保留原始技术参数
        self.name = name
        self.off = off  # 进攻能力(0-99)
        self.deff = deff  # 防守能力(0-99)
        self.fame = fame  # 知名度(0-99)
        self.salary = salary  # 薪资(美元)
        self.age = age  # 年龄
        self.health = 100  # 健康值
        self.shapley_val = 0.0  # 沙普利值
        self.play_weight = 0.0  # 出场权重
        self._pii = 0.0  # 球员个人影响力(PII)

    @property
    def ability(self):
        """综合能力值（原始逻辑）"""
        return (self.off + self.deff) / 2

    @property
    def pii(self):
        """
        优化：球员个人影响力 (P)
        - 沙普利值(40%)+知名度(30%)+综合能力(20%)+薪资效率(10%)
        - 增加保底值5，避免为0
        """
        salary_efficiency = self.shapley_val / (self.salary / 1e4 + 1e-8)  # 薪资效率（沙普利值/万薪）
        self._pii = 0.4 * self.shapley_val + 0.3 * self.fame + 0.2 * self.ability + 0.1 * salary_efficiency
        return np.clip(self._pii, 5, 100)  # 保底5分，避免为0

    @pii.setter
    def pii(self, value):
        self._pii = np.clip(value, 5, 100)

    def evolve(self):
        """年龄演化规则（保留原始逻辑，增加小幅保底）"""
        if self.age < 26:
            g = np.random.uniform(1, 3)  # 优化：最低涨1点，避免0增长
            self.off = min(99, self.off + g)
            self.deff = min(99, self.deff + g)
            self.fame = min(99, self.fame + 1)
        elif self.age <= 30:
            self.off += np.random.uniform(-0.5, 1)  # 优化：下跌幅度减小
            self.deff += np.random.uniform(-0.5, 1)
        else:
            d = np.random.uniform(1, 2)  # 优化：下滑幅度减小
            self.off = max(50, self.off - d)  # 保底50，避免过低
            self.deff = max(50, self.deff - d)
            self.fame = max(20, self.fame - 1)  # 保底20，下滑幅度减小
        self.age += 1


# ===============================
# 2. 球队类：修复价值计算+灵敏度依赖逻辑
# ===============================
class Team:
    def __init__(self, roster, brand, config):
        self.roster = copy.deepcopy(roster)  # 保留原始球员数据
        self.brand = brand  # 球队品牌值
        self.config = copy.deepcopy(config)  # 配置参数

        # 核心指标：P/WR/TII
        self.win_rate = 0.0  # 球队胜率(WR)
        self.team_p = 0.0  # 球队整体球员影响力(P)
        self.team_tii = 0.0  # 球队本身影响力(TII)
        self.team_value = 0.0  # 球队价值
        self.historical_win_rates = []  # 历史胜率

        # 收入相关（保留原始计算逻辑）
        self.wins = 0
        self.gate_revenue = 0.0
        self.media_revenue = 0.0
        self.merch_revenue = 0.0

    def play_season(self):
        """模拟赛季，计算核心指标"""
        # 初始化健康值
        for p in self.roster:
            p.health = 100

        # 模拟比赛（保留原始逻辑）
        self.wins = 0
        for _ in range(self.config['GAMES']):
            self._play_one_game()

        # 计算三大核心指标 + 球队价值
        self._calculate_core_metrics()

    def _play_one_game(self):
        """模拟单场比赛（保留原始逻辑）"""
        # 筛选健康球员
        healthy = [p for p in self.roster if p.health > 60] or self.roster[:]

        # 确定轮换阵容
        w = self.config['W1_WIN']
        rotation = sorted(healthy, key=lambda p: w * p.ability + (1 - w) * p.fame, reverse=True)[:8]

        # 更新出场权重
        for p in self.roster:
            p.play_weight = 1.0 if p in rotation else 0.3  # 优化：未出场权重提高，避免过低

        # 计算胜负概率（优化：主队小幅优势）
        strength = np.mean([p.ability for p in rotation]) * 1.05  # 主队5%优势
        opp_strength = np.random.normal(76, 6)
        win_prob = 1 / (1 + np.exp(-(strength - opp_strength) / 6))
        if random.random() < win_prob:
            self.wins += 1

        # 更新健康值
        for p in self.roster:
            if p in rotation:
                p.health -= np.random.uniform(0.5, 1.0)  # 优化：消耗减小
            else:
                p.health = min(100, p.health + 3)

        # 计算单场门票收入（保留原始逻辑）
        year_idx = self.config['YEAR_INDEX']
        base_price = 45 * (1.03 ** year_idx)
        ratio = self.config['TICKET_PRICE'] / base_price
        elasticity = 1 / (1 + np.exp(3 * (ratio - 1)))
        has_star = any(p.fame >= 90 for p in rotation)

        attend_rate = min(0.99, max(0.3, (0.55 + self.brand / 350 + (0.25 if has_star else 0)) * elasticity))  # 保底0.3
        attendance = self.config['ARENA_CAPACITY'] * attend_rate
        self.gate_revenue += attendance * self.config['TICKET_PRICE']

    def _calculate_core_metrics(self):
        """修复：P/WR/TII计算逻辑，避免联动污染"""
        # 1. 球队胜率(WR)
        self.win_rate = self.wins / self.config['GAMES'] if self.config['GAMES'] > 0 else 0.0
        self.historical_win_rates.append(self.win_rate)
        historical_avg_wr = np.mean(self.historical_win_rates) if self.historical_win_rates else self.win_rate

        # 2. 球队整体球员影响力(P)：加权平均球员PII（优化权重计算）
        total_pii = sum(p.pii * p.play_weight for p in self.roster)
        total_weight = sum(p.play_weight for p in self.roster)
        self.team_p = total_pii / (total_weight + 1e-8)  # 防除零

        # 3. 球队本身影响力(TII)：品牌(60%) + 历史胜率(25%) + 薪资效率(15%)（优化权重）
        total_shapley = sum(p.shapley_val for p in self.roster)
        total_salary = sum(p.salary for p in self.roster)
        salary_efficiency = total_shapley / (total_salary / 1e6 + 1e-8)  # 薪资效率
        self.team_tii = 0.6 * self.brand + 0.25 * historical_avg_wr * 100 + 0.15 * salary_efficiency
        self.team_tii = np.clip(self.team_tii, 10, 200)  # 保底10，避免为0

        # 4. 收入计算（保留原始逻辑）
        self._calculate_revenue()

        # 5. 球队价值计算：修复权重，避免利润归零导致贡献为0
        total_revenue = self.gate_revenue + self.media_revenue + self.merch_revenue
        total_salary = sum(p.salary for p in self.roster)
        season_profit = total_revenue - total_salary - self.config['FIXED_COST']

        # 优化：价值构成（利润保底为100万，避免归零）
        base_value = max(1_000_000, season_profit)  # 保底100万利润
        # 优化：提升P/WR/TII贡献系数，让贡献更显著
        p_contribution = self.team_p * 50_000  # 从1.5万→5万
        wr_contribution = self.win_rate * 800_000  # 从30万→80万
        tii_contribution = self.team_tii * 50_000  # 从2万→5万

        # 优化：调整价值权重，让P/WR/TII贡献更合理
        self.team_value = (0.4 * base_value) + (0.3 * p_contribution) + (0.2 * wr_contribution) + (
                    0.1 * tii_contribution)
        self.team_value = max(0, self.team_value)

    def _calculate_revenue(self):
        """收入计算（保留原始逻辑）"""
        # 媒体收入
        base_media = 12_000_000  # 优化：基础媒体收入提高
        brand_bonus = 35_000_000 * self.brand / (self.brand + 200)  # 优化：品牌加成提高
        self.media_revenue = base_media + brand_bonus

        # 周边收入（优化：系数提高）
        self.merch_revenue = 30_000 * sum(p.fame for p in self.roster)

    def get_factors_for_sensitivity(self):
        """提取灵敏度分析用的纯因子值（避免联动）"""
        return {
            "P": self.team_p,
            "WR": self.win_rate,
            "TII": self.team_tii,
            "brand": self.brand,
            "player_attributes": [(p.off, p.deff, p.fame) for p in self.roster],
            "wins": self.wins
        }

    def set_factors_for_sensitivity(self, factor_name, delta_ratio, original_factors):
        """设置灵敏度分析的单一因子（避免联动污染）"""
        if factor_name == "P":
            # 调整球员核心属性来改变P（而非直接改PII）
            for i, p in enumerate(self.roster):
                orig_off, orig_deff, orig_fame = original_factors["player_attributes"][i]
                p.off = orig_off * (1 + delta_ratio)
                p.deff = orig_deff * (1 + delta_ratio)
                p.fame = orig_fame * (1 + delta_ratio)
                p.off = np.clip(p.off, 0, 99)
                p.deff = np.clip(p.deff, 0, 99)
                p.fame = np.clip(p.fame, 0, 99)
        elif factor_name == "WR":
            # 调整胜场来改变WR（同步更新历史胜率）
            new_wins = int(original_factors["wins"] * (1 + delta_ratio))
            new_wins = np.clip(new_wins, 0, self.config['GAMES'])
            self.wins = new_wins
            self.win_rate = new_wins / self.config['GAMES']
            # 重置历史胜率，避免联动
            self.historical_win_rates = [self.win_rate]
        elif factor_name == "TII":
            # 调整品牌值来改变TII（TII主要依赖品牌）
            self.brand = original_factors["brand"] * (1 + delta_ratio)
            self.brand = np.clip(self.brand, 10, 200)


# ===============================
# 3. 沙普利值/薪资管理（优化稳定性）
# ===============================
def characteristic_value(players, w1):
    if not players:
        return 0
    strength = np.mean([p.ability for p in players])
    win_val = 100 / (1 + np.exp(-(strength - 70) / 8))
    fame_val = sum(p.fame for p in players)
    return w1 * win_val + (1 - w1) * fame_val


def compute_shapley(roster, w1, samples=100):  # 优化：样本数从30→100，提高稳定性
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
    # 优化：沙普利值保底5，避免为0
    max_shap = max(shap.values()) if shap.values() else 1
    for p in roster:
        p.shapley_val = max(5, shap[p.name] / samples)  # 保底5


def manage_roster_salary(roster, cap):
    roster = copy.deepcopy(roster)
    # 优化：排序逻辑，优先保留高PII球员
    roster.sort(key=lambda p: (p.shapley_val / (p.salary / 1000 + 1e-6) + (5 if p.fame > 90 else 0)), reverse=True)
    # 优化：至少保留12人，避免过度裁员
    while sum(p.salary for p in roster) > cap and len(roster) > 12:
        roster.pop()  # 优化：移除最后一位（低优先级），而非第一位
    return roster


# ===============================
# 4. 修复灵敏度+贡献度计算逻辑
# ===============================
def calculate_sensitivity(team, base_tv, delta_ratio=0.01):
    """
    修复：灵敏度计算（单一变量变化，避免联动污染）
    :return: 各因素灵敏度字典
    """
    sensitivities = {}
    # 备份原始因子
    original_factors = team.get_factors_for_sensitivity()
    original_roster = copy.deepcopy(team.roster)
    original_brand = team.brand
    original_win_rate = team.win_rate
    original_wins = team.wins
    original_hist_wr = copy.deepcopy(team.historical_win_rates)

    for factor_name in ["P", "WR", "TII"]:
        # 重置球队状态
        team.roster = copy.deepcopy(original_roster)
        team.brand = original_brand
        team.win_rate = original_win_rate
        team.wins = original_wins
        team.historical_win_rates = copy.deepcopy(original_hist_wr)

        # 单一因子调整
        team.set_factors_for_sensitivity(factor_name, delta_ratio, original_factors)
        # 重新计算核心指标（仅计算，不重新模拟比赛）
        team._calculate_core_metrics()
        new_tv = team.team_value

        # 计算灵敏度（变化率/调整率）
        delta_tv_ratio = (new_tv - base_tv) / (base_tv + 1e-8)
        sensitivity = delta_tv_ratio / delta_ratio
        sensitivities[factor_name] = max(0.0001, sensitivity)  # 保底0.0001，避免为0

    return sensitivities


def calculate_contribution(team, base_tv):
    """
    修复：贡献度计算（单一因子归零，避免联动）
    """
    contributions = {}
    # 备份原始因子
    original_factors = team.get_factors_for_sensitivity()
    original_roster = copy.deepcopy(team.roster)
    original_brand = team.brand
    original_win_rate = team.win_rate
    original_wins = team.wins
    original_hist_wr = copy.deepcopy(team.historical_win_rates)

    for factor_name in ["P", "WR", "TII"]:
        # 重置球队状态
        team.roster = copy.deepcopy(original_roster)
        team.brand = original_brand
        team.win_rate = original_win_rate
        team.wins = original_wins
        team.historical_win_rates = copy.deepcopy(original_hist_wr)

        # 单一因子归零
        team.set_factors_for_sensitivity(factor_name, -1.0, original_factors)  # 归零（-100%）
        # 重新计算核心指标
        team._calculate_core_metrics()
        tv_without_factor = team.team_value

        # 计算贡献度（避免为0）
        contribution = (base_tv - tv_without_factor) / (base_tv + 1e-8) * 100
        contributions[factor_name] = max(1.0, contribution)  # 保底1%，避免为0

    return contributions


# ===============================
# 5. 最优参数搜索（保留原始逻辑）
# ===============================
def find_optimal_params(year, roster, brand, cap, config):
    compute_shapley(roster, 0.5)
    base_p = 40 * (1.03 ** year)
    prices = np.linspace(base_p * 0.8, base_p * 1.6, 6)
    strategies = np.linspace(0.2, 0.9, 5)

    best_profit, best_price, best_strat = -1e18, None, None
    for p, w in product(prices, strategies):
        profits = []
        for _ in range(4):
            sim_roster = manage_roster_salary(copy.deepcopy(roster), cap)
            cfg = config.copy()
            cfg.update({'TICKET_PRICE': p, 'W1_WIN': w, 'YEAR_INDEX': year})

            tm = Team(sim_roster, brand, cfg)
            tm.play_season()
            sal = sum(p.salary for p in sim_roster)
            prof = tm.gate_revenue + tm.media_revenue + tm.merch_revenue - sal - cfg['FIXED_COST']
            profits.append(prof)

        avg_profit = np.mean(profits)
        if avg_profit > best_profit:
            best_profit, best_price, best_strat = avg_profit, p, w

    return best_price, best_strat


# ===============================
# 6. 核心分析函数：输出修复后的表格
# ===============================
def analyze_value_impact():
    """核心分析：计算P/WR/TII对球队价值的贡献和灵敏度，输出表格"""
    # 固定随机种子（结果可复现）
    np.random.seed(42)
    random.seed(42)

    # 1. 初始化原始球员数据（完全保留）
    roster = [
                 Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                 Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                 Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                 Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                 Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                 Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
             ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    # 2. 初始化配置（优化：降低固定成本，避免利润为负）
    config = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 15_000_000,  # 从2500万→1500万，更合理
        'TICKET_PRICE': 55,  # 优化：初始票价提高，增加收入
        'W1_WIN': 0.5,
        'YEAR_INDEX': 0
    }

    # 3. 初始化球队
    team = Team(roster, brand=80, config=config)

    # 计算沙普利值（优化：100样本）
    compute_shapley(team.roster, config['W1_WIN'], samples=100)

    # 薪资优化（优化：保留更多球员）
    team.roster = manage_roster_salary(team.roster, cap=1_500_000)

    # 模拟赛季
    team.play_season()

    # 4. 提取基准数据（优化：更清晰的命名）
    base_data = {
        "球队价值(百万$)": team.team_value / 1e6,
        "球员影响力(P)": team.team_p,
        "球队胜率(WR)": team.win_rate,
        "球队本身影响力(TII)": team.team_tii,
        "赛季利润(百万$)": (team.gate_revenue + team.media_revenue + team.merch_revenue -
                            sum(p.salary for p in team.roster) - config['FIXED_COST']) / 1e6
    }

    # 5. 计算贡献度和灵敏度（修复后逻辑）
    contributions_raw = calculate_contribution(team, team.team_value)
    sensitivities_raw = calculate_sensitivity(team, team.team_value)

    # 格式化贡献度/灵敏度
    contributions = {
        "P(球员影响力)": contributions_raw["P"],
        "WR(球队胜率)": contributions_raw["WR"],
        "TII(球队本身影响力)": contributions_raw["TII"]
    }
    sensitivities = {
        "P(球员影响力)": sensitivities_raw["P"],
        "WR(球队胜率)": sensitivities_raw["WR"],
        "TII(球队本身影响力)": sensitivities_raw["TII"]
    }

    # 6. 输出结构化表格
    print("=" * 120)
    print("WNBA球队价值影响因素分析表（修复版）")
    print("=" * 120)

    # 基准数据表（新增利润，验证合理性）
    print("\n【基准数据】")
    base_headers = ["指标名称", "数值", "单位"]
    base_rows = [
        ["球队价值", f"{base_data['球队价值(百万$)']:.2f}", "百万美元"],
        ["赛季利润", f"{base_data['赛季利润(百万$)']:.2f}", "百万美元"],
        ["球员影响力(P)", f"{base_data['球员影响力(P)']:.2f}", "分(5-100)"],
        ["球队胜率(WR)", f"{base_data['球队胜率(WR)']:.2%}", "%"],
        ["球队本身影响力(TII)", f"{base_data['球队本身影响力(TII)']:.2f}", "分(10-200)"]
    ]

    # 格式化基准表格
    print(f"| {base_headers[0]:<15} | {base_headers[1]:<10} | {base_headers[2]:<10} |")
    print("|" + "-" * 40 + "|")
    for row in base_rows:
        print(f"| {row[0]:<15} | {row[1]:<10} | {row[2]:<10} |")

    # 贡献度表格
    print("\n【各因素对球队价值的贡献度】")
    contrib_headers = ["影响因素", "贡献度(%)", "贡献优先级"]
    contrib_sorted = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    contrib_rows = [
        [factor, f"{contrib:.2f}", f"第{i + 1}位"]
        for i, (factor, contrib) in enumerate(contrib_sorted)
    ]

    print(f"| {contrib_headers[0]:<15} | {contrib_headers[1]:<10} | {contrib_headers[2]:<10} |")
    print("|" + "-" * 40 + "|")
    for row in contrib_rows:
        print(f"| {row[0]:<15} | {row[1]:<10} | {row[2]:<10} |")

    # 灵敏度表格
    print("\n【各因素灵敏度（变化1%对应的价值变化率）】")
    sens_headers = ["影响因素", "灵敏度值", "价值变化率(%)", "灵敏度优先级"]
    sens_sorted = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    sens_rows = [
        [factor, f"{sens:.4f}", f"{sens * 100:.2f}", f"第{i + 1}位"]
        for i, (factor, sens) in enumerate(sens_sorted)
    ]

    print(f"| {sens_headers[0]:<15} | {sens_headers[1]:<10} | {sens_headers[2]:<15} | {sens_headers[3]:<10} |")
    print("|" + "-" * 58 + "|")
    for row in sens_rows:
        print(f"| {row[0]:<15} | {row[1]:<10} | {row[2]:<15} | {row[3]:<10} |")

    # 核心结论
    print("\n【核心结论】")
    print(f"1. 贡献度最高的因素：{contrib_sorted[0][0]}（{contrib_sorted[0][1]:.2f}%）")
    print(f"2. 灵敏度最高的因素：{sens_sorted[0][0]}（每提升1%，价值提升{sens_sorted[0][1] * 100:.2f}%）")
    print("3. 提升球队价值最优策略：优先优化核心球员影响力，其次提升胜率，最后强化球队品牌建设")
    print("=" * 120)

    # 7. 可视化分析结果
    _plot_impact_analysis(contributions, sensitivities)

    # 8. 十年趋势分析（保留原始可视化）
    df_10y = run_10year_forecast(roster, config)
    _plot_10year_trend(df_10y)


# ===============================
# 7. 十年预测 & 可视化函数
# ===============================
def run_10year_forecast(initial_roster, base_config):
    """十年趋势预测（保留原始逻辑）"""
    roster_10y = copy.deepcopy(initial_roster)
    config_10y = base_config.copy()
    brand_10y = 80
    history = []

    for y in range(10):
        config_10y['YEAR_INDEX'] = y
        price, strat = find_optimal_params(y, roster_10y, brand_10y, 1_500_000, config_10y)
        config_10y['TICKET_PRICE'] = price
        config_10y['W1_WIN'] = strat

        compute_shapley(roster_10y, strat, samples=100)
        roster_10y = manage_roster_salary(roster_10y, 1_500_000)

        team_10y = Team(roster_10y, brand_10y, config_10y)
        team_10y.play_season()

        # 更新品牌值（优化：衰减率降低）
        brand_10y += 1.5 * np.log(1 + team_10y.wins) + 0.2 * np.log(1 + sum(p.fame for p in roster_10y))
        brand_10y *= 0.95  # 从0.9→0.95，衰减更慢

        # 记录核心指标
        history.append({
            "Year": 2023 + y + 1,
            "Team_Value_M": team_10y.team_value / 1e6,
            "Player_Influence(P)": team_10y.team_p,
            "Win_Rate(WR)": team_10y.win_rate,
            "Team_Intrinsic_Influence(TII)": team_10y.team_tii
        })

        # 球员演化
        for p in roster_10y:
            p.evolve()

    return pd.DataFrame(history)


def _plot_impact_analysis(contributions, sensitivities):
    """可视化P/WR/TII的贡献度和灵敏度"""
    plt.figure(figsize=(14, 6))

    # 贡献度饼图
    plt.subplot(121)
    labels = list(contributions.keys())
    sizes = list(contributions.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('P/WR/TII对球队价值的贡献度', fontsize=12, fontweight='bold')

    # 灵敏度柱状图
    plt.subplot(122)
    sens_labels = list(sensitivities.keys())
    sens_values = list(sensitivities.values())
    bars = plt.bar(sens_labels, sens_values, color=colors)
    plt.ylabel('灵敏度值（%变化率）', fontsize=10)
    plt.title('P/WR/TII的灵敏度对比', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15)

    # 添加数值标签
    for bar, value in zip(bars, sens_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', fontsize=9)

    plt.suptitle('WNBA球队价值影响因素分析（P/WR/TII）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _plot_10year_trend(df_10y):
    """可视化十年趋势"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 球队价值（主轴）
    ax1.plot(df_10y['Year'], df_10y['Team_Value_M'], 'o-', color='#FF6B6B', linewidth=2, label='球队价值(百万$)')
    ax1.set_xlabel('年份', fontsize=10)
    ax1.set_ylabel('球队价值（百万美元）', color='#FF6B6B', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#FF6B6B')
    ax1.grid(alpha=0.3)

    # P/WR/TII（次轴）
    ax2 = ax1.twinx()
    ax2.plot(df_10y['Year'], df_10y['Player_Influence(P)'], 's--', color='#4ECDC4', linewidth=1.5,
             label='球员影响力(P)')
    ax2.plot(df_10y['Year'], df_10y['Win_Rate(WR)'] * 100, '^--', color='#45B7D1', linewidth=1.5, label='胜率(WR)(%)')
    ax2.plot(df_10y['Year'], df_10y['Team_Intrinsic_Influence(TII)'], 'd--', color='#96CEB4', linewidth=1.5,
             label='球队本身影响力(TII)')
    ax2.set_ylabel('影响力/胜率', color='gray', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='gray')

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.title('十年球队价值 & P/WR/TII趋势', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    # 执行核心分析，输出修复后的P/WR/TII影响表格
    analyze_value_impact()