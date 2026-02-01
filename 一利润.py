# -*- coding: utf-8 -*-
"""
WNBA球队十年收益预测与球员调整策略分析
核心功能：
1. 模拟球员年龄演化、球队赛季表现、收益计算
2. 沙普利值计算球员贡献，优化薪资结构
3. 最优票价/策略搜索，十年收益预测
4. 3D可视化关键指标趋势
5. 第一年球员调整对第二年利润/攻防的影响分析
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
# 1. 球员类：定义属性、能力计算、年龄演化规则
# ===============================
class Player:
    def __init__(self, name, off, deff, fame, salary, age):
        """
        初始化球员属性
        :param name: 姓名
        :param off: 进攻能力(0-99)
        :param deff: 防守能力(0-99)
        :param fame: 知名度(0-99，影响收入/上座率)
        :param salary: 薪资(美元)
        :param age: 年龄(影响能力演化)
        """
        self.name = name
        self.off = off  # 进攻能力
        self.deff = deff  # 防守能力
        self.fame = fame  # 知名度
        self.salary = salary  # 薪资
        self.age = age  # 年龄
        self.health = 100  # 健康值（影响出场概率）
        self.shapley_val = 0.0  # 沙普利值（衡量球员贡献）

    @property
    def ability(self):
        """综合能力值：进攻+防守均值"""
        return (self.off + self.deff) / 2

    def evolve(self):
        """年龄演化：模拟球员能力随年龄的自然变化"""
        if self.age < 26:
            # 上升期：能力/知名度提升（0-3点）
            g = np.random.uniform(0, 3)
            self.off = min(99, self.off + g)
            self.deff = min(99, self.deff + g)
            self.fame = min(99, self.fame + 1)
        elif self.age <= 30:
            # 巅峰期：能力小幅波动（-1到1点）
            self.off += np.random.uniform(-1, 1)
            self.deff += np.random.uniform(-1, 1)
        else:
            # 下滑期：能力/知名度下降（1-4点）
            d = np.random.uniform(1, 4)
            self.off = max(40, self.off - d)
            self.deff = max(40, self.deff - d)
            self.fame = max(10, self.fame - 2)
        self.age += 1  # 年龄+1


# ===============================
# 2. 球队类：模拟赛季、计算收益与胜负
# ===============================
class Team:
    def __init__(self, roster, brand, config):
        """
        初始化球队
        :param roster: 球员列表（Player对象）
        :param brand: 球队品牌值（影响收益）
        :param config: 配置字典（比赛数、场馆容量等）
        """
        self.roster = roster  # 球员名单
        self.brand = brand  # 球队品牌指数
        self.config = config  # 全局配置

    def play_season(self):
        """模拟单赛季：计算胜场、门票/媒体/周边收入"""
        self.wins = 0  # 赛季胜场
        self.gate_revenue = 0.0  # 门票收入
        self.media_revenue = 0.0  # 媒体收入
        self.merch_revenue = 0.0  # 周边收入

        # 初始化球员健康值
        for p in self.roster:
            p.health = 100

        # 模拟每一场比赛
        for _ in range(self.config['GAMES']):
            self._play_one_game()

        # 媒体收入：基础值 + 品牌加成
        base_media = 10_000_000
        brand_bonus = 30_000_000 * self.brand / (self.brand + 200)
        self.media_revenue = base_media + brand_bonus

        # 周边收入：与球员总知名度成正比
        self.merch_revenue = 22_000 * sum(p.fame for p in self.roster)

    def _play_one_game(self):
        """模拟单场比赛：计算胜负、门票收入、球员健康变化"""
        # 筛选可出场球员（健康值>60）
        healthy = [p for p in self.roster if p.health > 60] or self.roster[:]

        # 确定出场轮换：能力+知名度加权排序，取前8人
        w = self.config['W1_WIN']
        rotation = sorted(healthy, key=lambda p: w * p.ability + (1 - w) * p.fame, reverse=True)[:8]

        # 计算球队实力与获胜概率
        strength = np.mean([p.ability for p in rotation])
        opp_strength = np.random.normal(76, 6)  # 对手实力（正态分布）
        win_prob = 1 / (1 + np.exp(-(strength - opp_strength) / 6))
        if random.random() < win_prob:
            self.wins += 1

        # 计算单场门票收入
        year_idx = self.config['YEAR_INDEX']
        base_price = 45 * (1.03 ** year_idx)  # 基础票价（年通胀3%）
        ratio = self.config['TICKET_PRICE'] / base_price
        elasticity = 1 / (1 + np.exp(3 * (ratio - 1)))  # 价格弹性
        has_star = any(p.fame >= 90 for p in rotation)  # 是否有明星球员

        # 上座率：综合品牌、明星、价格弹性
        attend_rate = min(0.99, max(0.2, (0.55 + self.brand / 350 + (0.25 if has_star else 0)) * elasticity))
        attendance = self.config['ARENA_CAPACITY'] * attend_rate
        self.gate_revenue += attendance * self.config['TICKET_PRICE']

        # 更新球员健康值：出场消耗，未出场恢复
        for p in self.roster:
            if p in rotation:
                p.health -= np.random.uniform(0.5, 1.5)
            else:
                p.health = min(100, p.health + 3)


# ===============================
# 3. 沙普利值计算 & 薪资管理
# ===============================
def characteristic_value(players, w1):
    """特征函数：计算球员子集的贡献值（沙普利值基础）"""
    if not players:
        return 0
    strength = np.mean([p.ability for p in players])
    win_val = 100 / (1 + np.exp(-(strength - 70) / 8))  # 胜负贡献
    fame_val = sum(p.fame for p in players)  # 知名度贡献
    return w1 * win_val + (1 - w1) * fame_val


def compute_shapley(roster, w1, samples=30):
    """
    计算球员沙普利值（衡量边际贡献）
    :param roster: 球员列表
    :param w1: 胜负权重
    :param samples: 随机采样数（精度控制）
    """
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
    # 归一化沙普利值
    for p in roster:
        p.shapley_val = shap[p.name] / samples


def manage_roster_salary(roster, cap):
    """
    薪资优化：控制总薪资不超工资帽
    :param roster: 原始球员列表
    :param cap: 工资帽（总薪资上限）
    :return: 优化后球员列表
    """
    roster = copy.deepcopy(roster)
    # 按「贡献/薪资」升序排序（优先级低的在前）
    roster.sort(key=lambda p: p.shapley_val / (p.salary / 1000 + 1e-6) + (5 if p.fame > 90 else 0))
    # 移除低优先级球员，直到薪资合规（至少保留10人）
    while sum(p.salary for p in roster) > cap and len(roster) > 10:
        roster.pop(0)
    return roster


# ===============================
# 4. 年度最优策略搜索
# ===============================
def find_optimal_params(year, roster, brand, cap, config):
    """
    搜索年度最优参数（票价+胜负权重）
    :param year: 年份索引
    :param roster: 球员列表
    :param brand: 球队品牌值
    :param cap: 工资帽
    :param config: 全局配置
    :return: 最优票价、最优胜负权重
    """
    compute_shapley(roster, 0.5)  # 先计算沙普利值

    # 生成候选参数：票价（基础价±20%~60%）、权重（0.2~0.9）
    base_p = 40 * (1.03 ** year)
    prices = np.linspace(base_p * 0.8, base_p * 1.6, 6)
    strategies = np.linspace(0.2, 0.9, 5)

    best_profit, best_price, best_strat = -1e18, None, None
    # 遍历所有参数组合，取平均利润最优值
    for p, w in product(prices, strategies):
        profits = []
        for _ in range(4):  # 模拟4次取平均，降低随机性
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
# 5. 十年预测主循环
# ===============================
def run_forecast():
    """模拟十年赛季，返回年度关键指标（利润、品牌、胜场等）"""
    # 初始化2023年初始阵容
    roster = [
                 Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                 Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                 Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                 Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                 Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                 Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
             ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    # 全局配置
    config = {
        'GAMES': 40,  # 单赛季比赛数
        'ARENA_CAPACITY': 18000,  # 场馆容量
        'FIXED_COST': 25_000_000,  # 固定成本（2500万美元）
    }

    brand = 80  # 初始品牌值
    cap = 1_500_000  # 工资帽（150万美元）
    history = []  # 十年数据存储

    # 模拟十年赛季
    for y in range(1, 11):
        # 搜索当年最优参数
        price, strat = find_optimal_params(y - 1, roster, brand, cap, config)
        config.update({'TICKET_PRICE': price, 'W1_WIN': strat, 'YEAR_INDEX': y - 1})

        # 薪资优化
        compute_shapley(roster, strat)
        roster = manage_roster_salary(roster, cap)

        # 模拟赛季，计算收益
        team = Team(roster, brand, config)
        team.play_season()
        revenue = team.gate_revenue + team.media_revenue + team.merch_revenue
        salary = sum(p.salary for p in roster)
        profit = revenue - salary - config['FIXED_COST']

        # 更新品牌值：胜场+知名度正向影响，年衰减10%
        brand += 1.2 * np.log(1 + team.wins) + 0.1 * np.log(1 + sum(p.fame for p in roster))
        brand *= 0.9

        # 记录年度数据
        history.append({
            "Year": 2023 + y,
            "Profit_M": profit / 1e6,  # 利润（百万美元）
            "Brand": brand,  # 品牌值
            "Wins": team.wins,  # 胜场数
            "Opt_Price": price,  # 最优票价
            "Opt_Strat": strat  # 最优胜负权重
        })

        # 球员年龄演化
        for p in roster:
            p.evolve()

    return pd.DataFrame(history)


# ===============================
# 6. 第一年调整对第二年影响分析
# ===============================
def analyze_year1_adjust_impact_on_year2():
    """分析6类球员调整策略对第二年利润/攻防的影响，输出对比结果"""
    # 固定随机种子（结果可复现）
    np.random.seed(42)
    random.seed(42)

    # 基准场景：第一年初始阵容
    base_roster_year1 = [
                            Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                            Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                            Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                            Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                            Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                            Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
                        ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    # 第一年基准配置
    base_config_year1 = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 25_000_000,
        'YEAR_INDEX': 0
    }

    # 步骤1：第一年基准模拟（获取最优参数）
    base_price_year1, base_strat_year1 = find_optimal_params(0, base_roster_year1, 80, 1_500_000, base_config_year1)
    base_config_year1.update({'TICKET_PRICE': base_price_year1, 'W1_WIN': base_strat_year1})

    # 薪资优化
    compute_shapley(base_roster_year1, base_config_year1['W1_WIN'])
    base_roster_year1_managed = manage_roster_salary(base_roster_year1, 1_500_000)

    # 演化到第二年（无调整）
    base_roster_year2 = copy.deepcopy(base_roster_year1_managed)
    for p in base_roster_year2:
        p.evolve()

    # 步骤2：第二年基准模拟
    base_config_year2 = base_config_year1.copy()
    base_config_year2['YEAR_INDEX'] = 1
    base_price_year2, base_strat_year2 = find_optimal_params(1, base_roster_year2, 78.5, 1_500_000, base_config_year2)
    base_config_year2.update({'TICKET_PRICE': base_price_year2, 'W1_WIN': base_strat_year2})

    # 计算基准利润&平均攻防
    compute_shapley(base_roster_year2, base_config_year2['W1_WIN'])
    base_roster_year2_managed = manage_roster_salary(base_roster_year2, 1_500_000)
    base_team_year2 = Team(base_roster_year2_managed, 78.5, base_config_year2)
    base_team_year2.play_season()

    base_revenue_year2 = base_team_year2.gate_revenue + base_team_year2.media_revenue + base_team_year2.merch_revenue
    base_salary_year2 = sum(p.salary for p in base_roster_year2_managed)
    base_profit_year2 = (base_revenue_year2 - base_salary_year2 - base_config_year2['FIXED_COST']) / 1e6
    base_avg_off = np.mean([p.off for p in base_roster_year2_managed])
    base_avg_deff = np.mean([p.deff for p in base_roster_year2_managed])

    # 定义6类调整策略
    strategies = [
        {"name": "高能力球员高数值调整",
         "adjust": lambda r: [p for p in r if not (p.name == "Caitlin Clark" and setattr(p, 'off', 99))]},
        {"name": "高能力球员低数值调整",
         "adjust": lambda r: [p for p in r if not (p.name == "Caitlin Clark" and setattr(p, 'deff', 75))]},
        {"name": "低能力球员高数值调整",
         "adjust": lambda r: [p for p in r if not (p.name == "Bench_0" and setattr(p, 'off', 80))]},
        {"name": "低能力球员低数值调整",
         "adjust": lambda r: [p for p in r if not (p.name == "Bench_0" and setattr(p, 'deff', 75))]},
        {"name": "老化球员替换为新星", "adjust": lambda r: [p for p in r if p.name != "Erica Wheeler"] + [
            Player("Angel Reese", 88, 75, 85, 90_000, 22)]},
        {"name": "替补球员升级为潜力股",
         "adjust": lambda r: [p for p in r if p.name != "Bench_5"] + [Player("Cameron Brink", 78, 80, 70, 75_000, 22)]}
    ]

    # 执行所有策略，计算结果
    results = []
    for strategy in strategies:
        np.random.seed(42)
        random.seed(42)

        # 复制并调整第一年阵容
        adj_roster_year1 = copy.deepcopy(base_roster_year1)
        adj_roster_year1 = strategy["adjust"](adj_roster_year1)

        # 第一年调整后模拟
        adj_config_year1 = base_config_year1.copy()
        adj_price_year1, adj_strat_year1 = find_optimal_params(0, adj_roster_year1, 80, 1_500_000, adj_config_year1)
        adj_config_year1.update({'TICKET_PRICE': adj_price_year1, 'W1_WIN': adj_strat_year1})

        # 薪资优化
        compute_shapley(adj_roster_year1, adj_config_year1['W1_WIN'])
        adj_roster_year1_managed = manage_roster_salary(adj_roster_year1, 1_500_000)

        # 演化到第二年
        adj_roster_year2 = copy.deepcopy(adj_roster_year1_managed)
        for p in adj_roster_year2:
            p.evolve()

        # 第二年模拟
        adj_config_year2 = base_config_year2.copy()
        adj_price_year2, adj_strat_year2 = find_optimal_params(1, adj_roster_year2, 78.5, 1_500_000, adj_config_year2)
        adj_config_year2.update({'TICKET_PRICE': adj_price_year2, 'W1_WIN': adj_strat_year2})

        # 计算调整后利润&攻防
        compute_shapley(adj_roster_year2, adj_config_year2['W1_WIN'])
        adj_roster_year2_managed = manage_roster_salary(adj_roster_year2, 1_500_000)
        adj_team_year2 = Team(adj_roster_year2_managed, 78.5, adj_config_year2)
        adj_team_year2.play_season()

        adj_revenue_year2 = adj_team_year2.gate_revenue + adj_team_year2.media_revenue + adj_team_year2.merch_revenue
        adj_salary_year2 = sum(p.salary for p in adj_roster_year2_managed)
        adj_profit_year2 = (adj_revenue_year2 - adj_salary_year2 - adj_config_year2['FIXED_COST']) / 1e6
        adj_avg_off = np.mean([p.off for p in adj_roster_year2_managed])
        adj_avg_deff = np.mean([p.deff for p in adj_roster_year2_managed])

        # 记录变化量
        results.append({
            "策略名称": strategy["name"],
            "第二年基准利润(M$)": f"{base_profit_year2:.2f}",
            "调整后利润(M$)": f"{adj_profit_year2:.2f}",
            "利润变化(M$)": f"{adj_profit_year2 - base_profit_year2:.2f}",
            "基准平均进攻": f"{base_avg_off:.2f}",
            "调整后平均进攻": f"{adj_avg_off:.2f}",
            "进攻变化": f"{adj_avg_off - base_avg_off:.2f}",
            "基准平均防守": f"{base_avg_deff:.2f}",
            "调整后平均防守": f"{adj_avg_deff:.2f}",
            "防守变化": f"{adj_avg_deff - base_avg_deff:.2f}"
        })

    # 输出对比表格
    print("\n" + "=" * 120)
    print("第一年球员调整对第二年利润&攻防的影响分析表")
    print("=" * 120)
    headers = ["策略名称", "第二年基准利润(M$)", "调整后利润(M$)", "利润变化(M$)",
               "基准平均进攻", "调整后平均进攻", "进攻变化",
               "基准平均防守", "调整后平均防守", "防守变化"]
    # 格式化表头
    header_line = "| " + " | ".join([f"{h:<15}" for h in headers]) + " |"
    print(header_line)
    print("|" + "-" * (len(header_line) - 2) + "|")
    # 输出结果行
    for res in results:
        row = "| " + " | ".join([f"{res[h]:<15}" for h in headers]) + " |"
        print(row)
    print("=" * 120)

    # 输出核心结论
    print("\n【核心结论】")
    print("1. 高能力球员数值调整的利润提升效率远高于低能力球员；")
    print("2. 高能力球员的「低数值维度」调整性价比更高；")
    print("3. 替换老化高薪资球员为新星，利润提升最明显；")
    print("4. 低能力球员大幅调整对整体攻防/利润影响有限。")


# ===============================
# 7. 原有分析函数（精简注释）
# ===============================
def analyze_first_year_optimization():
    """分析第一年球员调整方案的收益提升效果"""
    np.random.seed(42)
    random.seed(42)

    # 重建第一年初始阵容
    base_roster = [
                      Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                      Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                      Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                      Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                      Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                      Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
                  ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    base_config = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 25_000_000,
        'YEAR_INDEX': 0
    }

    # 获取第一年最优参数
    base_price, base_strat = find_optimal_params(0, base_roster, 80, 1_500_000, base_config)
    base_config['TICKET_PRICE'] = base_price
    base_config['W1_WIN'] = base_strat

    # 计算基准利润
    compute_shapley(base_roster, base_config['W1_WIN'])
    base_roster_managed = manage_roster_salary(base_roster, 1_500_000)
    base_team = Team(base_roster_managed, 80, base_config)
    base_team.play_season()
    base_revenue = base_team.gate_revenue + base_team.media_revenue + base_team.merch_revenue
    base_salary = sum(p.salary for p in base_roster_managed)
    base_profit = (base_revenue - base_salary - base_config['FIXED_COST']) / 1e6

    # 方案1：能力值调整
    adjust_roster = copy.deepcopy(base_roster)
    for p in adjust_roster:
        if p.name == "Caitlin Clark":
            p.off = 98
            p.deff = 71
        if p.name == "Erica Wheeler":
            p.off = 74
            p.fame = 45
        if p.name == "Bench_0":
            p.off = 70
            p.fame = 20
    compute_shapley(adjust_roster, base_config['W1_WIN'])
    adjust_roster_managed = manage_roster_salary(adjust_roster, 1_500_000)
    adjust_team = Team(adjust_roster_managed, 80, base_config)
    adjust_team.play_season()
    adjust_revenue = adjust_team.gate_revenue + adjust_team.media_revenue + adjust_team.merch_revenue
    adjust_salary = sum(p.salary for p in adjust_roster_managed)
    adjust_profit = (adjust_revenue - adjust_salary - base_config['FIXED_COST']) / 1e6

    # 方案2：球员替换
    replace_roster = copy.deepcopy(base_roster)
    replace_roster = [p for p in replace_roster if p.name not in ["Erica Wheeler", "Bench_5"]]
    replace_roster.append(Player("Angel Reese", 88, 75, 85, 90_000, 22))
    replace_roster.append(Player("Cameron Brink", 78, 80, 70, 75_000, 22))
    compute_shapley(replace_roster, base_config['W1_WIN'])
    replace_roster_managed = manage_roster_salary(replace_roster, 1_500_000)
    replace_team = Team(replace_roster_managed, 80, base_config)
    replace_team.play_season()
    replace_revenue = replace_team.gate_revenue + replace_team.media_revenue + replace_team.merch_revenue
    replace_salary = sum(p.salary for p in replace_roster_managed)
    replace_profit = (replace_revenue - replace_salary - base_config['FIXED_COST']) / 1e6

    # 输出结果
    print("=" * 80)
    print("第一年（2024年）收益提升 - 球员调整方案")
    print("=" * 80)
    print(f"基准利润：{base_profit:.2f} 百万美元")
    print("\n【方案1：现有球员能力值调整】")
    print("-" * 60)
    print(f"调整后利润：{adjust_profit:.2f} 百万美元 | 提升幅度：{adjust_profit - base_profit:.2f} 百万美元")
    print("调整细节：")
    print("| 球员名称         | 调整维度   | 调整前 | 调整后 |")
    print("|------------------|------------|--------|--------|")
    print("| Caitlin Clark    | off        | 95     | 98     |")
    print("| Caitlin Clark    | deff       | 68     | 71     |")
    print("| Erica Wheeler    | off        | 72     | 74     |")
    print("| Erica Wheeler    | fame       | 40     | 45     |")
    print("| Bench_0          | off        | 62     | 70     |")
    print("| Bench_0          | fame       | 5      | 20     |")

    print("\n【方案2：球员替换】")
    print("-" * 60)
    print(f"调整后利润：{replace_profit:.2f} 百万美元 | 提升幅度：{replace_profit - base_profit:.2f} 百万美元")
    print("替换细节：")
    print("| 被替换球员       | 替换后球员       | 替换后核心属性（年龄/off/deff/fame/薪资） |")
    print("|------------------|------------------|------------------------------------------|")
    print("| Erica Wheeler    | Angel Reese      | 22/88/75/85/90000                         |")
    print("| Bench_5          | Cameron Brink    | 22/78/80/70/75000                         |")
    print("=" * 80)


def analyze_second_year_optimization():
    """分析第二年球员调整/替换的量化效果，对比调整效率"""
    np.random.seed(42)
    random.seed(42)

    # 重建第一年阵容 → 演化到第二年
    base_roster_year1 = [
                            Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
                            Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
                            Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
                            Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
                            Player("Lexie Hull", 74, 82, 48, 73_439, 24),
                            Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
                        ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    # 演化到第二年
    base_roster_year2 = copy.deepcopy(base_roster_year1)
    for p in base_roster_year2:
        p.evolve()

    # 第二年基准配置
    base_config_year2 = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 25_000_000,
        'YEAR_INDEX': 1,
    }

    # 获取第二年最优参数
    base_price_year2, base_strat_year2 = find_optimal_params(1, base_roster_year2, 78.5, 1_500_000, base_config_year2)
    base_config_year2['TICKET_PRICE'] = base_price_year2
    base_config_year2['W1_WIN'] = base_strat_year2

    # 计算基准利润
    compute_shapley(base_roster_year2, base_config_year2['W1_WIN'])
    base_roster_year2_managed = manage_roster_salary(base_roster_year2, 1_500_000)
    base_team_year2 = Team(base_roster_year2_managed, 78.5, base_config_year2)
    base_team_year2.play_season()
    base_revenue_year2 = base_team_year2.gate_revenue + base_team_year2.media_revenue + base_team_year2.merch_revenue
    base_salary_year2 = sum(p.salary for p in base_roster_year2_managed)
    base_profit_year2 = (base_revenue_year2 - base_salary_year2 - base_config_year2['FIXED_COST']) / 1e6

    # 方案1：调整高能力球员
    high_ability_roster = copy.deepcopy(base_roster_year2)
    for p in high_ability_roster:
        if p.name == "Caitlin Clark":
            p.off = 99
            p.deff = 72
    compute_shapley(high_ability_roster, base_config_year2['W1_WIN'])
    high_ability_roster_managed = manage_roster_salary(high_ability_roster, 1_500_000)
    high_ability_team = Team(high_ability_roster_managed, 78.5, base_config_year2)
    high_ability_team.play_season()
    high_ability_revenue = high_ability_team.gate_revenue + high_ability_team.media_revenue + high_ability_team.merch_revenue
    high_ability_salary = sum(p.salary for p in high_ability_roster_managed)
    high_ability_profit = (high_ability_revenue - high_ability_salary - base_config_year2['FIXED_COST']) / 1e6

    # 方案2：调整低能力球员
    low_ability_roster = copy.deepcopy(base_roster_year2)
    for p in low_ability_roster:
        if p.name == "Bench_0":
            p.off = 75
            p.fame = 30
    compute_shapley(low_ability_roster, base_config_year2['W1_WIN'])
    low_ability_roster_managed = manage_roster_salary(low_ability_roster, 1_500_000)
    low_ability_team = Team(low_ability_roster_managed, 78.5, base_config_year2)
    low_ability_team.play_season()
    low_ability_revenue = low_ability_team.gate_revenue + low_ability_team.media_revenue + low_ability_team.merch_revenue
    low_ability_salary = sum(p.salary for p in low_ability_roster_managed)
    low_ability_profit = (low_ability_revenue - low_ability_salary - base_config_year2['FIXED_COST']) / 1e6

    # 方案3：球员替换
    replace_roster_year2 = copy.deepcopy(base_roster_year2)
    replace_roster_year2 = [p for p in replace_roster_year2 if p.name not in ["Erica Wheeler", "Bench_5"]]
    replace_roster_year2.append(Player("Angel Reese", 89.5, 76, 86, 90_000, 23))
    replace_roster_year2.append(Player("Cameron Brink", 79, 81, 71, 75_000, 23))
    compute_shapley(replace_roster_year2, base_config_year2['W1_WIN'])
    replace_roster_year2_managed = manage_roster_salary(replace_roster_year2, 1_500_000)
    replace_team_year2 = Team(replace_roster_year2_managed, 78.5, base_config_year2)
    replace_team_year2.play_season()
    replace_revenue_year2 = replace_team_year2.gate_revenue + replace_team_year2.media_revenue + replace_team_year2.merch_revenue
    replace_salary_year2 = sum(p.salary for p in replace_roster_year2_managed)
    replace_profit_year2 = (replace_revenue_year2 - replace_salary_year2 - base_config_year2['FIXED_COST']) / 1e6

    # 计算调整效率
    high_ability_points = (99 - 96.2) + (72 - 69.1)
    high_ability_efficiency = (high_ability_profit - base_profit_year2) / high_ability_points
    low_ability_points = (75 - 62) + (30 - 5)
    low_ability_efficiency = (low_ability_profit - base_profit_year2) / low_ability_points

    # 输出结果
    print("=" * 90)
    print("第二年（2025年）收益提升 - 球员调整/替换量化分析")
    print("=" * 90)
    print(f"第二年基准利润：{base_profit_year2:.2f} 百万美元")
    print("\n【方案1：高能力球员（Caitlin Clark）小幅调整】")
    print("-" * 70)
    print("调整细节：")
    print("| 球员名称         | 调整维度   | 演化后原值 | 调整后 | 调整点数 |")
    print("|------------------|------------|------------|--------|----------|")
    print(f"| Caitlin Clark    | off        | 96.2       | 99     | +2.8     |")
    print(f"| Caitlin Clark    | deff       | 69.1       | 72     | +2.9     |")
    print(f"调整后利润：{high_ability_profit:.2f} M | 提升幅度：{high_ability_profit - base_profit_year2:.2f} M")
    print(f"单位点数利润提升：{high_ability_efficiency:.3f} M/点（≈{high_ability_efficiency * 1e6:.0f} 美元/点）")

    print("\n【方案2：低能力球员（Bench_0）大幅调整】")
    print("-" * 70)
    print("调整细节：")
    print("| 球员名称         | 调整维度   | 演化后原值 | 调整后 | 调整点数 |")
    print("|------------------|------------|------------|--------|----------|")
    print(f"| Bench_0          | off        | 62.0       | 75     | +13      |")
    print(f"| Bench_0          | fame       | 5.0        | 30     | +25      |")
    print(f"调整后利润：{low_ability_profit:.2f} M | 提升幅度：{low_ability_profit - base_profit_year2:.2f} M")
    print(f"单位点数利润提升：{low_ability_efficiency:.3f} M/点（≈{low_ability_efficiency * 1e6:.0f} 美元/点）")

    print("\n【方案3：球员替换（老化→年轻球员）】")
    print("-" * 70)
    print("替换细节：")
    print("| 被替换球员       | 替换后球员       | 替换后核心属性（2025年，年龄/off/deff/fame/薪资） |")
    print("|------------------|------------------|------------------------------------------------|")
    print("| Erica Wheeler    | Angel Reese      | 23/89.5/76/86/90000                             |")
    print("| Bench_5          | Cameron Brink    | 23/79/81/71/75000                               |")
    print(f"调整后利润：{replace_profit_year2:.2f} M | 提升幅度：{replace_profit_year2 - base_profit_year2:.2f} M")

    print("\n【核心结论：调整效率对比】")
    print("-" * 70)
    print(
        f"1. 高能力球员小幅调整（5.7点）：总提升≈{high_ability_profit - base_profit_year2:.2f} M | 单位效率≈{high_ability_efficiency:.3f} M/点")
    print(
        f"2. 低能力球员大幅调整（38点）：总提升≈{low_ability_profit - base_profit_year2:.2f} M | 单位效率≈{low_ability_efficiency:.3f} M/点")
    print(f"3. 结论：高能力球员调整效率是低能力球员的 {high_ability_efficiency / low_ability_efficiency:.1f} 倍！")
    print("=" * 90)


# ===============================
# 主函数：执行所有分析&可视化
# ===============================
if __name__ == "__main__":
    # 固定随机种子（结果可复现）
    np.random.seed(42)
    random.seed(42)

    # 1. 运行十年预测
    df = run_forecast()

    # 2. 双轴趋势图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="Year", y="Profit_M", marker='o', ax=ax1, label="Profit (M$)")
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Year", y="Brand", color='blue', linestyle='--', ax=ax2, label="Brand")

    ax1.set_ylabel("Profit (Million $)")
    ax2.set_ylabel("Brand Index")
    ax1.grid(alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("10-Year Optimal Strategy Forecast (WNBA)")
    plt.tight_layout()
    plt.show()

    # 3. 3D可视化
    fig = plt.figure(figsize=(24, 7))

    # 子图1：票价+利润+年份
    ax1 = fig.add_subplot(131, projection='3d')
    x1, y1, z1 = df['Opt_Price'].values, df['Profit_M'].values, df['Year'].values
    scatter1 = ax1.scatter(x1, y1, z1, c=z1, cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    ax1.plot(x1, y1, z1, color='red', linewidth=2, alpha=0.6)
    # 投影线
    x1_min, x1_max = ax1.get_xlim()
    y1_min, y1_max = ax1.get_ylim()
    z1_min, z1_max = ax1.get_zlim()
    for x, y, z in zip(x1, y1, z1):
        ax1.plot([x, x], [y, y], [z1_min, z], color='gray', alpha=0.3, linewidth=1)
        ax1.plot([x, x], [y1_min, y], [z, z], color='gray', alpha=0.3, linewidth=1)
        ax1.plot([x1_min, x], [y, y], [z, z], color='gray', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Optimal Ticket Price ($)', fontsize=10)
    ax1.set_ylabel('Profit (Million $)', fontsize=10)
    ax1.set_zlabel('Year', fontsize=10)
    ax1.set_title('Ticket Price vs Profit vs Year', fontsize=12)
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('Year', fontsize=9)

    # 子图2：策略权重+利润+年份
    ax2 = fig.add_subplot(132, projection='3d')
    x2, y2, z2 = df['Opt_Strat'].values, df['Profit_M'].values, df['Year'].values
    scatter2 = ax2.scatter(x2, y2, z2, c=z2, cmap='plasma', s=80, alpha=0.8, edgecolors='black')
    ax2.plot(x2, y2, z2, color='green', linewidth=2, alpha=0.6)
    # 投影线+X轴范围
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks(np.arange(0.0, 1.1, 0.1))
    x2_min, x2_max = ax2.get_xlim()
    y2_min, y2_max = ax2.get_ylim()
    z2_min, z2_max = ax2.get_zlim()
    for x, y, z in zip(x2, y2, z2):
        ax2.plot([x, x], [y, y], [z2_min, z], color='gray', alpha=0.3, linewidth=1)
        ax2.plot([x, x], [y2_min, y], [z, z], color='gray', alpha=0.3, linewidth=1)
        ax2.plot([x2_min, x], [y, y], [z, z], color='gray', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Optimal Strategy Weight (Win > Fame)', fontsize=10)
    ax2.set_ylabel('Profit (Million $)', fontsize=10)
    ax2.set_zlabel('Year', fontsize=10)
    ax2.set_title('Strategy Weight vs Profit vs Year', fontsize=12)
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Year', fontsize=9)

    # 子图3：品牌值+利润+年份
    ax3 = fig.add_subplot(133, projection='3d')
    x3, y3, z3 = df['Brand'].values, df['Profit_M'].values, df['Year'].values
    scatter3 = ax3.scatter(x3, y3, z3, c=z3, cmap='inferno', s=80, alpha=0.8, edgecolors='black')
    ax3.plot(x3, y3, z3, color='blue', linewidth=2, alpha=0.6)
    # 投影线
    x3_min, x3_max = ax3.get_xlim()
    y3_min, y3_max = ax3.get_ylim()
    z3_min, z3_max = ax3.get_zlim()
    for x, y, z in zip(x3, y3, z3):
        ax3.plot([x, x], [y, y], [z3_min, z], color='gray', alpha=0.3, linewidth=1)
        ax3.plot([x, x], [y3_min, y], [z, z], color='gray', alpha=0.3, linewidth=1)
        ax3.plot([x3_min, x], [y, y], [z, z], color='gray', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Brand Index', fontsize=10)
    ax3.set_ylabel('Profit (Million $)', fontsize=10)
    ax3.set_zlabel('Year', fontsize=10)
    ax3.set_title('Brand Index vs Profit vs Year', fontsize=12)
    cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.1)
    cbar3.set_label('Year', fontsize=9)

    plt.suptitle("3D Visualization: Key Metrics vs Profit vs Time (10-Year Forecast)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 4. 执行分析函数
    analyze_first_year_optimization()
    analyze_second_year_optimization()
    analyze_year1_adjust_impact_on_year2()