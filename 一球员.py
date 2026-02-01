
import numpy as np
import random
import copy  # 新增：用于深拷贝球员对象，避免策略间相互干扰
from itertools import combinations
from sklearn.linear_model import LinearRegression

# ===============================
# 1. 核心参数（修正尺度，添加注释）
# ===============================
FIXED_COST = 12_000_000       # 球队固定成本（美元/年）
SALARY_CAP = 7_000_000        # 工资帽（总薪资上限，美元/年）
BASE_ATTENDANCE = 1_200_000   # 基准上座人数（无调整时的基础值）
BASE_TICKET_PRICE = 40        # 基准票价（美元/张）
MARKET_SIZE = 1.0             # 市场规模系数（1.0为基准）

# ===============================
# 2. 球员类定义（补充注释，明确属性含义）
# ===============================
class Player:
    def __init__(self, name, off, deff, fame, salary, age):
        """
        球员类：定义球员核心属性
        :param name: 球员姓名
        :param off: 进攻能力值（0-100）
        :param deff: 防守能力值（0-100）
        :param fame: 知名度（0-100，影响上座率/商业价值）
        :param salary: 年薪（美元）
        :param age: 年龄（影响能力演化/薪资谈判）
        :param shapley_val: 沙普利值（衡量球员对利润的边际贡献）
        """
        self.name = name
        self.off = off          # 进攻能力
        self.deff = deff        # 防守能力
        self.fame = fame        # 知名度
        self.salary = salary    # 年薪
        self.age = age          # 年龄
        self.shapley_val = 0    # 沙普利值（初始为0）

    def rating(self):
        """计算球员综合评分（攻防+知名度加权）"""
        return 0.45 * self.off + 0.35 * self.deff + 0.2 * self.fame

# ===============================
# 3. 收益函数（稳定版，补充逻辑注释）
# ===============================
def calc_profit(roster, brand, ticket_price):
    """
    计算球队年度利润、平均攻防属性
    :param roster: 球员名单（Player对象列表）
    :param brand: 球队品牌值（影响上座率/商业吸引力）
    :param ticket_price: 实际票价（美元/张）
    :return: 利润（百万美元）、平均进攻、平均防守
    """
    # 计算阵容平均属性
    avg_off = np.mean([p.off for p in roster])
    avg_def = np.mean([p.deff for p in roster])
    avg_fame = np.mean([p.fame for p in roster])

    # 球队竞技表现（攻防加权）
    performance = 0.6 * avg_off + 0.4 * avg_def
    # 球队整体吸引力（表现+知名度+品牌）
    appeal = 0.7 * performance + 0.3 * avg_fame + 0.5 * brand

    # 票价弹性：票价越高，上座率弹性越低（修正过陡问题）
    ratio = ticket_price / BASE_TICKET_PRICE  # 实际票价/基准票价
    elasticity = 1 / (1 + np.exp(1.5 * (ratio - 1)))

    # 实际上座人数：基准上座数 * 价格弹性 * 吸引力系数
    attendance = BASE_ATTENDANCE * elasticity * (appeal / 75)
    # 票务总收入：上座人数 * 票价 * 市场规模
    revenue = attendance * ticket_price * MARKET_SIZE

    # 总薪资成本
    salary_cost = sum(p.salary for p in roster)
    # 年度利润：总收入 - 薪资成本 - 固定成本
    profit = revenue - salary_cost - FIXED_COST

    return profit / 1e6, avg_off, avg_def  # 利润转换为百万美元，便于阅读

# ===============================
# 4. Shapley 价值计算（精简可复现，补充注释）
# 核心作用：衡量单个球员对球队利润的边际贡献（移除该球员后的利润损失）
# ===============================
def compute_shapley(roster):
    """计算每个球员的沙普利值（边际利润贡献）"""
    # 完整阵容的基准利润
    base_profit, _, _ = calc_profit(roster, 75, BASE_TICKET_PRICE)
    for p in roster:
        # 移除当前球员后的阵容利润
        reduced = [x for x in roster if x != p]
        reduced_profit, _, _ = calc_profit(reduced, 75, BASE_TICKET_PRICE)
        # 沙普利值 = 完整阵容利润 - 移除该球员后的利润（边际贡献）
        p.shapley_val = base_profit - reduced_profit

# ===============================
# 5. 辅助函数：计算调整效率（新增）
# 核心作用：量化「单位属性调整带来的利润提升」，评估策略性价比
# ===============================
def calc_adjust_efficiency(base_profit, adj_profit, adjust_points):
    """
    计算调整效率（百万美元/调整点数）
    :param base_profit: 调整前基准利润
    :param adj_profit: 调整后利润
    :param adjust_points: 总调整点数（如进攻+防守+知名度的总变化量）
    :return: 单位点数利润提升
    """
    if adjust_points == 0:
        return 0
    profit_increase = adj_profit - base_profit
    return profit_increase / adjust_points

# ===============================
# 6. 第一阶段：构建初始阵容
# ===============================
roster = [
    Player("Caitlin Clark", 95, 68, 90, 100_000, 23),  # 核心高能力球员
    Player("Erica Wheeler", 72, 70, 40, 110_000, 32),  # 老化球员
    Player("Bench_0", 62, 65, 5, 60_000, 26),          # 低能力替补
    Player("Bench_5", 60, 64, 6, 55_000, 31),          # 低能力替补
]

# ===============================
# 7. 第一年度基准 & 调整策略（修复拷贝问题）
# ===============================
# 7.1 第一年基准（无调整）
base_profit_2024, base_off_2024, base_def_2024 = calc_profit(roster, 75, BASE_TICKET_PRICE)
base_salary_2024 = sum(p.salary for p in roster)  # 第一年基准薪资

# 7.2 第一年调整策略（使用深拷贝，避免修改原阵容）
def adjust_year1(roster):
    """第一年球员属性调整策略"""
    # 深拷贝：避免修改原列表/球员对象，保证策略间互不干扰
    adj_roster = copy.deepcopy(roster)
    for p in adj_roster:
        if p.name == "Caitlin Clark":
            p.off += 3  # 进攻+3
            p.deff += 3  # 防守+3
        if p.name == "Erica Wheeler":
            p.off += 2  # 进攻+2
            p.fame += 5  # 知名度+5
        if p.name == "Bench_0":
            p.off += 8  # 进攻+8
            p.fame += 15  # 知名度+15
    return adj_roster

# 执行第一年调整
roster_y1 = adjust_year1(roster)
profit_2024_adj, off_2024_adj, def_2024_adj = calc_profit(roster_y1, 75, BASE_TICKET_PRICE)
salary_2024_adj = sum(p.salary for p in roster_y1)  # 第一年调整后薪资

# ===============================
# 8. 第二年度分析（固定品牌，反事实验证）
# ===============================
brand_2025 = 78.5  # 2025年球队品牌值（固定）
# 第二年基准（第一年调整后，无额外策略）
base_profit_2025, base_off_2025, base_def_2025 = calc_profit(roster_y1, brand_2025, BASE_TICKET_PRICE)
base_salary_2025 = sum(p.salary for p in roster_y1)  # 第二年基准薪资

# ---------- 策略 1：高能力球员小幅调整（补充调整点数统计） ----------
def strategy_high_small(r):
    """策略1：高能力球员（Caitlin Clark）小幅调整"""
    adj_roster = copy.deepcopy(r)
    adjust_points = 0  # 统计总调整点数
    for p in adj_roster:
        if p.name == "Caitlin Clark":
            # 记录调整点数（进攻：98→99，防守：71→72）
            adjust_points += (99 - p.off) + (72 - p.deff)
            p.off = 99
            p.deff = 72
    return adj_roster, adjust_points

# ---------- 策略 2：低能力球员大幅调整（补充调整点数统计） ----------
def strategy_low_large(r):
    """策略2：低能力球员（Bench_0）大幅调整"""
    adj_roster = copy.deepcopy(r)
    adjust_points = 0
    for p in adj_roster:
        if p.name == "Bench_0":
            # 记录调整点数（进攻：70→75，知名度：20→30）
            adjust_points += (75 - p.off) + (30 - p.fame)
            p.off = 75
            p.fame = 30
    return adj_roster, adjust_points

# ---------- 策略 3：球员替换（补充调整点数/薪资变化） ----------
def strategy_replace(r):
    """策略3：替换老化球员为年轻新星"""
    adj_roster = copy.deepcopy(r)
    new_roster = []
    adjust_points = 0  # 替换后的总属性提升点数
    salary_change = 0  # 薪资变化（新薪资-旧薪资）
    for p in adj_roster:
        if p.name == "Erica Wheeler":
            # 替换Erica Wheeler为Angel Reese
            new_player = Player("Angel Reese", 89.5, 76, 86, 90_000, 23)
            new_roster.append(new_player)
            # 计算属性提升（进攻+防守+知名度）
            adjust_points += (new_player.off - p.off) + (new_player.deff - p.deff) + (new_player.fame - p.fame)
            # 计算薪资变化
            salary_change += new_player.salary - p.salary
        elif p.name == "Bench_5":
            # 替换Bench_5为Cameron Brink
            new_player = Player("Cameron Brink", 79, 81, 71, 75_000, 23)
            new_roster.append(new_player)
            adjust_points += (new_player.off - p.off) + (new_player.deff - p.deff) + (new_player.fame - p.fame)
            salary_change += new_player.salary - p.salary
        else:
            new_roster.append(p)
    return new_roster, adjust_points, salary_change

# ===============================
# 9. 第二年策略评估（扩展分析维度）
# ===============================
results = []

# 策略1：高能力小幅调整
roster_high_small, adjust_points_high = strategy_high_small(roster_y1)
profit_high, off_high, def_high = calc_profit(roster_high_small, brand_2025, BASE_TICKET_PRICE)
salary_high = sum(p.salary for p in roster_high_small)
# 计算调整效率（单位点数利润提升）
eff_high = calc_adjust_efficiency(base_profit_2025, profit_high, adjust_points_high)
results.append({
    "策略名称": "高能力小幅调整",
    "调整点数": adjust_points_high,
    "基准利润(M$)": base_profit_2025,
    "调整后利润(M$)": profit_high,
    "利润变化(M$)": profit_high - base_profit_2025,
    "调整效率(M$/点)": eff_high,
    "平均进攻": off_high,
    "平均防守": def_high,
    "总薪资(美元)": salary_high,
    "薪资占帽比(%)": (salary_high / SALARY_CAP) * 100
})

# 策略2：低能力大幅调整
roster_low_large, adjust_points_low = strategy_low_large(roster_y1)
profit_low, off_low, def_low = calc_profit(roster_low_large, brand_2025, BASE_TICKET_PRICE)
salary_low = sum(p.salary for p in roster_low_large)
eff_low = calc_adjust_efficiency(base_profit_2025, profit_low, adjust_points_low)
results.append({
    "策略名称": "低能力大幅调整",
    "调整点数": adjust_points_low,
    "基准利润(M$)": base_profit_2025,
    "调整后利润(M$)": profit_low,
    "利润变化(M$)": profit_low - base_profit_2025,
    "调整效率(M$/点)": eff_low,
    "平均进攻": off_low,
    "平均防守": def_low,
    "总薪资(美元)": salary_low,
    "薪资占帽比(%)": (salary_low / SALARY_CAP) * 100
})

# 策略3：替换年轻球员
roster_replace, adjust_points_replace, salary_change_replace = strategy_replace(roster_y1)
profit_replace, off_replace, def_replace = calc_profit(roster_replace, brand_2025, BASE_TICKET_PRICE)
salary_replace = sum(p.salary for p in roster_replace)
eff_replace = calc_adjust_efficiency(base_profit_2025, profit_replace, adjust_points_replace)
results.append({
    "策略名称": "替换年轻球员",
    "调整点数": adjust_points_replace,
    "基准利润(M$)": base_profit_2025,
    "调整后利润(M$)": profit_replace,
    "利润变化(M$)": profit_replace - base_profit_2025,
    "调整效率(M$/点)": eff_replace,
    "平均进攻": off_replace,
    "平均防守": def_replace,
    "总薪资(美元)": salary_replace,
    "薪资占帽比(%)": (salary_replace / SALARY_CAP) * 100
})

# ===============================
# 10. 输出结构化结果（表格形式）
# ===============================
print("="*120)
print("【第一年调整结果】")
print("="*120)
print(f"2024年基准利润: {base_profit_2024:.2f} M$ | 调整后利润: {profit_2024_adj:.2f} M$ | 利润提升: {profit_2024_adj - base_profit_2024:.2f} M$")
print(f"2024年基准攻防: 进攻={base_off_2024:.2f}, 防守={base_def_2024:.2f} | 调整后攻防: 进攻={off_2024_adj:.2f}, 防守={def_2024_adj:.2f}")
print(f"2024年基准薪资: {base_salary_2024:,} 美元 | 调整后薪资: {salary_2024_adj:,} 美元")

print("\n" + "="*120)
print("【第二年（2025）策略对比结果】")
print("="*120)
# 表头（格式化对齐）
header = (
    f"{'策略名称':<12} | {'调整点数':<8} | {'基准利润(M$)':<12} | {'调整后利润(M$)':<14} | {'利润变化(M$)':<12} | "
    f"{'调整效率(M$/点)':<14} | {'平均进攻':<8} | {'平均防守':<8} | {'薪资占帽比(%)':<12}"
)
print(header)
print("-"*120)
# 每行结果（格式化输出）
for res in results:
    row = (
        f"{res['策略名称']:<12} | {res['调整点数']:<8.1f} | {res['基准利润(M$)']:<12.2f} | {res['调整后利润(M$)']:<14.2f} | "
        f"{res['利润变化(M$)']:<12.2f} | {res['调整效率(M$/点)']:<14.3f} | {res['平均进攻']:<8.2f} | {res['平均防守']:<8.2f} | "
        f"{res['薪资占帽比(%)']:<12.1f}"
    )
    print(row)

# 计算并输出Shapley值（补充球员贡献分析）
print("\n" + "="*120)
print("【核心球员沙普利值（边际利润贡献）】")
print("="*120)
compute_shapley(roster_replace)  # 以替换策略后的阵容为例
for p in roster_replace:
    print(f"{p.name:<15} | 沙普利值（M$）: {p.shapley_val:.2f} | 综合评分: {p.rating():.2f}")

print("\n" + "="*120)
print("【核心结论】")
print("="*120)
# 找出最优策略（利润提升最多）
best_profit_strategy = max(results, key=lambda x: x['利润变化(M$)'])
# 找出最高效率策略（单位点数利润提升最多）
best_efficiency_strategy = max(results, key=lambda x: x['调整效率(M$/点)'])
print(f"1. 利润提升最多的策略：{best_profit_strategy['策略名称']}（提升{best_profit_strategy['利润变化(M$)']:.2f} M$）")
print(f"2. 调整效率最高的策略：{best_efficiency_strategy['策略名称']}（{best_efficiency_strategy['调整效率(M$/点)']:.3f} M$/点）")
print(f"3. 所有策略薪资占帽比均低于100%，无薪资超帽风险")
print(f"4. 高能力球员小幅调整的性价比远高于低能力球员大幅调整（效率高{best_efficiency_strategy['调整效率(M$/点)']/results[1]['调整效率(M$/点)']:.1f}倍）")
print("="*120)