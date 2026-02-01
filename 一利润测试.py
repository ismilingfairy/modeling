import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# ===============================
# 1. 球员类（含年龄演化）
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

    @property
    def ability(self):
        return (self.off + self.deff) / 2

    def evolve(self):
        if self.age < 26:
            g = np.random.uniform(0, 3)
            self.off = min(99, self.off + g)
            self.deff = min(99, self.deff + g)
            self.fame = min(99, self.fame + 1)
        elif self.age <= 30:
            self.off += np.random.uniform(-1, 1)
            self.deff += np.random.uniform(-1, 1)
        else:
            d = np.random.uniform(1, 4)
            self.off = max(40, self.off - d)
            self.deff = max(40, self.deff - d)
            self.fame = max(10, self.fame - 2)

        self.age += 1

# ===============================
# 2. 团队与赛季模拟
# ===============================

class Team:
    def __init__(self, roster, brand, config):
        self.roster = roster
        self.brand = brand
        self.config = config

    def play_season(self):
        self.wins = 0
        self.gate_revenue = 0.0
        self.media_revenue = 0.0
        self.merch_revenue = 0.0

        for p in self.roster:
            p.health = 100

        for _ in range(self.config['GAMES']):
            self._play_one_game()

        base_media = 10_000_000
        brand_bonus = 30_000_000 * self.brand / (self.brand + 200)
        self.media_revenue = base_media + brand_bonus

        self.merch_revenue = 22_000 * sum(p.fame for p in self.roster)

    def _play_one_game(self):
        healthy = [p for p in self.roster if p.health > 60]
        if not healthy:
            healthy = self.roster[:]

        w = self.config['W1_WIN']
        rotation = sorted(
            healthy,
            key=lambda p: w * p.ability + (1 - w) * p.fame,
            reverse=True
        )[:8]

        strength = np.mean([p.ability for p in rotation])
        opp = np.random.normal(76, 6)
        win_prob = 1 / (1 + np.exp(-(strength - opp) / 6))

        if random.random() < win_prob:
            self.wins += 1

        year_idx = self.config['YEAR_INDEX']
        base_price = 45 * (1.03 ** year_idx)
        ratio = self.config['TICKET_PRICE'] / base_price

        elasticity = 1 / (1 + np.exp(3 * (ratio - 1)))

        has_star = any(p.fame >= 90 for p in rotation)
        brand_factor = self.brand / 350

        attend_rate = min(
            0.99,
            max(0.2, (0.55 + brand_factor + (0.25 if has_star else 0)) * elasticity)
        )

        attendance = self.config['ARENA_CAPACITY'] * attend_rate
        self.gate_revenue += attendance * self.config['TICKET_PRICE']

        for p in self.roster:
            if p in rotation:
                p.health -= np.random.uniform(0.5, 1.5)
            else:
                p.health = min(100, p.health + 3)

# ===============================
# 3. Shapley & 管理
# ===============================

def characteristic_value(players, w1):
    if not players:
        return 0
    strength = np.mean([p.ability for p in players])
    win_val = 100 / (1 + np.exp(-(strength - 70) / 8))
    fame = sum(p.fame for p in players)
    return w1 * win_val + (1 - w1) * fame

def compute_shapley(roster, w1, samples=30):
    shap = {p.name: 0 for p in roster}
    for _ in range(samples):
        perm = roster[:]
        random.shuffle(perm)
        coalition, prev = [], 0
        for p in perm:
            coalition.append(p)
            val = characteristic_value(coalition, w1)
            shap[p.name] += val - prev
            prev = val
    for p in roster:
        p.shapley_val = shap[p.name] / samples

def manage_roster_salary(roster, cap):
    roster = roster[:]  # 防副作用
    roster.sort(
        key=lambda p: p.shapley_val / (p.salary / 1000 + 1e-6) + (5 if p.fame > 90 else 0)
    )
    while sum(p.salary for p in roster) > cap and len(roster) > 10:
        roster.pop(0)
    return roster

# ===============================
# 4. 年度最优策略搜索
# ===============================

def find_optimal_params(year, roster, brand, cap, config):
    compute_shapley(roster, 0.5)

    base_p = 40 * (1.03 ** year)
    prices = np.linspace(base_p * 0.8, base_p * 1.6, 6)
    strategies = np.linspace(0.2, 0.9, 5)

    best = (-1e18, None, None)

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

        avg = np.mean(profits)
        if avg > best[0]:
            best = (avg, p, w)

    return best[1], best[2]

# ===============================
# 5. 十年预测主循环
# ===============================

def run_forecast():
    roster = [
        Player("Caitlin Clark", 95, 68, 99, 76_535, 22),
        Player("Aliyah Boston", 86, 84, 78, 78_469, 22),
        Player("Kelsey Mitchell", 89, 64, 65, 212_000, 28),
        Player("NaLyssa Smith", 82, 68, 55, 80_943, 23),
        Player("Lexie Hull", 74, 82, 48, 73_439, 24),
        Player("Erica Wheeler", 72, 68, 40, 202_000, 33),
    ] + [Player(f"Bench_{i}", 62, 60, 5, 64_154, 24) for i in range(6)]

    config = {
        'GAMES': 40,
        'ARENA_CAPACITY': 18000,
        'FIXED_COST': 25_000_000
    }

    brand = 80
    cap = 1_500_000
    history = []

    for y in range(1, 11):
        price, strat = find_optimal_params(y - 1, roster, brand, cap, config)
        config.update({'TICKET_PRICE': price, 'W1_WIN': strat, 'YEAR_INDEX': y - 1})

        compute_shapley(roster, strat)
        roster = manage_roster_salary(roster, cap)

        team = Team(roster, brand, config)
        team.play_season()

        revenue = team.gate_revenue + team.media_revenue + team.merch_revenue
        salary = sum(p.salary for p in roster)
        profit = revenue - salary - config['FIXED_COST']

        brand += 1.2 * np.log(1 + team.wins) + 0.1 * np.log(1 + sum(p.fame for p in roster))
        brand *= 0.9

        history.append({
            "Year": 2023 + y,
            "Profit_M": profit / 1e6,
            "Brand": brand,
            "Wins": team.wins,
            "Opt_Price": price,
            "Opt_Strat": strat
        })

        cap *= 1.05
        for p in roster:
            p.evolve()
            p.salary *= 1.04

    return pd.DataFrame(history)

# ===============================
# 6. 可视化
# ===============================

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    df = run_forecast()

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
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.regplot(data=df, x="Opt_Price", y="Profit_M", ax=axes[0], color="steelblue")
    axes[0].set_title("Ticket Price vs Profit")

    sns.regplot(data=df, x="Opt_Strat", y="Profit_M", ax=axes[1], color="darkgreen")
    axes[1].set_title("Strategy vs Profit")

    sns.regplot(data=df, x="Brand", y="Profit_M", ax=axes[2], color="purple")
    axes[2].set_title("Brand vs Profit")

    for ax in axes:
        ax.set_ylabel("Profit (Million $)")
        ax.grid(alpha=0.3)

    plt.suptitle("Key Drivers of Financial Performance", fontsize=14)
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["Brand"], df["Opt_Strat"], df["Profit_M"])
    ax.set_xlabel("Brand")
    ax.set_ylabel("Strategy")
    ax.set_zlabel("Profit (M$)")
    plt.show()