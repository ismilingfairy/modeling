# -*- coding: utf-8 -*-
"""
WNBA Team 10-Year Revenue Forecast (Final Optimized Version)
Core Modifications:
1. Player peak period adjusted to 27-32 years old; decline starts after 32.
2. Added comprehensive player injury history statistics.
3. Stratified injury probability (Clark 30% > Other Core 25% > Bench 5%) and limited total injuries to prevent team paralysis.
4. Intensified impact of Clark's injuries (Ability/Brand/Profit penalties doubled), minimized impact of bench player injuries.
5. Controlled parameter ranges to ensure results remain consistent with the original version.
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
# 1. Player Class: Core adjustment to peak period + Injury logic
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

        # Core injury attributes
        self.is_core = name in ["Caitlin Clark", "Aliyah Boston", "Kelsey Mitchell"]
        self.is_clark = name == "Caitlin Clark"  # Flag for Clark specifically
        self.is_injured = False
        self.injury_level = 0
        self.injury_impact = 0.0

        # New: Record personal injury history
        self.injury_history = []  # Format: [(Year, Level, Impact), ...]

    @property
    def ability(self):
        """Intensify Clark's injury impact on ability, reduce bench impact."""
        base_ability = (self.off + self.deff) / 2
        if self.is_injured:
            # Impact coefficient layering: Clark (2.5x) > Core (2x) > Bench (0.8x)
            if self.is_clark:
                impact = self.injury_impact * 2.5
            elif self.is_core:
                impact = self.injury_impact * 2
            else:
                impact = self.injury_impact * 0.8  # Reduced impact for bench
            base_ability *= (1 - impact)
        return base_ability

    def evolve(self):
        """Adjust ability evolution: Peak 27-32, decline after 32."""
        if self.age < 27:  # Rising period (Adjusted to before 27)
            g = np.random.uniform(-1, 3)  # Small fluctuation
            self.off = min(99, max(40, self.off + g))
            self.deff = min(99, max(40, self.deff + g))
            self.fame = min(99, max(10, self.fame + np.random.uniform(-0.5, 1)))
        elif self.age <= 32:  # Peak period (27-32, extended to 32)
            self.off += np.random.uniform(-1, 1)  # Stability in peak
            self.deff += np.random.uniform(-1, 1)
        else:  # Decline starts after 32 (Core adjustment)
            d = np.random.uniform(1, 4)  # Slower decline to ensure stability
            self.off = max(40, self.off - d)
            self.deff = max(40, self.deff - d)
            self.fame = max(10, self.fame - np.random.uniform(1, 2))  # Fame decline slows down
        self.age += 1

    def generate_season_injury(self, year):
        """Stratified injury probability + Limit severe injuries."""
        self.is_injured = False
        self.injury_level = 0
        self.injury_impact = 0.0

        # Stratified probabilities: Clark(30%) > Core(25%) > Bench(5%)
        if self.is_clark:
            injury_prob = 0.30  # Highest for Clark
        elif self.is_core:
            injury_prob = 0.25
        else:
            injury_prob = 0.05  # Low for bench

        if random.random() < injury_prob:
            self.is_injured = True
            # Injury level weights: Clark prone to severe, Bench almost none
            if self.is_clark:
                self.injury_level = random.choices([1, 2, 3], weights=[0.1, 0.2, 0.7])[0]
            elif self.is_core:
                self.injury_level = random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
            else:
                self.injury_level = random.choices([1, 2, 3], weights=[0.8, 0.2, 0.0])[0]  # No severe for bench

            if self.injury_level == 1:
                self.injury_impact = 0.1
            elif self.injury_level == 2:
                self.injury_impact = 0.3
            elif self.injury_level == 3:
                self.injury_impact = 1.0

        # Record history
        self.injury_history.append((year, self.injury_level, self.injury_impact))


# ===============================
# 2. Team Class: Control injury count + Intensify Clark impact
# ===============================
class Team:
    def __init__(self, roster, brand, config):
        self.roster = roster
        self.brand = brand
        self.config = config

    def play_season(self, year):
        self.wins = 0
        self.gate_revenue = 0.0
        self.media_revenue = 0.0
        self.merch_revenue = 0.0

        # Generate injuries (Initial)
        for p in self.roster:
            p.generate_season_injury(year)
            if p.injury_level == 3:
                p.health = 0
            else:
                p.health = 100

        # Core mod: Limit total injuries to keep team operational
        core_injured = [p for p in self.roster if p.is_core and p.is_injured]
        bench_injured = [p for p in self.roster if not p.is_core and p.is_injured]

        # Max 2 core injuries, Max 5 total injuries
        if len(core_injured) > 2:
            # Randomly recover excess core players
            for p in random.sample(core_injured[2:], len(core_injured) - 2):
                p.is_injured = False
                p.injury_level = 0
                p.injury_impact = 0.0
                p.injury_history[-1] = (year, 0, 0.0)  # Update history
        if len(core_injured) + len(bench_injured) > 5:
            # Randomly recover excess bench players
            excess = len(core_injured) + len(bench_injured) - 5
            for p in random.sample(bench_injured, min(excess, len(bench_injured))):
                p.is_injured = False
                p.injury_level = 0
                p.injury_impact = 0.0
                p.injury_history[-1] = (year, 0, 0.0)  # Update history

        # Recalculate injury stats (After limits)
        self.core_injured_severe = sum(1 for p in self.roster if p.is_core and p.injury_level >= 2)
        self.core_injured_any = sum(1 for p in self.roster if p.is_core and p.is_injured)
        self.clark_injured = 1 if any(p.is_clark and p.is_injured for p in self.roster) else 0

        # Simulate games
        for _ in range(self.config['GAMES']):
            self._play_one_game()

        # Media Revenue: Extra penalty for Clark injury
        base_media = 10_000_000
        brand_bonus = 30_000_000 * self.brand / (self.brand + 200)
        media_penalty = 0.2 * self.core_injured_severe
        if self.clark_injured:
            media_penalty += 0.15  # Extra 15% penalty
        self.media_revenue = (base_media + brand_bonus) * (1 - media_penalty)

        # Merch Revenue: Extra penalty for Clark, reduced for bench
        total_fame = sum(p.fame for p in self.roster if not (p.is_injured and p.injury_level == 3))
        core_merch_penalty = 0.25 * self.core_injured_any
        bench_merch_penalty = 0.05 * len([p for p in self.roster if not p.is_core and p.is_injured])
        if self.clark_injured:
            core_merch_penalty += 0.2  # Extra 20% penalty
        self.merch_revenue = 22_000 * total_fame * (1 - core_merch_penalty - bench_merch_penalty)

    def _play_one_game(self):
        """Maintain logic, fine-tune parameters for stability."""
        # Filter available players
        healthy = [p for p in self.roster if p.health > 60 and p.injury_level != 3] or self.roster[:]

        # Determine rotation
        w = self.config['W1_WIN']
        rotation = sorted(healthy, key=lambda p: w * p.ability + (1 - w) * p.fame, reverse=True)[:8]

        # Team Strength
        strength = np.mean([p.ability for p in rotation])
        opp_strength = np.random.normal(77, 7)  # Fine-tune mean/variance
        win_prob = 1 / (1 + np.exp(-(strength - opp_strength) / 6))
        if random.random() < win_prob:
            self.wins += 1

        # Gate Revenue
        year_idx = self.config['YEAR_INDEX']
        base_price = 45 * (1.03 ** year_idx)
        ratio = self.config['TICKET_PRICE'] / base_price
        elasticity = 1 / (1 + np.exp(3 * (ratio - 1)))

        has_healthy_core = any(p.is_core and not p.is_injured for p in rotation)
        core_injured_penalty = 0.25 * self.core_injured_severe
        attend_rate = min(0.99, max(0.2,
                                    (0.52 + self.brand / 420 + (
                                        0.18 if has_healthy_core else 0)) * elasticity - core_injured_penalty))

        # Fluctuate attendance slightly
        attend_rate *= np.random.uniform(0.95, 1.05)
        attendance = self.config['ARENA_CAPACITY'] * attend_rate

        # Cap ticket price increase
        max_price = base_price * 1.28
        ticket_price = min(self.config['TICKET_PRICE'], max_price)
        self.gate_revenue += attendance * ticket_price

        # Update Health
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
# 3. Utility Functions
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


# ===============================
# 4. Optimal Strategy Search
# ===============================
def find_optimal_params(year, roster, brand, cap, config):
    compute_shapley(roster, 0.5)

    # Limit ticket price candidates
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

            tm = Team(sim_roster, brand, cfg)
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
# 5. 10-Year Forecast Loop
# ===============================
def run_forecast():
    # Initial Roster
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
        'FIXED_COST': 25_000_000,
    }

    brand = 80
    cap = 1_500_000
    history = []
    injury_records = []
    prev_core_severe_injuries = 0

    for y in range(1, 11):
        current_year = 2023 + y
        price, strat = find_optimal_params(y - 1, roster, brand, cap, config)
        config.update({'TICKET_PRICE': price, 'W1_WIN': strat, 'YEAR_INDEX': y - 1})

        compute_shapley(roster, strat)
        roster = manage_roster_salary(roster, cap)

        team = Team(roster, brand, config)
        team.play_season(current_year)
        revenue = team.gate_revenue + team.media_revenue + team.merch_revenue
        salary = sum(p.salary for p in roster)
        profit = revenue - salary - config['FIXED_COST']
        profit *= np.random.uniform(0.9, 1.1)

        # Brand Value Logic
        brand += 1.2 * np.log(1 + team.wins) + 0.1 * np.log(1 + sum(p.fame for p in roster))
        injury_improvement = max(0, prev_core_severe_injuries - team.core_injured_severe)
        injury_recovery_bonus = injury_improvement * 0.028 * brand
        brand += injury_recovery_bonus

        # Decay Rate
        win_bonus = 0.018 * team.wins
        injury_penalty = 0.048 * team.core_injured_severe
        if team.clark_injured:
            injury_penalty += 0.02
        decay_rate = 0.9 - injury_penalty + win_bonus
        decay_rate = np.clip(decay_rate, 0.85, 0.95)
        brand *= decay_rate
        brand *= np.random.uniform(0.98, 1.02)

        prev_core_severe_injuries = team.core_injured_severe

        # Record Injuries
        yearly_injuries = []
        for p in roster:
            if p.is_injured:
                injury_desc = {
                    "name": p.name,
                    "is_core": p.is_core,
                    "is_clark": p.is_clark,
                    "level": p.injury_level,
                    "level_desc": {1: "Minor", 2: "Moderate", 3: "Severe"}[p.injury_level],
                    "impact": p.injury_impact
                }
                yearly_injuries.append(injury_desc)

        injury_records.append(yearly_injuries)

        # Record Annual Data
        history.append({
            "Year": current_year,
            "Profit_M": profit / 1e6,
            "Brand": brand,
            "Wins": team.wins,
            "Opt_Price": price,
            "Opt_Strat": strat,
            "Injury_Count": len(yearly_injuries),
            "Core_Injury_Count": sum(1 for inj in yearly_injuries if inj["is_core"]),
            "Core_Severe_Injury_Count": team.core_injured_severe,
            "Clark_Injured": team.clark_injured,
            "Brand_Recovery_Bonus": injury_recovery_bonus
        })

        # Evolve Players
        for p in roster:
            p.evolve()

    # Summarize Player Injury History
    injury_summary = []
    for p in roster:
        total_injuries = sum(1 for hist in p.injury_history if hist[1] > 0)
        severe_injuries = sum(1 for hist in p.injury_history if hist[1] >= 2)
        avg_impact = np.mean([hist[2] for hist in p.injury_history if hist[1] > 0]) if total_injuries > 0 else 0
        injury_summary.append({
            "Player_Name": p.name,
            "Is_Core": p.is_core,
            "Is_Clark": p.is_clark,
            "Total_Injuries": total_injuries,
            "Severe_Injuries": severe_injuries,
            "Avg_Injury_Impact": avg_impact,
            "Career_Age_Range": f"{22}~{22 + 9}" if "Bench" not in p.name else f"{24}~{33}",
            "Peak_Period": "27~32 yrs"
        })

    return pd.DataFrame(history), injury_records, pd.DataFrame(injury_summary)


# ===============================
# 6. Visualization
# ===============================
def plot_injury_profit_analysis(df, injury_records):
    # Removed Chinese font settings to ensure compatibility
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    # Subplot 1: Profit + Brand Trend
    fig, ax1 = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=df, x="Year", y="Profit_M", marker='o', ax=ax1,
                 color='#e74c3c', linewidth=2.5, markersize=8, label="Profit (Million $)")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Profit (Million $)", fontsize=12, color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Year", y="Brand", color='#3498db', linestyle='--', ax=ax2,
                 marker='s', markersize=6, label="Brand Value")
    ax2.set_ylabel("Brand Value", fontsize=12, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    # Annotate Injuries (Clark Highlight)
    for idx, row in df.iterrows():
        year = row["Year"]
        profit = row["Profit_M"]
        injuries = injury_records[idx]

        if len(injuries) == 0:
            injury_text = "No Injuries"
        else:
            clark_injuries = [inj for inj in injuries if inj["is_clark"]]
            core_injuries = [inj for inj in injuries if inj["is_core"] and not inj["is_clark"]]
            bench_injuries = [inj for inj in injuries if not inj["is_core"]]

            injury_text = ""
            if clark_injuries:
                injury_text += "[Clark Injury]\n" + "\n".join([
                    f"{inj['name']}({inj['level_desc']})" for inj in clark_injuries
                ])
            if core_injuries:
                injury_text += "\nOther Core Inj:\n" + "\n".join([
                    f"{inj['name']}({inj['level_desc']})" for inj in core_injuries[:1]
                ])
            if bench_injuries:
                injury_text += f"\nBench Inj: {len(bench_injuries)}"

        # Annotate Brand Recovery Bonus
        recovery_bonus = row.get("Brand_Recovery_Bonus", 0)
        if recovery_bonus > 0:
            injury_text += f"\nBrand Recovery: +{recovery_bonus:.1f}"

        ax1.annotate(
            injury_text,
            xy=(year, profit),
            xytext=(10, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
            fontsize=8
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.set_title("WNBA Team 10-Year Profit/Brand Trend (Clark Injury Highlighted)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("profit_injury_trend.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Subplot 2: Core Injury vs Profit
    fig, ax = plt.subplots(figsize=(12, 7))
    # Differentiate Clark Injury with Colors
    colors = ['red' if df.loc[i, 'Clark_Injured'] == 1 else 'blue' for i in range(len(df))]
    scatter = ax.scatter(
        df["Core_Injury_Count"], df["Profit_M"],
        c=colors, s=150, alpha=0.8, edgecolors='black'
    )

    z = np.polyfit(df["Core_Injury_Count"], df["Profit_M"], 1)
    p = np.poly1d(z)
    ax.plot(df["Core_Injury_Count"], p(df["Core_Injury_Count"]),
            "r--", linewidth=2, label=f"Trend Line: y={z[0]:.2f}x + {z[1]:.2f}")

    for idx, row in df.iterrows():
        ax.annotate(
            str(row["Year"]) + ("(Clark Inj)" if row["Clark_Injured"] == 1 else ""),
            xy=(row["Core_Injury_Count"], row["Profit_M"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax.set_xlabel("Core Player Injuries (Count/Year)", fontsize=12)
    ax.set_ylabel("Profit (Million $)", fontsize=12)
    ax.set_title("Correlation: Core Player (incl. Clark) Injuries vs Annual Profit", fontsize=16, pad=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # Legend for Clark Status
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Clark Injured'),
        Patch(facecolor='blue', label='Clark Healthy')
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper right')

    plt.tight_layout()
    plt.savefig("core_injury_profit_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_metrics(df):
    fig = plt.figure(figsize=(24, 7))

    # Plot 1: Price vs Profit vs Year
    ax1 = fig.add_subplot(131, projection='3d')
    x1, y1, z1 = df['Opt_Price'].values, df['Profit_M'].values, df['Year'].values
    scatter1 = ax1.scatter(x1, y1, z1, c=z1, cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    ax1.plot(x1, y1, z1, color='red', linewidth=2, alpha=0.6)
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

    # Plot 2: Strategy vs Profit vs Year
    ax2 = fig.add_subplot(132, projection='3d')
    x2, y2, z2 = df['Opt_Strat'].values, df['Profit_M'].values, df['Year'].values
    scatter2 = ax2.scatter(x2, y2, z2, c=z2, cmap='plasma', s=80, alpha=0.8, edgecolors='black')
    ax2.plot(x2, y2, z2, color='green', linewidth=2, alpha=0.6)
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

    # Plot 3: Brand vs Profit vs Year
    ax3 = fig.add_subplot(133, projection='3d')
    x3, y3, z3 = df['Brand'].values, df['Profit_M'].values, df['Year'].values
    scatter3 = ax3.scatter(x3, y3, z3, c=z3, cmap='inferno', s=80, alpha=0.8, edgecolors='black')
    ax3.plot(x3, y3, z3, color='blue', linewidth=2, alpha=0.6)
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


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    # Fix seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run Forecast
    df, injury_records, injury_summary = run_forecast()

    # Visualize
    plot_injury_profit_analysis(df, injury_records)
    plot_3d_metrics(df)

    # Print Core Statistics
    print("=" * 80)
    print("WNBA Team 10-Year Injury-Profit Core Statistics (Final Optimized Version)")
    print("=" * 80)
    print(f"Avg Annual Profit: {df['Profit_M'].mean():.2f} Million $")
    print(f"Avg Core Injury Count: {df['Core_Injury_Count'].mean():.2f} per year")
    print(f"Clark Total Injuries: {df['Clark_Injured'].sum()} times/10 years")
    print(f"Avg Bench Injury Count: {(df['Injury_Count'] - df['Core_Injury_Count']).mean():.2f} per year")
    core_injury_profit_corr = df['Core_Injury_Count'].corr(df['Profit_M'])
    print(f"Correlation (Core Injuries vs Profit): {core_injury_profit_corr:.3f} (Negative = Inverse Relation)")
    profit_std = df['Profit_M'].std()
    print(f"Profit Volatility: {profit_std / df['Profit_M'].mean() * 100:.1f}%")
    brand_change = df['Brand'].iloc[-1] - df['Brand'].iloc[0]
    print(f"10-Year Brand Change: {brand_change:.1f} (Start: 80, End: {df['Brand'].iloc[-1]:.1f})")
    print(f"Total Brand Recovery Bonus: {df['Brand_Recovery_Bonus'].sum():.1f}")
    print("=" * 80)

    # Print Detailed Injury History
    print("\n" + "=" * 115)
    print("Player 10-Year Injury History Statistics")
    print("=" * 115)
    # Formatted Table
    headers = ["Player Name", "Is Core", "Is Clark", "Total Inj", "Severe Inj", "Avg Impact", "Career Age", "Peak Period"]
    header_line = "| " + " | ".join([f"{h:<15}" for h in headers]) + " |"
    print(header_line)
    print("|" + "-" * (len(header_line) - 2) + "|")
    for _, row in injury_summary.iterrows():
        row_line = "| " + " | ".join([
            f"{row['Player_Name']:<15}",
            f"{str(row['Is_Core']):<15}",
            f"{str(row['Is_Clark']):<15}",
            f"{row['Total_Injuries']:<15}",
            f"{row['Severe_Injuries']:<15}",
            f"{row['Avg_Injury_Impact']:.3f}".ljust(15),
            f"{row['Career_Age_Range']:<15}",
            f"{row['Peak_Period']:<15}"
        ]) + " |"
        print(row_line)
    print("=" * 115)
