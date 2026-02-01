import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 准备验证数据 (Ground Truth vs. Model Data)
# ==========================================

class ModelValidator:
    def __init__(self):
        self.results = {}

    def validate_city_selection(self, model_outputs):
        """
        验证选址决策
        Ground Truth: WNBA 选择了 Golden State (SF)
        """
        # 真实历史结果 (2023年10月官方宣布)
        ground_truth = "Golden State (SF)"

        # 获取模型排名第一的城市
        model_top_pick = model_outputs.sort_values(by='Total_Impact', ascending=False).iloc[0]['City']

        is_correct = (model_top_pick == ground_truth)

        return {
            "Ground_Truth": ground_truth,
            "Model_Prediction": model_top_pick,
            "Is_Correct": is_correct,
            "Rankings": model_outputs[['City', 'Total_Impact']].to_dict('records')
        }

    def validate_protection_list(self, model_list, expert_consensus_list):
        """
        验证保护名单 (使用 Jaccard 相似系数)
        model_list: 模型输出的保护球员名单 (List)
        expert_consensus_list: ESPN/Athletic 专家模拟的必须保护名单 (List)
        """
        set_model = set(model_list)
        set_expert = set(expert_consensus_list)

        intersection = set_model.intersection(set_expert)
        union = set_model.union(set_expert)

        jaccard_index = len(intersection) / len(union)

        # 分类结果
        true_positives = list(intersection)  # 模型预测对了
        false_positives = list(set_model - set_expert)  # 模型保护了专家没保护的
        false_negatives = list(set_expert - set_model)  # 模型漏了专家认为该保护的

        return {
            "Jaccard_Score": jaccard_index,
            "TP": true_positives,
            "FP": false_positives,
            "FN": false_negatives
        }


# ==========================================
# 2. 模拟运行数据
# ==========================================

# 2.1 模拟城市模型输出 (基于之前的计算逻辑)
# 假设这是 H2D2 模型计算出的 Impact Score
city_simulation_data = pd.DataFrame([
    {"City": "Golden State (SF)", "Total_Impact": 3.45},  # 真实赢家
    {"City": "Portland", "Total_Impact": 3.20},  # 强力竞争者
    {"City": "Toronto", "Total_Impact": 1.85},
    {"City": "Philadelphia", "Total_Impact": 1.75},
    {"City": "Nashville", "Total_Impact": -4.20}
])

# 2.2 模拟名单数据 (针对印第安纳狂热队)
# 模型输出 (基于模糊夏普利值 Top 6)
model_protected = [
    "Caitlin Clark", "Aliyah Boston", "Kelsey Mitchell",
    "NaLyssa Smith", "Katie Lou Samuelson", "Erica Wheeler"
]

# 专家共识 (假设数据，基于 ESPN WNBA Mock Expansion Draft)
# 专家可能认为 Temi Fagbenle 比 Erica Wheeler 更值得保护(因年龄/合同)
expert_consensus = [
    "Caitlin Clark", "Aliyah Boston", "Kelsey Mitchell",
    "NaLyssa Smith", "Katie Lou Samuelson", "Temi Fagbenle"
]

# ==========================================
# 3. 执行验证
# ==========================================

validator = ModelValidator()

# 验证 1: 城市选择
city_res = validator.validate_city_selection(city_simulation_data)

# 验证 2: 保护名单
roster_res = validator.validate_protection_list(model_protected, expert_consensus)


# ==========================================
# 4. 可视化验证结果
# ==========================================

def plot_validation_dashboard():
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2)

    # --- 图 1: 城市选择准确性验证 (排名对比) ---
    ax1 = fig.add_subplot(gs[0, 0])

    # 准备数据
    cities = [r['City'] for r in city_res['Rankings']]
    scores = [r['Total_Impact'] for r in city_res['Rankings']]
    colors = ['#28a745' if c == city_res['Ground_Truth'] else '#6c757d' for c in cities]

    bars = ax1.barh(cities, scores, color=colors)

    # 标注
    ax1.set_title(f"Validation 1: Historical Decision Match\nCorrectly Predicted: {city_res['Is_Correct']}",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Model Calculated Impact Score")
    ax1.axvline(0, color='black', linewidth=0.8)

    # 添加 Ground Truth 标签
    for bar, city in zip(bars, cities):
        if city == city_res['Ground_Truth']:
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     "★ ACTUAL WINNER", va='center', color='green', fontweight='bold')

    # --- 图 2: 名单重合度 Venn Diagram 替代方案 (Confusion Bar) ---
    ax2 = fig.add_subplot(gs[0, 1])

    # 计算数值
    tp = len(roster_res['TP'])
    fp = len(roster_res['FP'])
    fn = len(roster_res['FN'])

    # 绘制堆叠条
    ax2.barh([1], [tp], color='#28a745', label=f'Match ({tp})')
    ax2.barh([1], [fp], left=[tp], color='#ffc107', label=f'Model Only ({fp})')
    ax2.barh([1], [fn], left=[tp + fp], color='#dc3545', label=f'Expert Only ({fn})')

    # 装饰
    ax2.set_yticks([])
    ax2.set_title(
        f"Validation 2: Roster Protection Consensus\nJaccard Similarity Index: {roster_res['Jaccard_Score']:.2f}",
        fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Players")
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    # 文本说明差异
    diff_text = f"Model chose: {roster_res['FP'][0]}\nExpert chose: {roster_res['FN'][0]}"
    ax2.text(tp / 2, 1, '\n'.join(roster_res['TP']), ha='center', va='center', color='white', fontweight='bold')
    ax2.text(tp + fp + fn + 0.5, 1, f"Disagreement:\n{diff_text}", va='center', ha='left', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


print(f"--- 验证报告 ---")
print(f"[城市预测] 真实结果: {city_res['Ground_Truth']} | 模型预测: {city_res['Model_Prediction']}")
print(f"[名单一致性] Jaccard系数: {roster_res['Jaccard_Score']:.2f} (越高越好, max=1.0)")
print(f"             差异点: 模型偏好 {roster_res['FP']}, 专家偏好 {roster_res['FN']}")

plot_validation_dashboard()