"""
业务逻辑：
先筛掉低质量广告（约束过滤）。
再算哪些广告最有价值（eCPM & Pareto）。
用加权排序平衡点击率和价格。
给广告机会时兼顾“最优”和“探索新广告”。
最后在一天内动态分配预算，避免不均衡。

假设：CTR × Bid × 预算分配
CTR 是静态的，不会波动。
Bid 是固定的。
广告的价值完全由点击率和出价决定。
我们只关心 CTR 和 Bid。
探索率是固定的。
流量和竞争在各个时段均匀。
成本和 Bid 成比例。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

np.random.seed(43)


# -----------------------------
# 1) 生成模拟广告候选数据
# -----------------------------
def generate_ads(n_ads=500):
    """
    生成 n_ads 条模拟广告的真值（真实CTR/CVR/bid），以及模型的预测（有噪声）。
    返回 DataFrame 包含：
      - ad_id, bid, ctr_true, cvr_true, ctr_pred, cvr_pred
    说明：
      - ctr_true ~ Beta(2, 50) -> 低点击率分布（常见广告）
      - cvr_true ~ Beta(1, 200) -> 更低的转化率
      - bid ~ Uniform(0.1, 5.0) （以货币单位表示每次曝光成本/估值的简化）
      - ctr_pred, cvr_pred: 在真实值上加噪并剪裁到 [0,1]
    """
    ctr_true = np.random.beta(2, 50, size=n_ads)
    cvr_true = np.random.beta(1, 200, size=n_ads)
    bid = np.random.uniform(0.1, 5.0, size=n_ads)

    # 预测值在真值上加高斯噪声（模拟预测误差）
    ctr_pred = np.clip(ctr_true + np.random.normal(0, 0.02, size=n_ads), 0.0, 1.0)
    cvr_pred = np.clip(cvr_true + np.random.normal(0, 0.001, size=n_ads), 0.0, 1.0)

    df = pd.DataFrame({
        "ad_id": np.arange(n_ads),
        "bid": bid,
        "ctr_true": ctr_true,
        "cvr_true": cvr_true,
        "ctr_pred": ctr_pred,
        "cvr_pred": cvr_pred,
    })
    return df


# -----------------------------
# 2) Pareto 前沿筛选（非支配解）
# -----------------------------
def pareto_frontier(points: np.ndarray) -> List[int]:
    """
    输入 points: (N,2) 数组，每行为 (x=CTR, y=Value)
    返回不被其他点支配的索引（Pareto 非支配集）
    非支配定义：点 i 被点 j 支配 iff j 在每个目标上 >= i 且至少在一个目标 > i。
    （我们希望 CTR 越大越好，Value 越大越好）
    """
    N = points.shape[0]
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]:
            continue
        for j in range(N):
            if i == j:
                continue
            # j dominates i ?
            if (points[j, 0] >= points[i, 0]) and (points[j, 1] >= points[i, 1]) \
                    and ((points[j, 0] > points[i, 0]) or (points[j, 1] > points[i, 1])):
                dominated[i] = True
                break
    return [i for i in range(N) if not dominated[i]]


# -----------------------------
# 3) 约束过滤（CTR 下界等）
# -----------------------------
def apply_constraints(df: pd.DataFrame, ctr_min: float, budget_remaining: float) -> pd.DataFrame:
    """
    简单过滤：删除预测CTR低于阈值的广告；（实际可以更复杂：budget check, freq cap 等）
    """
    df_filtered = df[df["ctr_pred"] >= ctr_min].copy()
    # 如果预算不够，也可以按 bid 阈值过滤，这里仅返回过滤结果
    return df_filtered.reset_index(drop=True)


# -----------------------------
# 4) 多目标重排（给定权重 w，score = w*norm(ctr) + (1-w)*norm(value)）
# -----------------------------
def normalize(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def rerank_with_weight(df: pd.DataFrame, w: float) -> pd.DataFrame:
    """
    value = predicted CVR * bid  (估计每次曝光的“价值”或预期收益相关因子)
    score = w * norm(ctr_pred) + (1-w) * norm(value_pred)
    返回按 score 降序排序的 DataFrame（并包含 score 列）
    """
    value_pred = df["cvr_pred"].values * df["bid"].values
    ctr_norm = normalize(df["ctr_pred"].values)
    value_norm = normalize(value_pred)
    score = w * ctr_norm + (1 - w) * value_norm
    df2 = df.copy()
    df2["score"] = score
    df2["value_pred"] = value_pred
    return df2.sort_values("score", ascending=False).reset_index(drop=True)


# -----------------------------
# 5) 简单预算调度（每个 time slot 分配预算）
# -----------------------------
def simple_budget_scheduler(budget_remaining: float, slots_left: int, pacing=1.0):
    """
    返回当前 time slot 可用的预算上限（简单的剩余预算 / 剩余时间比例）
    pacing 参数可以控制保守程度（0.8 -> 留点余量，1.2 -> 更激进）
    """
    if slots_left <= 0:
        return 0.0
    return (budget_remaining / slots_left) * pacing


# -----------------------------
# 6) MAB: Thompson Sampling（把每种 weight 当作一个 arm）
# -----------------------------
class ThompsonSamplerArms:
    def __init__(self, arms: List[float]):
        self.arms = arms  # 权重集合
        # 使用 Beta(alpha,beta) 先验来捕捉成功率（这里的成功我们定义为"conversion"发生）
        self.alpha = np.ones(len(arms))  # 成功计数 + 1
        self.beta = np.ones(len(arms))   # 失败计数 + 1
        self.counts = np.zeros(len(arms), dtype=int)
        self.sum_rewards = np.zeros(len(arms), dtype=float)  # 记录累计 conversion（或收益）

    def select_arm(self):
        # 抽样每个 arm 的成功率 theta ~ Beta(alpha, beta)，选择 theta 最大的 arm
        samples = np.random.beta(self.alpha, self.beta)
        idx = int(np.argmax(samples))
        return idx

    def update(self, arm_idx: int, successes: int, trials: int):
        """
        successes: 在本次 trials 次曝光下观测到的 conversion 次数
        trials: 曝光次数（或曝光数）
        更新 Beta 后验：
          alpha += successes
          beta += trials - successes
        """
        self.alpha[arm_idx] += successes
        self.beta[arm_idx] += (trials - successes)
        self.counts[arm_idx] += trials
        self.sum_rewards[arm_idx] += successes

    def stats(self):
        return {
            "arms": self.arms,
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "counts": self.counts.copy(),
            "sum_rewards": self.sum_rewards.copy()
        }


# -----------------------------
# 7) 全流程仿真：按 time slots 模拟投放
# -----------------------------
def simulate_day(df_ads: pd.DataFrame,
                 ctr_min=0.001,
                 budget_total=10000.0,
                 time_slots=24,
                 impressions_per_slot=500,
                 arms_weights=None):
    """
    仿真一天的投放流程
    - df_ads: 候选广告数据表（包含预测ctr/cvr/bid及真实ctr/cvr）
    - ctr_min: 约束阈值
    - budget_total: 一天总预算 (简化)
    - time_slots: 将一天划分为多少 time slot (例如按小时)
    - impressions_per_slot: 每个 slot 的曝光预算（上限）
    - arms_weights: 用于多目标重排的 w 值集合（若 None, 用默认值）
    返回日志字典（总转化/点击/花费等）
    """
    if arms_weights is None:
        arms_weights = [0.0, 0.25, 0.5, 0.75, 1.0]  # 从只看value到只看CTR的一系列策略

    # MAB 初始化
    mab = ThompsonSamplerArms(arms_weights)

    # initial filtering by CTR threshold (offline step)
    df_base = apply_constraints(df_ads, ctr_min=ctr_min, budget_remaining=budget_total)
    if df_base.shape[0] == 0:
        raise RuntimeError("No ads left after CTR min filter. Lower ctr_min or use more ads.")

    # 记录全局指标
    total_spent = 0.0
    total_impressions = 0
    total_clicks = 0
    total_conversions = 0
    total_revenue = 0.0  # 假定 revenue = bid * conversion_value_scale
    conversion_value_scale = 10.0  # 假设一次转化价值是 bid * 10（只是举例）

    # 剩余预算与 slots
    budget_remaining = budget_total
    slots_left = time_slots

    # 为效率：把 df_base 转成 numpy arrays 便于快速索引
    ads_arr = df_base.copy().reset_index(drop=True)
    n_candidates = len(ads_arr)

    for slot in range(time_slots):
        if budget_remaining <= 0:
            print(f"[slot {slot}] 预算耗尽，停止投放")
            break

        slots_left = time_slots - slot
        slot_budget = simple_budget_scheduler(budget_remaining, slots_left, pacing=1.0)
        # 这个 slot 最多能投放的曝光数（受每次曝光 cost 与 slot_budget 限制）
        # 这里简化：每次曝光的 cost = bid * cost_scale
        cost_scale = 0.2  # 简化常数：把 bid 映射到每次曝光的真实花费
        max_imps_by_budget = int(slot_budget / (ads_arr["bid"].min() * cost_scale))
        imps_allowed = min(impressions_per_slot, max_imps_by_budget)

        if imps_allowed <= 0:
            # 预算太小导致该时段无法投放
            print(f"[slot {slot}] slot_budget {slot_budget:.2f} 无法支撑一次曝光，跳过")
            continue

        # MAB 选择哪个 weight (arm) 本 slot 使用
        arm_idx = mab.select_arm()
        w = arms_weights[arm_idx]

        # 在 candidate 基础上按当前 w 做重排，选择 top-N 用来投放（此处 N = 顶部 pool 大小）
        df_ranked = rerank_with_weight(ads_arr, w)
        # 我们在本 slot 里按 round-robin 或 Top-k 轮流曝光。这里简单取 top K 并按概率抽取。
        top_k = min(50, len(df_ranked))
        top_pool = df_ranked.iloc[:top_k].reset_index(drop=True)

        # 现在开始在 top_pool 中做 imps_allowed 次曝光模拟（随机按 score 权重抽取）
        scores = top_pool["score"].values
        # 为了避免所有权重为零，稍加平滑
        prob = (scores - scores.min()) + 1e-6
        prob = prob / prob.sum()

        # 抽样出要曝光的 ad 索引（with replacement）
        chosen_idx = np.random.choice(np.arange(top_k), size=imps_allowed, p=prob)

        # 在 slot 中累计 arm-level 成功与试验次数（用于 TS 更新）
        arm_trials = imps_allowed
        arm_successes = 0

        # 对每次曝光模拟（使用真实 ctr_true 和 cvr_true 来模拟点击/转化）
        for idx in chosen_idx:
            ad = top_pool.iloc[idx]
            bid = float(ad["bid"])
            # 模拟是否发生点击（Bernoulli p = ctr_true）
            click = np.random.rand() < float(ad["ctr_true"])
            # 如果 clicked，再模拟 conversion（Bernoulli p = cvr_true）
            conversion = False
            if click:
                conversion = np.random.rand() < float(ad["cvr_true"])

            # cost & revenue 记账（简化假设）
            impression_cost = bid * cost_scale
            revenue = 0.0
            if conversion:
                # 如果发生转化，按 conversion_value_scale * bid 计收入（仅为示例）
                revenue = bid * conversion_value_scale

            # 更新全局统计
            total_spent += impression_cost
            budget_remaining -= impression_cost
            total_impressions += 1
            total_clicks += int(click)
            total_conversions += int(conversion)
            total_revenue += revenue

            arm_successes += int(conversion)

            # 如果预算被耗尽，提前停止曝光
            if budget_remaining <= 0:
                break

        # 将本 slot 的观测（conversion count & trials）回传给 MAB（Thompson Sampling）
        mab.update(arm_idx, successes=arm_successes, trials=arm_trials)

        # 打印槽级别日志（可选）
        if slot % max(1, time_slots // 6) == 0:
            print(f"[slot {slot}] chosen w={w:.2f}, imps={imps_allowed}, slot_budget={slot_budget:.2f}, "
                  f"arm_successes={arm_successes}, budget_remain={budget_remaining:.2f}")

    # 返回汇总
    summary = {
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "total_spent": total_spent,
        "total_revenue": total_revenue,
        "remaining_budget": budget_remaining,
        "mab_stats": mab.stats()
    }
    return summary


# -----------------------------
# 主函数：运行仿真并展示结果
# -----------------------------
if __name__ == "__main__":
    # 1. 生成模拟广告集
    df_ads = generate_ads(n_ads=1000)

    # 2. 设定一些业务参数
    CTR_MIN = 0.001  # 预测CTR阈值（低于就不投）
    BUDGET = 10000.0  # 一天预算（示例）
    SLOTS = 24
    IMPS_PER_SLOT = 800

    # 3. 运行仿真
    result = simulate_day(df_ads,
                          ctr_min=CTR_MIN,
                          budget_total=BUDGET,
                          time_slots=SLOTS,
                          impressions_per_slot=IMPS_PER_SLOT,
                          arms_weights=[0.0, 0.25, 0.5, 0.75, 1.0])

    # 4. 输出汇总
    print("\n=== Simulation Summary ===")
    print(f"Total Impressions: {result['total_impressions']}")
    print(f"Total Clicks:      {result['total_clicks']}")
    print(f"Total Conversions: {result['total_conversions']}")
    print(f"Total Spent:       {result['total_spent']:.2f}")
    print(f"Total Revenue:     {result['total_revenue']:.2f}")
    print(f"Remaining Budget:  {result['remaining_budget']:.2f}")
    print("MAB (arms, alpha, beta, counts, sum_rewards):")
    print(result["mab_stats"])
