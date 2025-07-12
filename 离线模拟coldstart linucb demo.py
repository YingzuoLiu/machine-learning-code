# Cold Start + LinUCB Personalized Recommendation System (Airbnb-like)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- 1. Set Up ------------------------- #
user_groups = ['new', 'returning']
scenes = ['morning', 'evening']
n_users = 200
embedding_dim = 5

users = pd.DataFrame({
    'user_id': range(n_users),
    'group': np.random.choice(user_groups, size=n_users, p=[0.4, 0.6]),
    'scene': np.random.choice(scenes, size=n_users, p=[0.5, 0.5]),
    'embedding': [np.random.randn(embedding_dim) for _ in range(n_users)]
})

n_listings = 10
listings = pd.DataFrame({
    'listing_id': range(n_listings),
    'is_new': [True if i < 3 else False for i in range(n_listings)],
    'base_ctr': [np.nan if i < 3 else np.random.uniform(0.2, 0.5) for i in range(n_listings)],
    'embedding': [np.random.randn(embedding_dim) for _ in range(n_listings)]
})

hidden_ctr = [np.random.uniform(0.1, 0.4) if i < 3 else ctr for i, ctr in enumerate(listings['base_ctr'])]

# ------------------------- 2. Alpha Settings ------------------------- #
group_scene_alpha = {
    ('new', 'morning'): 0.7,
    ('new', 'evening'): 0.5,
    ('returning', 'morning'): 0.3,
    ('returning', 'evening'): 0.1
}

# ------------------------- 3. Initialize Policies ------------------------- #
listing_stats_linucb = {
    i: {
        'A': np.identity(embedding_dim * 2),
        'b': np.zeros(embedding_dim * 2)
    } for i in range(n_listings)
}

avg_reward_greedy = [0.0 for _ in range(n_listings)]
count_greedy = [0 for _ in range(n_listings)]

count_ucb = [0 for _ in range(n_listings)]
sum_reward_ucb = [0.0 for _ in range(n_listings)]

# Bootstrapping counters
bootstrapping_phase = {i: 5 if listings.loc[i, 'is_new'] else 0 for i in range(n_listings)}

log_records = []

# ------------------------- 4. Main Loop ------------------------- #
for t in range(1000):
    user = users.sample(1).iloc[0]
    u_embed = user['embedding']
    group, scene = user['group'], user['scene']
    alpha = group_scene_alpha[(group, scene)]

    scores = {'linucb': [], 'greedy': [], 'ucb': []}

    for i in range(n_listings):
        l_embed = listings.loc[i, 'embedding']
        x = np.concatenate([u_embed, l_embed])

        # LinUCB
        stats = listing_stats_linucb[i]
        A_inv = np.linalg.inv(stats['A'])
        theta = A_inv @ stats['b']
        p_linucb = x @ theta + alpha * np.sqrt(x @ A_inv @ x)
        scores['linucb'].append(p_linucb)

        # Greedy
        p_greedy = avg_reward_greedy[i] if count_greedy[i] > 0 else 0.5
        scores['greedy'].append(p_greedy)

        # UCB1
        total_count = sum(count_ucb)
        if count_ucb[i] == 0:
            p_ucb = 1.0
        else:
            p_ucb = (sum_reward_ucb[i] / count_ucb[i]) + alpha * np.sqrt(np.log(total_count + 1) / count_ucb[i])
        scores['ucb'].append(p_ucb)

    for policy in ['linucb', 'greedy', 'ucb']:
        # Bootstrapping phase: force recommend unseen new listings
        forced = None
        for i in range(n_listings):
            if bootstrapping_phase[i] > 0:
                forced = i
                bootstrapping_phase[i] -= 1
                break

        chosen = forced if forced is not None else int(np.argmax(scores[policy]))
        reward = np.random.rand() < hidden_ctr[chosen]

        # Update LinUCB
        if policy == 'linucb':
            x = np.concatenate([u_embed, listings.loc[chosen, 'embedding']])
            stats = listing_stats_linucb[chosen]
            stats['A'] += np.outer(x, x)
            stats['b'] += reward * x

        # Update Greedy
        if policy == 'greedy':
            count_greedy[chosen] += 1
            avg_reward_greedy[chosen] += (reward - avg_reward_greedy[chosen]) / count_greedy[chosen]

        # Update UCB
        if policy == 'ucb':
            count_ucb[chosen] += 1
            sum_reward_ucb[chosen] += reward

        log_records.append({
            'step': t,
            'policy': policy,
            'listing_id': chosen,
            'reward': reward,
            'is_new': listings.loc[chosen, 'is_new']
        })

# ------------------------- 5. Visualization ------------------------- #
log_df = pd.DataFrame(log_records)
plt.figure(figsize=(12, 6))

for policy in ['linucb', 'greedy', 'ucb']:
    policy_df = log_df[log_df['policy'] == policy]
    ctr_series = policy_df['reward'].rolling(100, min_periods=1).mean()
    plt.plot(ctr_series.reset_index(drop=True), label=policy.title())

plt.title('CTR Comparison: LinUCB vs Greedy vs UCB1 (with User + Listing Embeddings)')
plt.xlabel('Steps')
plt.ylabel('Rolling CTR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
