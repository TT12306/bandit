import numpy as np

# 参数初始化
num_machines = 10
H = np.zeros(num_machines)  # 初始化行动的偏好值
rewards_mean = np.random.normal(500, 100, num_machines)  # 每个老虎机的奖励均值
alpha = 0.003 # 太大会出现较多老虎机没有得到探索

# 初始化奖励和次数
rewards = np.zeros(num_machines)  # 初始化奖励
times_played = np.zeros(num_machines)  # 初始化每个老虎机拉的次数

# 获取行动的概率分布
def get_action_probabilities(H):
    exp_H = np.exp(H - np.max(H))
    return exp_H / np.sum(exp_H)
    ##

# 更新行动的偏好值
def update_preferences(H, chosen_machine, reward, baseline, alpha):
    pi_t = get_action_probabilities(H)
    H[chosen_machine] += alpha * (reward - baseline) * (1 - pi_t[chosen_machine])
    for i in range(len(H)):
        if i != chosen_machine:
            H[i] -= alpha * (reward - baseline) * pi_t[i]
    return H

# 基准值（平均奖励）
def get_baseline(rewards, times_played):
    total_rewards = np.sum(rewards)
    total_played = np.sum(times_played)
    if total_played > 0:
        baseline = total_rewards / total_played
    else:
        baseline = 0
    return baseline

# 迭代
for i in range(100):
    print('Epoch {} start: '.format(i + 1))
    pi_t = get_action_probabilities(H)
    
    # 根据得到的概率分布选择一个老虎机
    chosen_machine = np.random.choice(range(num_machines), p=pi_t)
    reward = np.random.normal(rewards_mean[chosen_machine], 10)
    rewards[chosen_machine] += reward
    times_played[chosen_machine] += 1
    
    baseline = get_baseline(rewards, times_played)
    
    for j in range(num_machines):
        if times_played[j] > 0:
            average_reward = rewards[j] / times_played[j]
        else:
            average_reward = 0
        print('第{}个老虎机 Reward avg: {} | Reward Mean: {} | 选择次数：{}'.format(j + 1, average_reward, rewards_mean[j], times_played[j]))

    # 更新偏好值
    H = update_preferences(H, chosen_machine, reward, baseline, alpha)
    
    print("选择的老虎机: {}, Reward: {}".format(chosen_machine + 1, reward))
    print('偏好:', H)
    print('概率分布:', pi_t)
    print('================================================')

# 最终选择最好的老虎机
best_machine = np.argmax(H)
print("最优老虎机: TigerMachine{}!".format(best_machine + 1))
