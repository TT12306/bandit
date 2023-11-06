import numpy as np
import random

lr = 0.2
min_lr = 0.01
lr_decay = (lr - min_lr) / 100  # 每个epoch减少的学习率 线性衰减
num_machines = 10
initial_scores = np.full(num_machines, 1000)  # 设置初始分数为1000
rewards_mean = np.random.normal(500, 100, num_machines)  # 每个老虎机的奖励均值
for j in range(num_machines):
    print('第{}个老虎机 Score: {} | Reward Mean: {}'.format(j + 1, initial_scores[j], rewards_mean[j]))


# 初始化奖励和次数
rewards = np.zeros(num_machines)
times_played = np.zeros(num_machines)

# 更新分数和奖励
def update_score_reward(score, reward, round_score, t, chosen_machine):
    reward = np.random.normal(rewards_mean[chosen_machine], 10)
    avg = (score * (t + 1) + reward) / (t + 2)
    return avg, reward, t + 1


# 迭代
for i in range(100):
    print('Epoch {} start: '.format(i + 1))

    for j in range(num_machines):
        print('第{}个老虎机 Score: {} | Reward Mean: {}'.format(j + 1, initial_scores[j], rewards_mean[j]))

    p = np.random.random()
    print("选择老虎机: ", end='')

    if p < lr:  # 执行一次随机选择
        print('\nRandom! ', end='')
        chosen_machine = np.random.choice(num_machines)
        print('TigerMachine{}'.format(chosen_machine + 1))
        initial_scores[chosen_machine], rewards[chosen_machine], times_played[chosen_machine] = update_score_reward(
            initial_scores[chosen_machine], rewards[chosen_machine], initial_scores[chosen_machine], times_played[chosen_machine], chosen_machine)
    else:  # 正常选择分数高的老虎机
        chosen_machine = np.argmax(initial_scores)
        print('TigerMachine{}'.format(chosen_machine + 1))
        initial_scores[chosen_machine], rewards[chosen_machine], times_played[chosen_machine] = update_score_reward(
            initial_scores[chosen_machine], rewards[chosen_machine], initial_scores[chosen_machine], times_played[chosen_machine], chosen_machine)
    print('The rewards are:', rewards)
    print('The scores are:', initial_scores)
    
    print('================================================')

    # 更新学习率
    lr = max(min_lr, lr - lr_decay)

chosen_machine = np.argmax(initial_scores)
print("最优老虎机: TigerMachine{}!".format(chosen_machine + 1))
