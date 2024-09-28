import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from env import SchedulingEnv
from model import baseline_DQN, baselines, BLOR
from utils import get_args


args = get_args()
# store result
performance_lamda = np.zeros(args.Baseline_num)
performance_total_rewards = np.zeros(args.Baseline_num)
performance_success = np.zeros(args.Baseline_num)
performance_success_time = np.zeros(args.Baseline_num)
performance_match = np.zeros(args.Baseline_num)
performance_finishT = np.zeros(args.Baseline_num)
performance_cost = np.zeros(args.Baseline_num)
performance_assigned_malicious_num = np.zeros(args.Baseline_num)
performance_assigned_normal_num = np.zeros(args.Baseline_num)
performance_assigned_trusted_num = np.zeros(args.Baseline_num)

# gen env
env = SchedulingEnv(args)


# build model
brainRL = baseline_DQN(env.actionNum, env.s_features)
brainOthers = baselines(env.actionNum, env.oracleTypes)

global_step = 0
my_learn_step = 0
DQN_Reward_list = []
My_reward_list = []
for episode in range(args.Epoch):

    print('----------------------------Episode', episode, '----------------------------')
    request_c = 1  # request counter
    time_period = 1  # time period counter
    BLOR_c = 1  # BLOR update counter
    performance_c = 0
    env.reset(args)  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    env.reset_reputation_factors()
    env.initial_reputation()
    performance_respTs = []
    brainRL.reward_list.clear()
    while True:
        # update reputation
        if request_c % 60 == 0:
            time_period += 1
            # random policy
            reputation_attributes_RAN = env.get_reputation_factors(1)
            env.update_reputation(reputation_attributes_RAN, time_period, 1)
            # round robin policy
            reputation_attributes_RR = env.get_reputation_factors(2)
            env.update_reputation(reputation_attributes_RR, time_period, 2)
            # earliest policy
            reputation_attributes_early = env.get_reputation_factors(3)
            env.update_reputation(reputation_attributes_early, time_period, 3)
            # DQN policy
            reputation_attributes_DQN = env.get_reputation_factors(4)
            env.update_reputation(reputation_attributes_DQN, time_period, 4)
            # PSG policy
            reputation_attributes_PSG = env.get_reputation_factors(7)
            env.update_reputation(reputation_attributes_PSG, time_period, 7)

            env.reset_reputation_factors()

        # baseline DQN
        global_step += 1
        finish, request_attrs = env.workload(request_c)
        DQN_state = env.getState(request_attrs, 4)
        if global_step != 1:
                brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        reward_DQN = env.feedback(request_attrs, action_DQN, 4)
        # DQN_Reward_Training.append(reward_DQN)
        if episode==1:
            DQN_Reward_list.append(reward_DQN)
        if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):  # learn
            brainRL.learn()
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

        # random policy
        state_Ran = env.getState(request_attrs, 1)
        action_random = brainOthers.random_choose_action()
        reward_random = env.feedback(request_attrs, action_random, 1)
        # round robin policy
        state_RR = env.getState(request_attrs, 2)
        action_RR = brainOthers.RR_choose_action(request_c)
        reward_RR = env.feedback(request_attrs, action_RR, 2)
        # earliest policy
        idleTimes = env.get_oracle_idleT(3)  # get oracle state
        action_early = brainOthers.early_choose_action(idleTimes)
        reward_early = env.feedback(request_attrs, action_early, 3)
        # BLOR policy
        # adopt round-robin policy to initialize BLOR factors at time period 1
        start_counter = (BLOR_c - 1) * 200
        RR_counter = start_counter + 15
        end_counter = BLOR_c * 200
        if request_c > start_counter and request_c < RR_counter + 1:
            state_BLOR = env.getState(request_attrs, 6)
            action_BLOR = brainOthers.RR_choose_action(request_c)
            reward_BLOR = env.feedback(request_attrs, action_BLOR, 6)
        elif request_c > RR_counter and request_c <  end_counter + 1:
            request_num_BLOR = env.get_request_num(6)
            success_num_BLOR = env.get_successful_validation(6)
            failure_num_BLOR = request_num_BLOR - success_num_BLOR
            brainBLOR = BLOR(success_num_BLOR, failure_num_BLOR, env.oracleCost)
            oracles_BLOR = brainBLOR.get_oracles(success_num_BLOR, failure_num_BLOR, env.oracleCost)
            action_BLOR = brainBLOR.choose_action(oracles_BLOR)
            reward_BLOR = env.feedback(request_attrs, action_BLOR, 6)
        if request_c % 200 == 0:
            env.reset_reputation_factors_BLOR()
            BLOR_c += 1
        # semiGreedy policy
        rewards_PSG, cost_PSG = env.feedback_PSG_FWA(request_attrs, 7)  # get oracle state
        action_PSG = brainOthers.PSG_choose_action(rewards_PSG, cost_PSG)
        reward_PSG = env.feedback(request_attrs, action_PSG, 7)


        if request_c % 500 == 0:
            acc_Rewards = env.get_accumulateRewards(args.Baseline_num, performance_c, request_c)
            cost = env.get_accumulateCost(args.Baseline_num, performance_c, request_c)
            finishTs = env.get_FinishTimes(args.Baseline_num, performance_c, request_c)
            avg_exeTs = env.get_executeTs(args.Baseline_num, performance_c, request_c)
            avg_waitTs = env.get_waitTs(args.Baseline_num, performance_c, request_c)
            avg_respTs = env.get_responseTs(args.Baseline_num, performance_c, request_c)
            performance_respTs.append(avg_respTs)
            successTs = env.get_successTimes(args.Baseline_num, performance_c, request_c)
            successInTime = env.get_successInTime(args.Baseline_num, performance_c, request_c)
            performance_c = request_c

        request_c += 1
        if finish:
            break

    # episode performance
    startP = 2000

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    total_success_time = env.get_totalSuccessInTime(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)
    print('total performance (after 2000 requests):')
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(name + " reward:", total_Rewards[i], ' avg_responseT:', avg_allRespTs[i],
              'success_rate:', total_success[i], 'success_time_rate:', total_success_time[i], ' finishT:', total_Ts[i], 'Cost:', total_cost[i])

    if episode != 0:
        performance_lamda[:] += env.get_total_responseTs(args.Baseline_num, 0)
        performance_total_rewards[:] += env.get_totalRewards(args.Baseline_num, 0)
        performance_success[:] += env.get_totalSuccess(args.Baseline_num, 0)
        performance_success_time[:] += env.get_totalSuccessInTime(args.Baseline_num, 0)
        performance_finishT[:] += env.get_totalTimes(args.Baseline_num, 0)
        performance_cost += env.get_totalCost(args.Baseline_num, 0)
        performance_match += env.get_totalMatchRate(args.Baseline_num)
        performance_assigned_malicious_num[:] += env.get_totalMaliciousNum(args.Baseline_num)
        performance_assigned_normal_num[:] += env.get_totalNormalNum(args.Baseline_num)
        performance_assigned_trusted_num[:] += env.get_totalTrustedNum(args.Baseline_num)

    if episode == 0:
        # plot DQN convergence curves
        sns.set_style("darkgrid")
        window_size = 30
        # calculate moving average reward
        rewards_series = pd.Series(brainRL.reward_list)
        moving_avg_rewards = rewards_series.rolling(window=window_size).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(brainRL.reward_list, label='reward')
        plt.plot(moving_avg_rewards, label='ma reward', color='darkorange')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./output/reward_episode{episode}.pdf', dpi=600)
        plt.show()

print('---------------------------- Final results ----------------------------')
performance_lamda = np.around(performance_lamda/(args.Epoch-1), 3)
performance_total_rewards = np.around(performance_total_rewards/(args.Epoch-1), 3)
performance_success = np.around(performance_success/(args.Epoch-1), 3)
performance_success_time = np.around(performance_success_time/(args.Epoch-1), 3)
performance_finishT = np.around(performance_finishT/(args.Epoch-1), 3)
performance_cost = np.around(performance_cost/(args.Epoch-1), 5)
performance_match = np.around(performance_match/(args.Epoch-1), 3)
performance_assigned_malicious_num = np.around(performance_assigned_malicious_num/(args.Epoch-1), 0)
performance_assigned_normal_num = np.around(performance_assigned_normal_num/(args.Epoch-1), 0)
performance_assigned_trusted_num = np.around(performance_assigned_trusted_num/(args.Epoch-1), 0)
print('avg_responseT:')
print(performance_lamda)
print('total_rewards:')
print(performance_total_rewards)
print('success_rate:')

print(performance_success)
print('success_time_rate:')
print(performance_success_time)
print('finishT:')
print(performance_finishT)
print('cost:')
print(performance_cost)
print('match rate:')
print(performance_match)
print('requests assigned to malicious oracle:')
print(performance_assigned_malicious_num)
print('requests assigned to normal oracle:')
print(performance_assigned_normal_num)
print('requests assigned to trusted oracle:')
print(performance_assigned_trusted_num)

# # plot DQN reputations curve
# plt.figure(figsize=(12, 8))
# for i in range(env.DQN_oracle_reputation_history.shape[1]):
#     plt.plot(np.arange(env.DQN_oracle_reputation_history.shape[0]), env.DQN_oracle_reputation_history[:, i], label=f'Oracle {i}')
# min_val = np.around(np.min(env.DQN_oracle_reputation_history) - 1, 0)
# max_val = np.around(np.max(env.DQN_oracle_reputation_history) + 1, 0)
# plt.yticks(np.arange(min_val, max_val, 1))
# plt.xlabel('Time Period')
# plt.ylabel('Reputation')
# plt.legend()
# plt.savefig(f'./output/reputation_episode{episode}.pdf', dpi=600)
# plt.show()
#
# # plot DQN reputations curve without malicious oracles
# plt.figure(figsize=(12, 8))
# malicious_oracles_index = [0, 5, 10]
# # delete malicious oracles
# data_remaining = np.delete(env.DQN_oracle_reputation_history, malicious_oracles_index, axis=1)
# remaining_columns = [i for i in range(env.DQN_oracle_reputation_history.shape[1]) if i not in malicious_oracles_index]
# plt.figure(figsize=(12, 8))
# x = np.arange(data_remaining.shape[0])
# for i, col in enumerate(remaining_columns):
#     y = data_remaining[:, i]
#     spline = make_interp_spline(x, y)
#     x_smooth = np.linspace(x.min(), x.max(), 300)
#     y_smooth = spline(x_smooth)
#     plt.plot(x_smooth, y_smooth, label=f'Oracle {col}')
#
# plt.xlabel('Time Period')
# plt.ylabel('Reputation')
# plt.legend()
# plt.savefig(f'./output/reputation_without_malicious_oracles_episode{episode}.pdf', dpi=600)
# plt.show()
#
# # plot DQN reputations curve in one type
# plt.figure(figsize=(12, 8))
# malicious_oracles_index = [0,5,6,7,8,9,10,11,12,13,14]
# # delete malicious oracles
# data_remaining = np.delete(env.DQN_oracle_reputation_history, malicious_oracles_index, axis=1)
# remaining_columns = [i for i in range(env.DQN_oracle_reputation_history.shape[1]) if i not in malicious_oracles_index]
# plt.figure(figsize=(12, 8))
# x = np.arange(data_remaining.shape[0])
# for i, col in enumerate(remaining_columns):
#     y = data_remaining[:, i]
#     spline = make_interp_spline(x, y)
#     x_smooth = np.linspace(x.min(), x.max(), 300)
#     y_smooth = spline(x_smooth)
#     plt.plot(x_smooth, y_smooth, label=f'Oracle {col}')
# min_val = np.around(np.min(data_remaining) - 3, 0)
# max_val = np.around(np.max(data_remaining) + 1, 0)
# plt.yticks(np.arange(min_val, max_val, 0.5))
# plt.xlabel('Time Period')
# plt.ylabel('Reputation')
# plt.legend()
# plt.savefig(f'./output/reputation_in_one_type_episode{episode}.pdf', dpi=600)
# plt.show()
#
#
# # plot DQN request num
# plt.figure(figsize=(10, 6))
# request_num_DQN = env.DQN_oracle_events[1]
# plt.bar(np.arange(len(request_num_DQN)), request_num_DQN, color='steelblue')
#
# plt.xlabel('Oracle ID')
# plt.ylabel('Request Number')
# plt.xticks(np.arange(len(request_num_DQN)))
# plt.savefig(f'./output/request_num_episode{episode}.pdf', dpi=600)
# plt.show()
print('')

'''
reward_index = [0, 999, 1999, 2999, 3999, 4999, 5999, 6999, 7999]
DQN_Reward_Reuslt = []
AIRL_Reward_Result = []
print(DQN_Reward_list)
for t in range(len(reward_index) - 1):
    DQN_r = sum(DQN_Reward_list[reward_index[t]: reward_index[t+1]])
    AIRL_r = sum(My_reward_list[reward_index[t]: reward_index[t+1]])
    DQN_Reward_Reuslt.append(DQN_r)
    AIRL_Reward_Result.append(AIRL_r)
print(DQN_Reward_Reuslt)
print(AIRL_Reward_Result)
'''