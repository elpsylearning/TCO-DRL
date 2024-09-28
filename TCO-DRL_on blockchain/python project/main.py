import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from env import SchedulingEnv
from model import baseline_DQN
from utils import get_args
from matplotlib.pyplot import MultipleLocator
import random
import threading
import socket
import struct
from scipy.interpolate import lagrange
import time
import psutil
import os

#blockchain
from web3 import Web3
import json
args = get_args()

# Get accounts created in ganache (virtual private chain nodes)
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

accounts_all = w3.eth.accounts
accounts = accounts_all[0:len(accounts_all)-1]
process_account = accounts_all[len(accounts_all)-1:]


SELECTION_CONTRACT_ADDR = ''

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

performance_total_latency = 0
performance_average_latency = 0
performance_average_gas_used = 0
performance_average_cpu_usage = 0
# performance_total_memory_usage = []


# gen env
env = SchedulingEnv(args)


# build model
brainRL = baseline_DQN(env.actionNum, env.s_features)

global_step = 0
my_learn_step = 0
DQN_Reward_list = []
My_reward_list = []


# Calling the contract returns the selection result
class Contract():
    def __init__(self):

        self.web3 = Web3(Web3.WebsocketProvider("ws://127.0.0.1:8545"))
        # Check if the connection is successful
        if self.web3.eth.getBlock(0) is None:
            print("Failed to connect!")
        elif self.web3.isConnected():
            with open(r'/home/blockchain/TCO-DRL/build/contracts/Selection_abi.json',
                      'r') as abi_definition:
                self.abi = json.load(abi_definition)
                # print('===============================', self.abi)
                self.SelectionAddr = SELECTION_CONTRACT_ADDR
                self.Selection = self.web3.eth.contract(address=self.SelectionAddr, abi=self.abi)
                print("Successfully connected")

                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def ReAction(self, action_oracle):
        # Record start time
        start_time = time.time()

        # Record start resource utilization
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent(interval=None)
        start_memory = process.memory_info().rss

        # Initiate a transaction
        self.action = self.Selection.functions.ReAction(action_oracle).transact({
            'from': '',
            'to': self.SelectionAddr
        })
        print("Uploading the selection results is the number {0} oracle".format(action_oracle))

        # Waiting for transaction receipt
        receipt = self.web3.eth.waitForTransactionReceipt(self.action)

        # Record end time
        end_time = time.time()

        # Record end resource utilization
        end_cpu = process.cpu_percent(interval=None)
        # end_memory = process.memory_info().rss

        # Calculate latency
        latency = end_time - start_time
        print("The latency of this transaction : {:.3f} s".format(latency))
        # Get and print Gas consumption
        gas_used = receipt['gasUsed']
        print("The gas consumption of this transaction: {}".format(gas_used))
        # Calculate resource utilization
        cpu_usage = end_cpu - start_cpu
        # memory_usage = end_memory - start_memory
        print("CPU usage: {:.2f}%".format(cpu_usage))
        # print("Memory usage: {:.2f} MB".format(memory_usage / (1024 * 1024)))

        print("Transaction receipt: {}".format(receipt))
        return latency, gas_used, cpu_usage


    def SeAction(self):
        a = self.Selection.functions.SeAction().call()
        return a



for episode in range(args.Epoch):

    print('----------------------------Episode', episode, '----------------------------')
    request_c = 1  # request counter
    time_period = 1  # time period counter
    performance_c = 0
    env.reset(args)  # attention: whether generate new workload, if yes, don't forget to modify reset() function
    env.reset_reputation_factors()
    env.initial_reputation()
    performance_respTs = []
    brainRL.reward_list.clear()

    # contract
    total_latency = 0
    total_gas_used = 0
    total_cpu_usage = 0
    # total_memory_usage = 0
    while True:
        # update reputation
        if request_c % 60 == 0:
            time_period += 1
            # DQN policy
            reputation_attributes_DQN = env.get_reputation_factors()
            env.update_reputation(reputation_attributes_DQN, time_period)
            env.reset_reputation_factors()

        # baseline DQN
        global_step += 1
        finish, request_attrs = env.workload(request_c)
        DQN_state = env.getState(request_attrs)
        if global_step != 1:
                brainRL.store_transition(last_state, last_action, last_reward, DQN_state)
        action_DQN = brainRL.choose_action(DQN_state)  # choose action
        action_DQN_int = int(action_DQN)
        reward_DQN = env.feedback(request_attrs, action_DQN)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #########################################################  Call Selection Contract  ############################################################
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        C1 = Contract()
        latency, gas_used, cpu_usage = C1.ReAction(action_DQN_int)
        total_latency += latency
        total_gas_used += gas_used
        total_cpu_usage += cpu_usage
        # total_memory_usage += memory_usage
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # DQN_Reward_Training.append(reward_DQN)
        if episode==1:
            DQN_Reward_list.append(reward_DQN)
        if (global_step > args.Dqn_start_learn) and (global_step % args.Dqn_learn_interval == 0):  # learn
            brainRL.learn()
        last_state = DQN_state
        last_action = action_DQN
        last_reward = reward_DQN

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
    startP = 1000

    total_Rewards = env.get_totalRewards(args.Baseline_num, startP)
    avg_allRespTs = env.get_total_responseTs(args.Baseline_num, startP)
    total_success = env.get_totalSuccess(args.Baseline_num, startP)
    total_success_time = env.get_totalSuccessInTime(args.Baseline_num, startP)
    total_Ts = env.get_totalTimes(args.Baseline_num, startP)
    total_cost = env.get_totalCost(args.Baseline_num, startP)

    # Calculate average latency
    average_latency = total_latency / request_c

    # Calculate average Gas consumption
    average_gas_used = total_gas_used / request_c

    # Calculate average CPU usage
    average_cpu_usage = total_cpu_usage / request_c


    print('total performance (after 1000 requests):')
    for i in range(len(args.Baselines)):
        name = "[" + args.Baselines[i] + "]"
        print(name + " reward:", total_Rewards[i], ' avg_responseT:', avg_allRespTs[i],
              'success_rate:', total_success[i], 'success_time_rate:', total_success_time[i], ' finishT:', total_Ts[i], 'Cost:', total_cost[i],
              'total latency:', total_latency, 'average latency:', average_latency, 'average Gas consumption:', average_gas_used, 'average CPU usage:', average_cpu_usage)

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
        performance_total_latency += total_latency
        performance_average_latency += average_latency
        performance_average_gas_used += average_gas_used
        performance_average_cpu_usage += average_cpu_usage
        # performance_total_memory_usage += total_memory_usage

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
performance_total_latency = np.around(performance_total_latency/(args.Epoch-1), 0)
performance_average_latency = np.around(performance_average_latency/(args.Epoch-1), 0)
performance_average_gas_used = np.around(performance_average_gas_used/(args.Epoch-1), 0)
performance_average_cpu_usage = np.around(performance_average_cpu_usage/(args.Epoch-1), 0)
# performance_total_memory_usage = np.around(performance_total_memory_usage/(args.Epoch-1), 0)
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

print('total_latency:')
print(performance_total_latency)
print('average_latency:')
print(performance_average_latency)
print('average_gas_used:')
print(performance_average_gas_used)
print('average_cpu_usage:')
print(performance_average_cpu_usage)
# print('total_memory_usage:')
# print(performance_total_memory_usage)

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