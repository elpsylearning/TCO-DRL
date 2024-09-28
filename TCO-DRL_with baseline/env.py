import numpy as np
from scipy import stats

np.random.seed(3)


class SchedulingEnv:
    def __init__(self, args):
        #Environment Settings
        self.policy_num = len(args.Baselines)

        # Oracle Setting
        self.oracleTypes = args.Oracle_Type
        self.oracleNum = args.Oracle_Num
        assert self.oracleNum == len(self.oracleTypes)
        self.oracleCapacity = args.Oracle_capacity
        self.actionNum = args.Oracle_Num
        self.s_features = 1 + 2*args.Oracle_Num  # request type & oracle idle time & oracle reputation
        self.oracleInitialReputation = args.Oracle_Initial_Reputation
        self.oracleAcc = args.Oracle_Acc
        self.oracleCost = args.Oracle_Cost
        self.oracleToken = args.Oracle_Tokens
        self.oracleBehaviorProbs = args.Oracle_Behavior_Probs
        self.oracleValidationProbs = args.Oracle_Validation_Probs

        # Request Setting
        self.requestMI = args.Request_len_Mean
        self.requestMI_std = args.Request_len_Std
        self.requestNum = args.Request_Num
        self.lamda = args.lamda
        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum)
        self.ddl = args.Request_ddl

        # Reputation Setting
        # reputation_factors: 1-assigned request num   2-successful validation num   3-total response time
        #                     4-behavior records
        self.RAN_reputation_factors = np.zeros((4, self.oracleNum))
        self.RR_reputation_factors = np.zeros((4, self.oracleNum))
        self.early_reputation_factors = np.zeros((4, self.oracleNum))
        self.DQN_reputation_factors = np.zeros((4, self.oracleNum))
        self.BLOR_reputation_factors = np.zeros((4, self.oracleNum))
        self.PSG_reputation_factors = np.zeros((4, self.oracleNum))

        # TimeWindow Setting
        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(args.Request_Num / args.Time_Period_Size) + 1
        # generate workload
        self.gen_workload(self.lamda)


        # request: 1-oracle id  2-start time  3-wait time  4-waitT+exeT  5-leave time  6-reward  7-actual_exeT
        #          8- success   9-reject    10- success without type factor

        # oracle: 1-idleT   2-request num   3-reputation   4-match num   5-successful validation num

        # Random
        self.RAN_events = np.zeros((10, self.requestNum))
        self.RAN_oracle_events = np.zeros((5, self.oracleNum))
        self.RAN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.RAN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.RAN_reputation_timewindow = np.zeros((0, self.oracleNum))
        # Round Robin
        self.RR_events = np.zeros((10, self.requestNum))
        self.RR_oracle_events = np.zeros((5, self.oracleNum))
        self.RR_oracle_events[2] = args.Oracle_Initial_Reputation
        self.RR_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.RR_reputation_timewindow = np.zeros((0, self.oracleNum))
        # Earliest
        self.early_events = np.zeros((10, self.requestNum))
        self.early_oracle_events = np.zeros((5, self.oracleNum))
        self.early_oracle_events[2] = args.Oracle_Initial_Reputation
        self.early_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.early_reputation_timewindow = np.zeros((0, self.oracleNum))
        # DQN
        self.DQN_events = np.zeros((10, self.requestNum))
        self.DQN_oracle_events = np.zeros((5, self.oracleNum))
        self.DQN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.DQN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.DQN_reputation_timewindow = np.zeros((0, self.oracleNum))
        # BLOR 
        self.BLOR_events = np.zeros((10, self.requestNum))
        self.BLOR_oracle_events = np.zeros((5, self.oracleNum))
        self.BLOR_oracle_events[2] = args.Oracle_Initial_Reputation
        self.BLOR_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.BLOR_reputation_timewindow = np.zeros((0, self.oracleNum))
        # SemiGreedy
        self.PSG_events = np.zeros((10, self.requestNum))
        self.PSG_oracle_events = np.zeros((5, self.oracleNum))
        self.PSG_oracle_events[2] = args.Oracle_Initial_Reputation
        self.PSG_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.PSG_reputation_timewindow = np.zeros((0, self.oracleNum))
        
    def gen_workload(self, lamda):
        # Generate arrival time of requests (poisson distribution)
        intervalT = stats.expon.rvs(scale=1 / lamda*60, size=self.requestNum)
        print("intervalT mean: ", round(np.mean(intervalT), 3),
              '  intervalT SD:', round(np.std(intervalT, ddof=1), 3))
        self.arrival_Times = np.around(intervalT.cumsum(), decimals=3)
        last_arrivalT = self.arrival_Times[- 1]
        print('last request arrivalT:', round(last_arrivalT, 3))

        # Generate requests' length(Normal distribution)
        self.requestsMI = np.random.normal(self.requestMI, self.requestMI_std, self.requestNum)
        self.requestsMI = self.requestsMI.astype(int)
        print("MI mean: ", round(np.mean(self.requestsMI), 3), '  MI SD:', round(np.std(self.requestsMI, ddof=1), 3))
        self.lengths = self.requestsMI / self.oracleCapacity
        print("length mean: ", round(np.mean(self.lengths), 3), '  length SD:', round(np.std(self.lengths, ddof=1), 3))

        # generate requests' type
        types = np.random.choice([0, 1, 2], size=self.requestNum, p=[1/3, 1/3, 1/3])
        self.request_type = types

    def reset(self, args):
        # if each episode generates new workload
        self.arrival_Times = np.zeros(self.requestNum)
        self.requestsMI = np.zeros(self.requestNum)
        self.lengths = np.zeros(self.requestNum)
        self.request_type = np.zeros(self.requestNum)
        self.ddl = args.Request_ddl
        self.gen_workload(args.lamda)
        self.oracleTypes = args.Oracle_Type

        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(args.Request_Num / args.Time_Period_Size) + 1
        # reset all records
        # Random
        self.RAN_events = np.zeros((10, self.requestNum))
        self.RAN_oracle_events = np.zeros((5, self.oracleNum))
        self.RAN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.RAN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.RAN_reputation_timewindow = np.zeros((0, self.oracleNum))
        # Round Robin
        self.RR_events = np.zeros((10, self.requestNum))
        self.RR_oracle_events = np.zeros((5, self.oracleNum))
        self.RR_oracle_events[2] = args.Oracle_Initial_Reputation
        self.RR_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.RR_reputation_timewindow = np.zeros((0, self.oracleNum))
        # Earliest
        self.early_events = np.zeros((10, self.requestNum))
        self.early_oracle_events = np.zeros((5, self.oracleNum))
        self.early_oracle_events[2] = args.Oracle_Initial_Reputation
        self.early_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.early_reputation_timewindow = np.zeros((0, self.oracleNum))
        # DQN
        self.DQN_events = np.zeros((10, self.requestNum))
        self.DQN_oracle_events = np.zeros((5, self.oracleNum))
        self.DQN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.DQN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.DQN_reputation_timewindow = np.zeros((0, self.oracleNum))
        # BLOR
        self.BLOR_events = np.zeros((10, self.requestNum))
        self.BLOR_oracle_events = np.zeros((5, self.oracleNum))
        self.BLOR_oracle_events[2] = args.Oracle_Initial_Reputation
        self.BLOR_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.BLOR_reputation_timewindow = np.zeros((0, self.oracleNum))
        # SemiGreedy
        self.PSG_events = np.zeros((10, self.requestNum))
        self.PSG_oracle_events = np.zeros((5, self.oracleNum))
        self.PSG_oracle_events[2] = args.Oracle_Initial_Reputation
        self.PSG_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.PSG_reputation_timewindow = np.zeros((0, self.oracleNum))

    def reset_reputation_factors(self):
        self.RAN_reputation_factors = np.zeros((4, self.oracleNum))
        self.RR_reputation_factors = np.zeros((4, self.oracleNum))
        self.early_reputation_factors = np.zeros((4, self.oracleNum))
        self.DQN_reputation_factors = np.zeros((4, self.oracleNum))
        self.PSG_reputation_factors = np.zeros((4, self.oracleNum))

    def reset_reputation_factors_BLOR(self):
        self.BLOR_reputation_factors = np.zeros((4, self.oracleNum))

    def workload(self, request_count):
        arrival_time = self.arrival_Times[request_count - 1]
        length = self.lengths[request_count - 1]
        requestType = self.request_type[request_count - 1]
        ddl = self.ddl
        if request_count == self.requestNum:
            finish = True
        else:
            finish = False
        request_attributes = [request_count - 1, arrival_time, length, requestType, ddl]
        return finish, request_attributes

    def feedback(self, request_attrs, action, policyID):
        request_id = request_attrs[0]
        arrival_time = request_attrs[1]
        length = request_attrs[2]
        request_type = request_attrs[3]
        ddl = request_attrs[4]
        acc = self.oracleAcc[action]

        cost = self.oracleCost[action]
        type = self.oracleTypes[action]
        validation_probs = self.oracleValidationProbs[action]
        behavior_probs = self.oracleBehaviorProbs[action]


        if policyID == 1:
            idleT = self.RAN_oracle_events[0, action]
            reputation = self.RAN_oracle_events[2, action]
        elif policyID == 2:
            idleT = self.RR_oracle_events[0, action]
            reputation = self.RR_oracle_events[2, action]
        elif policyID == 3:
            idleT = self.early_oracle_events[0, action]
            reputation = self.early_oracle_events[2, action]
        elif policyID == 4:
            idleT = self.DQN_oracle_events[0, action]
            reputation = self.DQN_oracle_events[2, action]
        # elif policyID == 5:
        #     idleT = self.FWA_oracle_events[0, action]
        #     reputation = self.FWA_oracle_events[2, action]
        elif policyID == 6:
            idleT = self.BLOR_oracle_events[0, action]
            reputation = self.BLOR_oracle_events[2, action]
        elif policyID == 7:
            idleT = self.PSG_oracle_events[0, action]
            reputation = self.PSG_oracle_events[2, action]

        exeT = length / acc

        # waitT & start exeT
        if idleT <= arrival_time:  # if no waitT
            waitT = 0
            startT = arrival_time
        else:
            waitT = idleT - arrival_time
            startT = idleT
        # malicious oracle affect exeT
        if action in [0, 5, 10]:
            exe_time = (length * 1.05) / acc
        else:
            exe_time = exeT

        # Probability of introducing noise, representing the probability that the oracle does not behave as expected
        noise_probability = 0

        # Generate a random number to decide whether to introduce noise
        if np.random.rand() < noise_probability:
        # In the case of noise, add additional response time (noise)
            real_exeT = exe_time + 1
        else:
            real_exeT = exe_time

        durationT = waitT + real_exeT
        leaveT = startT + real_exeT  # leave T
        new_idleT = leaveT  # update oracle idle time

        penalty = 0 if request_type == type else 1
        reward = (1 + 2.5 * np.exp(1.5 - cost)) * (exeT / durationT) + reputation - 4 * penalty
        match = 1 if request_type == type else 0

        # whether success
        success = 1 if durationT <= ddl and request_type == type else 0

        # whether success without type factor
        success_without_type = 1 if durationT <= ddl else 0
        
        successful_validation = 1 if np.random.rand() < validation_probs and durationT <= ddl else 0

        behavior_record = np.random.choice([0, 1, 5, 100], p=behavior_probs)

        if policyID == 1:
            self.RAN_events[0, request_id] = action
            self.RAN_events[1, request_id] = startT
            self.RAN_events[2, request_id] = waitT
            self.RAN_events[3, request_id] = durationT
            self.RAN_events[4, request_id] = leaveT
            self.RAN_events[5, request_id] = reward
            self.RAN_events[6, request_id] = exeT
            self.RAN_events[7, request_id] = success
            self.RAN_events[8, request_id] = cost
            self.RAN_events[9, request_id] = success_without_type
            
            # update oracle info
            self.RAN_oracle_events[1, action] += 1
            self.RAN_oracle_events[0, action] = new_idleT
            self.RAN_oracle_events[3, action] += match
            self.RAN_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.RAN_reputation_factors[0, action] += 1
            self.RAN_reputation_factors[1, action] += successful_validation
            self.RAN_reputation_factors[2, action] += durationT
            self.RAN_reputation_factors[3, action] += behavior_record

        elif policyID == 2:
            self.RR_events[0, request_id] = action
            self.RR_events[1, request_id] = startT
            self.RR_events[2, request_id] = waitT
            self.RR_events[3, request_id] = durationT
            self.RR_events[4, request_id] = leaveT
            self.RR_events[5, request_id] = reward
            self.RR_events[6, request_id] = exeT
            self.RR_events[7, request_id] = success
            self.RR_events[8, request_id] = cost
            self.RR_events[9, request_id] = success_without_type

            # update oracle info
            self.RR_oracle_events[1, action] += 1
            self.RR_oracle_events[0, action] = new_idleT
            self.RR_oracle_events[3, action] += match
            self.RR_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.RR_reputation_factors[0, action] += 1
            self.RR_reputation_factors[1, action] += successful_validation
            self.RR_reputation_factors[2, action] += durationT
            self.RR_reputation_factors[3, action] += behavior_record

        elif policyID == 3:
            self.early_events[0, request_id] = action
            self.early_events[1, request_id] = startT
            self.early_events[2, request_id] = waitT
            self.early_events[3, request_id] = durationT
            self.early_events[4, request_id] = leaveT
            self.early_events[5, request_id] = reward
            self.early_events[6, request_id] = exeT
            self.early_events[7, request_id] = success
            self.early_events[8, request_id] = cost
            self.early_events[9, request_id] = success_without_type
            
            # update oracle info
            self.early_oracle_events[1, action] += 1
            self.early_oracle_events[0, action] = new_idleT
            self.early_oracle_events[3, action] += match
            self.early_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.early_reputation_factors[0, action] += 1
            self.early_reputation_factors[1, action] += successful_validation
            self.early_reputation_factors[2, action] += durationT
            self.early_reputation_factors[3, action] += behavior_record

        elif policyID == 4:
            self.DQN_events[0, request_id] = action
            self.DQN_events[1, request_id] = startT
            self.DQN_events[2, request_id] = waitT
            self.DQN_events[3, request_id] = durationT
            self.DQN_events[4, request_id] = leaveT
            self.DQN_events[5, request_id] = reward
            self.DQN_events[6, request_id] = exeT
            self.DQN_events[7, request_id] = success
            self.DQN_events[8, request_id] = cost
            self.DQN_events[9, request_id] = success_without_type

            # update oracle info
            self.DQN_oracle_events[1, action] += 1
            self.DQN_oracle_events[0, action] = new_idleT
            self.DQN_oracle_events[3, action] += match
            self.DQN_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.DQN_reputation_factors[0, action] += 1
            self.DQN_reputation_factors[1, action] += successful_validation
            self.DQN_reputation_factors[2, action] += durationT
            self.DQN_reputation_factors[3, action] += behavior_record

        elif policyID == 6:
            self.BLOR_events[0, request_id] = action
            self.BLOR_events[1, request_id] = startT
            self.BLOR_events[2, request_id] = waitT
            self.BLOR_events[3, request_id] = durationT
            self.BLOR_events[4, request_id] = leaveT
            self.BLOR_events[5, request_id] = reward
            self.BLOR_events[6, request_id] = exeT
            self.BLOR_events[7, request_id] = success
            self.BLOR_events[8, request_id] = cost
            self.BLOR_events[9, request_id] = success_without_type

            # update oracle info
            self.BLOR_oracle_events[1, action] += 1
            self.BLOR_oracle_events[0, action] = new_idleT
            self.BLOR_oracle_events[3, action] += match
            self.BLOR_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.BLOR_reputation_factors[0, action] += 1
            self.BLOR_reputation_factors[1, action] += successful_validation
            self.BLOR_reputation_factors[2, action] += durationT
            self.BLOR_reputation_factors[3, action] += behavior_record

        elif policyID == 7:
            self.PSG_events[0, request_id] = action
            self.PSG_events[1, request_id] = startT
            self.PSG_events[2, request_id] = waitT
            self.PSG_events[3, request_id] = durationT
            self.PSG_events[4, request_id] = leaveT
            self.PSG_events[5, request_id] = reward
            self.PSG_events[6, request_id] = exeT
            self.PSG_events[7, request_id] = success
            self.PSG_events[8, request_id] = cost
            self.PSG_events[9, request_id] = success_without_type

            # update oracle info
            self.PSG_oracle_events[1, action] += 1
            self.PSG_oracle_events[0, action] = new_idleT
            self.PSG_oracle_events[3, action] += match
            self.PSG_oracle_events[4, action] += successful_validation

            # update reputation factors
            self.PSG_reputation_factors[0, action] += 1
            self.PSG_reputation_factors[1, action] += successful_validation
            self.PSG_reputation_factors[2, action] += durationT
            self.PSG_reputation_factors[3, action] += behavior_record


        return reward

    def feedback_PSG_FWA(self, request_attrs, policyID):
        arrival_time = request_attrs[1]
        length = request_attrs[2]
        request_type = request_attrs[3]
        acc = self.oracleAcc
        cost = self.oracleCost
        type = self.oracleTypes

        rewards = np.zeros(self.oracleNum)
        # if policyID == 5:
        #     idleT = self.FWA_oracle_events[0]
        #     reputation = self.FWA_oracle_events[2]
        if policyID == 7:
            idleT = self.PSG_oracle_events[0]
            reputation = self.PSG_oracle_events[2]

        # calculate rewards of all oracles
        for action in range(self.oracleNum):
            # waitT & start exeT
            if idleT[action] <= arrival_time:  # if no waitT
                waitT = 0
            else:
                waitT = idleT[action] - arrival_time

            exeT = length / acc[action]

            # malicious oracle affect exeT
            if action in [0, 5, 10]:
                exe_time = (length * 1.05) / acc[action]
            else:
                exe_time = exeT

            durationT = waitT + exe_time  # waitT+exeT
            # reward
            penalty = 0 if request_type == type[action] else 1
            rewards[action] = (1 + 2.5*np.exp(1.5 - cost[action])) * (exeT / durationT) + reputation[action] - 4 * penalty
        return rewards, cost


    def get_reputation_factors(self, policyID):
        total_requests = np.zeros(self.oracleNum)
        successful_validation_requests = np.zeros(self.oracleNum)
        total_response_time = np.zeros(self.oracleNum)
        behavior_records = np.zeros(self.oracleNum)

        success_rate = np.zeros(self.oracleNum)
        average_response_time = np.zeros(self.oracleNum)
        response_time_score = np.zeros(self.oracleNum)

        if policyID == 1:
            total_requests = self.RAN_reputation_factors[0]
            successful_validation_requests = self.RAN_reputation_factors[1]
            total_response_time = self.RAN_reputation_factors[2]
            behavior_records = self.RAN_reputation_factors[3]

        elif policyID == 2:
            total_requests = self.RR_reputation_factors[0]
            successful_validation_requests = self.RR_reputation_factors[1]
            total_response_time = self.RR_reputation_factors[2]
            behavior_records = self.RR_reputation_factors[3]

        elif policyID == 3:
            total_requests = self.early_reputation_factors[0]
            successful_validation_requests = self.early_reputation_factors[1]
            total_response_time = self.early_reputation_factors[2]
            behavior_records = self.early_reputation_factors[3]

        elif policyID == 4:
            total_requests = self.DQN_reputation_factors[0]
            successful_validation_requests = self.DQN_reputation_factors[1]
            total_response_time = self.DQN_reputation_factors[2]
            behavior_records = self.DQN_reputation_factors[3]

        # elif policyID == 5:
        #     total_requests = self.FWA_reputation_factors[0]
        #     successful_validation_requests = self.FWA_reputation_factors[1]
        #     total_response_time = self.FWA_reputation_factors[2]
        #     behavior_records = self.FWA_reputation_factors[3]
        #     # idle_time = self.FWA_oracle_events[0]
        elif policyID == 6:
            total_requests = self.BLOR_reputation_factors[0]
            successful_validation_requests = self.BLOR_reputation_factors[1]
            total_response_time = self.BLOR_reputation_factors[2]
            behavior_records = self.BLOR_reputation_factors[3]

        elif policyID == 7:
            total_requests = self.PSG_reputation_factors[0]
            successful_validation_requests = self.PSG_reputation_factors[1]
            total_response_time = self.PSG_reputation_factors[2]
            behavior_records = self.PSG_reputation_factors[3]

        # calculate reputation factors scores
        # calculate reliability score
        average_requests = self.timeperiodSize / self.oracleNum
        relative_response_frequency = total_requests / average_requests if average_requests > 0 else 0
        for i in range(self.oracleNum):
            success_rate[i] = successful_validation_requests[i] / total_requests[i] if total_requests[i] > 0 else 0
            average_response_time[i] = total_response_time[i] / total_requests[i] if total_requests[i] > 0 else 0
            response_time_score[i] = self.ddl / average_response_time[i] if average_response_time[i] > 0 else 0
        reliability_score = (relative_response_frequency * 0.2) + (success_rate * 0.4) + (
                response_time_score * 0.4)
        # calculate behavior score
        behavior_score = behavior_records
        # calculate staked tokens score
        staked_tokens_score = np.array(self.oracleToken) / (
             int(sum(self.oracleToken[o] for o in range(self.oracleNum)) / self.oracleNum))

        reputation_attributes = [reliability_score, behavior_score, staked_tokens_score]
        return reputation_attributes

    def initial_reputation(self):
        # Random
        self.RAN_oracle_reputation_history[0] = self.oracleInitialReputation
        self.RAN_reputation_timewindow = np.append(self.RAN_reputation_timewindow,
                                                   [self.RAN_oracle_reputation_history[0]], axis=0)
        self.RAN_oracle_events[2] = self.oracleInitialReputation
        # Round robin
        self.RR_oracle_reputation_history[0] = self.oracleInitialReputation
        self.RR_reputation_timewindow = np.append(self.RR_reputation_timewindow,
                                                  [self.RR_oracle_reputation_history[0]], axis=0)
        self.RR_oracle_events[2] = self.oracleInitialReputation
        # Earliest
        self.early_oracle_reputation_history[0] = self.oracleInitialReputation
        self.early_reputation_timewindow = np.append(self.early_reputation_timewindow,
                                                     [self.early_oracle_reputation_history[0]], axis=0)
        self.early_oracle_events[2] = self.oracleInitialReputation
        # DQN
        self.DQN_oracle_reputation_history[0] = self.oracleInitialReputation
        self.DQN_reputation_timewindow = np.append(self.DQN_reputation_timewindow,
                                                   [self.DQN_oracle_reputation_history[0]], axis=0)
        self.DQN_oracle_events[2] = self.oracleInitialReputation
        # # Firework
        # self.FWA_oracle_reputation_history[0] = self.oracleInitialReputation
        # self.FWA_reputation_timewindow = np.append(self.FWA_reputation_timewindow,
        #                                            [self.FWA_oracle_reputation_history[0]], axis=0)
        # self.FWA_oracle_events[2] = self.oracleInitialReputation
        # BLOR
        self.BLOR_oracle_reputation_history[0] = self.oracleInitialReputation
        self.BLOR_reputation_timewindow = np.append(self.BLOR_reputation_timewindow,
                                                    [self.BLOR_oracle_reputation_history[0]], axis=0)
        self.BLOR_oracle_events[2] = self.oracleInitialReputation
        # SemiGreedy
        self.PSG_oracle_reputation_history[0] = self.oracleInitialReputation
        self.PSG_reputation_timewindow = np.append(self.PSG_reputation_timewindow,
                                                   [self.PSG_oracle_reputation_history[0]], axis=0)
        self.PSG_oracle_events[2] = self.oracleInitialReputation



    def update_reputation(self, reputation_attributes, current_period, policyID):
        reliability_score = reputation_attributes[0]
        behavior_score = reputation_attributes[1]
        staked_tokens_score = reputation_attributes[2]
        reputation = np.zeros(self.oracleNum)



        # calculate oracle base reputation
        base_reputation = (reliability_score * 0.4) - (behavior_score * 0.4) + (staked_tokens_score * 0.2)

        if current_period > 1 and current_period <= self.timeperiodNum:
            if policyID == 1:
                for idx in range(0, self.RAN_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation += time_factor * self.RAN_reputation_timewindow[idx]  # (idx+1)th time period reputation

                reputation = reputation + base_reputation * np.tanh(0.6 / 1)

                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.RAN_oracle_reputation_history[current_period-1] = reputation
                self.RAN_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.RAN_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.RAN_reputation_timewindow = np.vstack((new_reputation, self.RAN_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.RAN_reputation_timewindow = np.vstack((new_reputation, self.RAN_reputation_timewindow[:-1, :]))

            elif policyID == 2:
                for idx in range(0, self.RR_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation_idx = time_factor * self.RR_reputation_timewindow[idx]  # (idx+1)th time period reputation
                    reputation += reputation_idx
                reputation += base_reputation * np.tanh(0.6 / 1)
                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.RR_oracle_reputation_history[current_period - 1] = reputation
                self.RR_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.RR_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.RR_reputation_timewindow = np.vstack((new_reputation, self.RR_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.RR_reputation_timewindow = np.vstack((new_reputation, self.RR_reputation_timewindow[:-1, :]))
            elif policyID == 3:
                for idx in range(0, self.early_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation_idx = time_factor * self.early_reputation_timewindow[idx]  # (idx+1)th time period reputation
                    reputation += reputation_idx
                reputation += base_reputation * np.tanh(0.6 / 1)
                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.early_oracle_reputation_history[current_period - 1] = reputation
                self.early_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.early_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.early_reputation_timewindow = np.vstack((new_reputation, self.early_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.early_reputation_timewindow = np.vstack((new_reputation, self.early_reputation_timewindow[:-1, :]))
            elif policyID == 4:
                for idx in range(0, self.DQN_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation_idx = time_factor * self.DQN_reputation_timewindow[idx]  # (idx+1)th time period reputation
                    reputation += reputation_idx
                reputation += base_reputation * np.tanh(0.6 / 1)
                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.DQN_oracle_reputation_history[current_period - 1] = reputation
                self.DQN_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.DQN_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.DQN_reputation_timewindow = np.vstack((new_reputation, self.DQN_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.DQN_reputation_timewindow = np.vstack((new_reputation, self.DQN_reputation_timewindow[:-1, :]))
            # elif policyID == 5:
            #     for idx in range(0, self.FWA_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
            #         k = idx + 2  # calculate time factor from 1/2
            #         time_factor = np.tanh(0.6 / k)
            #         reputation_idx = time_factor * self.FWA_reputation_timewindow[idx]  # (idx+1)th time period reputation
            #         reputation += reputation_idx
            #     reputation += base_reputation * np.tanh(0.6 / 1)
            #     if len(reputation) != self.oracleNum:
            #         raise ValueError("Reputation values must match the number of oracles")
            #     self.FWA_oracle_reputation_history[current_period - 1] = reputation
            #     self.FWA_oracle_events[2] = reputation
            #
            #     #  newly added reputation value is added as a new row
            #     new_reputation = np.array(reputation).reshape(1, -1)
            #     #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
            #     if self.FWA_reputation_timewindow.shape[0] < self.timewindowSize:
            #         self.FWA_reputation_timewindow = np.vstack((new_reputation, self.FWA_reputation_timewindow))
            #     #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
            #     else:
            #         self.FWA_reputation_timewindow = np.vstack((new_reputation, self.FWA_reputation_timewindow[:-1, :]))
            elif policyID == 6:
                for idx in range(0, self.BLOR_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation_idx = time_factor * self.BLOR_reputation_timewindow[
                        idx]  # (idx+1)th time period reputation
                    reputation += reputation_idx
                reputation += base_reputation * np.tanh(0.6 / 1)
                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.BLOR_oracle_reputation_history[current_period - 1] = reputation
                self.BLOR_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.BLOR_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.BLOR_reputation_timewindow = np.vstack((new_reputation, self.BLOR_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.BLOR_reputation_timewindow = np.vstack((new_reputation, self.BLOR_reputation_timewindow[:-1, :]))
            elif policyID == 7:
                for idx in range(0, self.PSG_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
                    k = idx + 2  # calculate time factor from 1/2
                    time_factor = np.tanh(0.6 / k)
                    reputation_idx = time_factor * self.PSG_reputation_timewindow[
                        idx]  # (idx+1)th time period reputation
                    reputation += reputation_idx
                reputation += base_reputation * np.tanh(0.6 / 1)
                if len(reputation) != self.oracleNum:
                    raise ValueError("Reputation values must match the number of oracles")
                self.PSG_oracle_reputation_history[current_period - 1] = reputation
                self.PSG_oracle_events[2] = reputation

                #  newly added reputation value is added as a new row
                new_reputation = np.array(reputation).reshape(1, -1)
                #  if the column length of reputation_timewindow < timewindowSize: the new reputation added as a new row to the reputation_timewindow
                if self.PSG_reputation_timewindow.shape[0] < self.timewindowSize:
                    self.PSG_reputation_timewindow = np.vstack((new_reputation, self.PSG_reputation_timewindow))
                #  if the column length of reputation_timewindow >= timewindowSize: delate oldest reputation and then add the new reputation as a new row to the reputation_timewindow
                else:
                    self.PSG_reputation_timewindow = np.vstack((new_reputation, self.PSG_reputation_timewindow[:-1, :]))
        else:
            raise ValueError("Not within the time period that needs to update reputation")


    def get_oracle_idleT(self, policyID):
        if policyID == 1:
            idleTimes = self.RAN_oracle_events[0, :]
        elif policyID == 2:
            idleTimes = self.RR_oracle_events[0, :]
        elif policyID == 3:
            idleTimes = self.early_oracle_events[0, :]
        elif policyID == 4:
            idleTimes = self.DQN_oracle_events[0, :]
        # elif policyID == 5:
        #     idleTimes = self.FWA_oracle_events[0, :]
        elif policyID == 6:
            idleTimes = self.BLOR_oracle_events[0, :]
        elif policyID == 7:
            idleTimes = self.PSG_oracle_events[0, :]
        
        return idleTimes

    def get_oracle_reputation(self, policyID):
        if policyID == 1:
            reputations = self.RAN_oracle_events[2, :]
        elif policyID == 2:
            reputations = self.RR_oracle_events[2, :]
        elif policyID == 3:
            reputations = self.early_oracle_events[2, :]
        elif policyID == 4:
            reputations = self.DQN_oracle_events[2, :]
        # elif policyID == 5:
        #     reputations = self.FWA_oracle_events[2, :]
        elif policyID == 6:
            reputations = self.BLOR_oracle_events[2, :]
        elif policyID == 7:
            reputations = self.PSG_oracle_events[2, :]

        return reputations

    def get_successful_validation(self, policyID):
        if policyID == 1:
            successful_validation = self.RAN_reputation_factors[1, :]
        elif policyID == 2:
            successful_validation = self.RR_reputation_factors[1, :]
        elif policyID == 3:
            successful_validation = self.early_reputation_factors[1, :]
        elif policyID == 4:
            successful_validation = self.DQN_reputation_factors[1, :]
        # elif policyID == 5:
        #     successful_validation = self.FWA_reputation_factors[1, :]
        elif policyID == 6:
            successful_validation = self.BLOR_reputation_factors[1, :]
        elif policyID == 7:
            successful_validation = self.PSG_reputation_factors[1, :]

        return np.array(successful_validation)

    def get_request_num(self, policyID):
        if policyID == 1:
            request_num = self.RAN_reputation_factors[0, :]
        elif policyID == 2:
            request_num = self.RR_reputation_factors[0, :]
        elif policyID == 3:
            request_num = self.early_reputation_factors[0, :]
        elif policyID == 4:
            request_num = self.DQN_reputation_factors[0, :]
        # elif policyID == 5:
        #     request_num = self.FWA_reputation_factors[0, :]
        elif policyID == 6:
            request_num = self.BLOR_reputation_factors[0, :]
        elif policyID == 7:
            request_num = self.PSG_reputation_factors[0, :]

        return np.array(request_num)


    def getState(self, request_attrs, policyID):
        arrivalT = request_attrs[1]
        # length = request_attrs[2]
        request_service = request_attrs[3]
        state_request = [request_service]

        if policyID == 1:  # random
            idleTimes = self.get_oracle_idleT(1)
            reputations = self.get_oracle_reputation(1)
        elif policyID == 2:  # RR
            idleTimes = self.get_oracle_idleT(2)
            reputations = self.get_oracle_reputation(2)
        elif policyID == 3:  # early
            idleTimes = self.get_oracle_idleT(3)
            reputations = self.get_oracle_reputation(3)
        elif policyID == 4:  # DQN
            idleTimes = self.get_oracle_idleT(4)
            reputations = self.get_oracle_reputation(4)
        # elif policyID == 5:  # FWA
        #     idleTimes = self.get_oracle_idleT(5)
        #     reputations = self.get_oracle_reputation(5)
        elif policyID == 6:  # BLOR
            idleTimes = self.get_oracle_idleT(6)
            reputations = self.get_oracle_reputation(6)
        elif policyID == 7:  # PSG
            idleTimes = self.get_oracle_idleT(7)
            reputations = self.get_oracle_reputation(7)
        waitTimes = [t - arrivalT for t in idleTimes]
        waitTimes = np.maximum(waitTimes, 0)
        state = np.hstack((state_request, waitTimes, reputations))
        return state

    def getStateP(self, request_id):
        duration = self.BLOR_events[3, request_id]
        return duration

    def get_accumulateRewards(self, policies, start, end):

        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:end])
        rewards[1] = sum(self.RR_events[5, start:end])
        rewards[2] = sum(self.early_events[5, start:end])
        rewards[3] = sum(self.DQN_events[5, start:end])
        rewards[4] = sum(self.BLOR_events[5, start:end])
        rewards[5] = sum(self.PSG_events[5, start:end])
        return np.around(rewards, 2)

    def get_accumulateCost(self, policies, start, end):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:end])
        Cost[1] = sum(self.RR_events[8, start:end])
        Cost[2] = sum(self.early_events[8, start:end])
        Cost[3] = sum(self.DQN_events[8, start:end])
        Cost[4] = sum(self.BLOR_events[8, start:end])
        Cost[5] = sum(self.PSG_events[8, start:end])
        return np.around(Cost, 2)

    def get_FinishTimes(self, policies, start, end):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, start:end])
        finishT[1] = max(self.RR_events[4, start:end])
        finishT[2] = max(self.early_events[4, start:end])
        finishT[3] = max(self.DQN_events[4, start:end])
        finishT[4] = max(self.BLOR_events[4, start:end])
        finishT[5] = max(self.PSG_events[4, start:end])
        return np.around(finishT, 2)

    def get_executeTs(self, policies, start, end):
        executeTs = np.zeros(policies)
        executeTs[0] = np.mean(self.RAN_events[6, start:end])
        executeTs[1] = np.mean(self.RR_events[6, start:end])
        executeTs[2] = np.mean(self.early_events[6, start:end])
        executeTs[3] = np.mean(self.DQN_events[6, start:end])
        executeTs[4] = np.mean(self.BLOR_events[6, start:end])
        executeTs[5] = np.mean(self.PSG_events[6, start:end])
        return np.around(executeTs, 3)

    def get_waitTs(self, policies, start, end):
        waitTs = np.zeros(policies)
        waitTs[0] = np.mean(self.RAN_events[2, start:end])
        waitTs[1] = np.mean(self.RR_events[2, start:end])
        waitTs[2] = np.mean(self.early_events[2, start:end])
        waitTs[3] = np.mean(self.DQN_events[2, start:end])
        waitTs[4] = np.mean(self.BLOR_events[2, start:end])
        waitTs[5] = np.mean(self.PSG_events[2, start:end])
        return np.around(waitTs, 3)

    def get_responseTs(self, policies, start, end):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:end])
        respTs[1] = np.mean(self.RR_events[3, start:end])
        respTs[2] = np.mean(self.early_events[3, start:end])
        respTs[3] = np.mean(self.DQN_events[3, start:end])
        respTs[4] = np.mean(self.BLOR_events[3, start:end])
        respTs[5] = np.mean(self.PSG_events[3, start:end])
        return np.around(respTs, 3)

    def get_successTimes(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[7, start:end]) / (end - start)
        successT[1] = sum(self.RR_events[7, start:end]) / (end - start)
        successT[2] = sum(self.early_events[7, start:end]) / (end - start)
        successT[3] = sum(self.DQN_events[7, start:end]) / (end - start)
        successT[4] = sum(self.BLOR_events[7, start:end]) / (end - start)
        successT[5] = sum(self.PSG_events[7, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_successInTime(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[9, start:end]) / (end - start)
        successT[1] = sum(self.RR_events[9, start:end]) / (end - start)
        successT[2] = sum(self.early_events[9, start:end]) / (end - start)
        successT[3] = sum(self.DQN_events[9, start:end]) / (end - start)
        successT[4] = sum(self.BLOR_events[9, start:end]) / (end - start)
        successT[5] = sum(self.PSG_events[9, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_rejectTimes(self, policies, start, end):
        reject = np.zeros(policies)
        reject[0] = sum(self.RAN_events[8, start:end])
        reject[1] = sum(self.RR_events[8, start:end])
        reject[2] = sum(self.early_events[8, start:end])
        reject[3] = sum(self.DQN_events[8, start:end])
        reject[4] = sum(self.BLOR_events[8, start:end])
        reject[5] = sum(self.PSG_events[8, start:end])
        return np.around(reject, 2)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.RAN_events[5, start:self.requestNum])
        rewards[1] = sum(self.RR_events[5, start:self.requestNum])
        rewards[2] = sum(self.early_events[5, start:self.requestNum])
        rewards[3] = sum(self.DQN_events[5, start:self.requestNum])
        rewards[4] = sum(self.BLOR_events[5, start:self.requestNum])
        rewards[5] = sum(self.PSG_events[5, start:self.requestNum])
        return np.around(rewards, 2)

    def get_totalMaliciousNum(self, policies):
        num = np.zeros(policies)
        maliciousOracleIndex = [0, 5, 10]
        num[0] = np.sum(self.RAN_oracle_events[1, maliciousOracleIndex])
        num[1] = np.sum(self.RR_oracle_events[1, maliciousOracleIndex])
        num[2] = np.sum(self.early_oracle_events[1, maliciousOracleIndex])
        num[3] = np.sum(self.DQN_oracle_events[1, maliciousOracleIndex])
        num[4] = np.sum(self.BLOR_oracle_events[1, maliciousOracleIndex])
        num[5] = np.sum(self.PSG_oracle_events[1, maliciousOracleIndex])
        return np.around(num, 1)

    def get_totalNormalNum(self, policies):
        num = np.zeros(policies)
        # normalOracleIndex = [2, 7, 12]
        normalOracleIndex = [2, 7, 12]
        num[0] = np.sum(self.RAN_oracle_events[1, normalOracleIndex])
        num[1] = np.sum(self.RR_oracle_events[1, normalOracleIndex])
        num[2] = np.sum(self.early_oracle_events[1, normalOracleIndex])
        num[3] = np.sum(self.DQN_oracle_events[1, normalOracleIndex])
        num[4] = np.sum(self.BLOR_oracle_events[1, normalOracleIndex])
        num[5] = np.sum(self.PSG_oracle_events[1, normalOracleIndex])
        return np.around(num, 1)

    def get_totalTrustedNum(self, policies):
        num = np.zeros(policies)
        trustedOracleIndex = [1, 3, 4, 6, 8, 9, 11, 13, 14]
        num[0] = np.sum(self.RAN_oracle_events[1, trustedOracleIndex])
        num[1] = np.sum(self.RR_oracle_events[1, trustedOracleIndex])
        num[2] = np.sum(self.early_oracle_events[1, trustedOracleIndex])
        num[3] = np.sum(self.DQN_oracle_events[1, trustedOracleIndex])
        num[4] = np.sum(self.BLOR_oracle_events[1, trustedOracleIndex])
        num[5] = np.sum(self.PSG_oracle_events[1, trustedOracleIndex])
        return np.around(num, 1)


    def get_totalMatchRate(self, policies):
        matchRate = np.zeros(policies)
        matchRate[0] = np.sum(self.RAN_oracle_events[3, :]) / self.requestNum
        matchRate[1] = np.sum(self.RR_oracle_events[3, :]) / self.requestNum
        matchRate[2] = np.sum(self.early_oracle_events[3, :]) / self.requestNum
        matchRate[3] = np.sum(self.DQN_oracle_events[3, :]) / self.requestNum
        matchRate[4] = np.sum(self.BLOR_oracle_events[3, :]) / self.requestNum
        matchRate[5] = np.sum(self.PSG_oracle_events[3, :]) / self.requestNum
        return np.around(matchRate, 3)

    def get_totalTimes(self, policies, start):
        finishT = np.zeros(policies)
        finishT[0] = max(self.RAN_events[4, :]) - self.arrival_Times[start]
        finishT[1] = max(self.RR_events[4, :]) - self.arrival_Times[start]
        finishT[2] = max(self.early_events[4, :]) - self.arrival_Times[start]
        finishT[3] = max(self.DQN_events[4, :]) - self.arrival_Times[start]
        finishT[4] = max(self.BLOR_events[4, :]) - self.arrival_Times[start]
        finishT[5] = max(self.PSG_events[4, :]) - self.arrival_Times[start]
        return np.around(finishT, 2)



    def get_all_responseTs(self, policies):
        respTs = np.zeros((policies, self.requestNum))
        respTs[0, :] = self.RAN_events[3, :]
        respTs[1, :] = self.RR_events[3, :]
        respTs[2, :] = self.early_events[3, :]
        respTs[3, :] = self.DQN_events[3, :]
        respTs[4, :] = self.BLOR_events[3, :]
        respTs[5, :] = self.PSG_events[3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.RAN_events[3, start:self.requestNum])
        respTs[1] = np.mean(self.RR_events[3, start:self.requestNum])
        respTs[2] = np.mean(self.early_events[3, start:self.requestNum])
        respTs[3] = np.mean(self.DQN_events[3, start:self.requestNum])
        respTs[4] = np.mean(self.BLOR_events[3, start:self.requestNum])
        respTs[5] = np.mean(self.PSG_events[3, start:self.requestNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[1] = sum(self.RR_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[2] = sum(self.early_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[3] = sum(self.DQN_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[4] = sum(self.BLOR_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[5] = sum(self.PSG_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(successT, 3)

    def get_totalSuccessInTime(self, policies, start):
        successT = np.zeros(policies)
        successT[0] = sum(self.RAN_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[1] = sum(self.RR_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[2] = sum(self.early_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[3] = sum(self.DQN_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[4] = sum(self.BLOR_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        successT[5] = sum(self.PSG_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(successT, 3)

    def get_totalCost(self, policies, start):

        Cost = np.zeros(policies)
        Cost[0] = sum(self.RAN_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        Cost[1] = sum(self.RR_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        Cost[2] = sum(self.early_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        Cost[3] = sum(self.DQN_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        Cost[4] = sum(self.BLOR_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        Cost[5] = sum(self.PSG_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(Cost, 3)