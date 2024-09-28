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
        self.DQN_reputation_factors = np.zeros((4, self.oracleNum))
        # TimeWindow Setting
        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(args.Request_Num / args.Time_Period_Size) + 1
        # generate workload
        self.gen_workload(self.lamda)


        # request: 1-oracle id  2-start time  3-wait time  4-waitT+exeT  5-leave time  6-reward  7-actual_exeT
        #          8- success   9-reject    10- success without type factor

        # oracle: 1-idleT   2-request num   3-reputation   4-match num   5-successful validation num
        # DQN
        self.DQN_events = np.zeros((10, self.requestNum))
        self.DQN_oracle_events = np.zeros((5, self.oracleNum))
        self.DQN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.DQN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.DQN_reputation_timewindow = np.zeros((0, self.oracleNum))
        
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
        self.ddl = args.Request_ddl  # 7s = waitT + exeT
        self.gen_workload(args.lamda)
        self.oracleTypes = args.Oracle_Type

        self.timewindowSize = args.Time_Window_Size
        self.timeperiodSize = args.Time_Period_Size
        self.timeperiodNum = int(args.Request_Num / args.Time_Period_Size) + 1
        # reset all records
        # DQN
        self.DQN_events = np.zeros((10, self.requestNum))
        self.DQN_oracle_events = np.zeros((5, self.oracleNum))
        self.DQN_oracle_events[2] = args.Oracle_Initial_Reputation
        self.DQN_oracle_reputation_history = np.zeros((self.timeperiodNum, self.oracleNum))
        self.DQN_reputation_timewindow = np.zeros((0, self.oracleNum))

    def reset_reputation_factors(self):
        self.DQN_reputation_factors = np.zeros((4, self.oracleNum))

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

    def feedback(self, request_attrs, action):
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

        idleT = self.DQN_oracle_events[0, action]
        reputation = self.DQN_oracle_events[2, action]
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

        # update event info
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

        return reward


    def get_reputation_factors(self):
        success_rate = np.zeros(self.oracleNum)
        average_response_time = np.zeros(self.oracleNum)
        response_time_score = np.zeros(self.oracleNum)

        total_requests = self.DQN_reputation_factors[0]
        successful_validation_requests = self.DQN_reputation_factors[1]
        total_response_time = self.DQN_reputation_factors[2]
        behavior_records = self.DQN_reputation_factors[3]

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
        # DQN
        self.DQN_oracle_reputation_history[0] = self.oracleInitialReputation
        self.DQN_reputation_timewindow = np.append(self.DQN_reputation_timewindow,
                                                   [self.DQN_oracle_reputation_history[0]], axis=0)
        self.DQN_oracle_events[2] = self.oracleInitialReputation

    def update_reputation(self, reputation_attributes, current_period):
        reliability_score = reputation_attributes[0]
        behavior_score = reputation_attributes[1]
        staked_tokens_score = reputation_attributes[2]
        reputation = np.zeros(self.oracleNum)



        # calculate oracle base reputation
        base_reputation = (reliability_score * 0.4) - (behavior_score * 0.4) + (staked_tokens_score * 0.2)

        if current_period > 1 and current_period <= self.timeperiodNum:
            for idx in range(0,
                             self.DQN_reputation_timewindow.shape[0]):  # calculate reputation from latest time period
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
        else:
            raise ValueError("Not within the time period that needs to update reputation")


    def get_oracle_idleT(self):
        idleTimes = self.DQN_oracle_events[0, :]
        return idleTimes

    def get_oracle_reputation(self):
        reputations = self.DQN_oracle_events[2, :]
        return reputations

    def get_successful_validation(self):
        successful_validation = self.DQN_reputation_factors[1, :]
        return np.array(successful_validation)

    def get_request_num(self):
        request_num = self.DQN_reputation_factors[0, :]
        return np.array(request_num)


    def getState(self, request_attrs):
        arrivalT = request_attrs[1]
        request_type = request_attrs[3]
        state_request = [request_type]
        # DQN
        idleTimes = self.get_oracle_idleT()
        reputations = self.get_oracle_reputation()

        waitTimes = [t - arrivalT for t in idleTimes]
        waitTimes = np.maximum(waitTimes, 0)
        state = np.hstack((state_request, waitTimes, reputations))
        return state

    def getStateP(self, request_id):
        duration = self.BLOR_events[3, request_id]
        return duration

    def get_accumulateRewards(self, policies, start, end):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.DQN_events[5, start:end])
        return np.around(rewards, 2)

    def get_accumulateCost(self, policies, start, end):
        Cost = np.zeros(policies)
        Cost[0] = sum(self.DQN_events[8, start:end])
        return np.around(Cost, 2)

    def get_FinishTimes(self, policies, start, end):
        finishT = np.zeros(policies)
        finishT[0] = max(self.DQN_events[4, start:end])
        return np.around(finishT, 2)

    def get_executeTs(self, policies, start, end):
        executeTs = np.zeros(policies)
        executeTs[0] = np.mean(self.DQN_events[6, start:end])
        return np.around(executeTs, 3)

    def get_waitTs(self, policies, start, end):
        waitTs = np.zeros(policies)
        waitTs[0] = np.mean(self.DQN_events[2, start:end])
        return np.around(waitTs, 3)

    def get_responseTs(self, policies, start, end):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.DQN_events[3, start:end])
        return np.around(respTs, 3)

    def get_successTimes(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.DQN_events[7, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_successInTime(self, policies, start, end):
        successT = np.zeros(policies)
        successT[0] = sum(self.DQN_events[9, start:end]) / (end - start)
        successT = np.around(successT, 3)
        return successT

    def get_rejectTimes(self, policies, start, end):
        reject = np.zeros(policies)
        reject[0] = sum(self.DQN_events[8, start:end])
        return np.around(reject, 2)

    def get_totalRewards(self, policies, start):
        rewards = np.zeros(policies)
        rewards[0] = sum(self.DQN_events[5, start:self.requestNum])
        return np.around(rewards, 2)

    def get_totalMaliciousNum(self, policies):
        num = np.zeros(policies)
        maliciousOracleIndex = [0, 5, 10]
        num[0] = np.sum(self.DQN_oracle_events[1, maliciousOracleIndex])
        return np.around(num, 1)

    def get_totalNormalNum(self, policies):
        num = np.zeros(policies)
        # normalOracleIndex = [2, 7, 12]
        normalOracleIndex = [2, 7, 12]
        num[0] = np.sum(self.DQN_oracle_events[1, normalOracleIndex])
        return np.around(num, 1)

    def get_totalTrustedNum(self, policies):
        num = np.zeros(policies)
        trustedOracleIndex = [1, 3, 4, 6, 8, 9, 11, 13, 14]
        num[0] = np.sum(self.DQN_oracle_events[1, trustedOracleIndex])
        return np.around(num, 1)

    def get_totalMatchRate(self, policies):
        matchRate = np.zeros(policies)
        matchRate[0] = np.sum(self.DQN_oracle_events[3, :]) / self.requestNum
        return np.around(matchRate, 3)

    def get_totalTimes(self, policies, start):
        finishT = np.zeros(policies)
        finishT[0] = max(self.DQN_events[4, :]) - self.arrival_Times[start]
        return np.around(finishT, 2)



    def get_all_responseTs(self, policies):
        respTs = np.zeros((policies, self.requestNum))
        respTs[0, :] = self.DQN_events[3, :]
        return np.around(respTs, 3)

    def get_total_responseTs(self, policies, start):
        respTs = np.zeros(policies)
        respTs[0] = np.mean(self.DQN_events[3, start:self.requestNum])
        return np.around(respTs, 3)

    def get_totalSuccess(self, policies, start):
        successT = np.zeros(policies)
        successT[0] = sum(self.DQN_events[7, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(successT, 3)

    def get_totalSuccessInTime(self, policies, start):
        successT = np.zeros(policies)
        successT[0] = sum(self.DQN_events[9, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(successT, 3)

    def get_totalCost(self, policies, start):
        Cost = np.zeros(policies)
        Cost[0] = sum(self.DQN_events[8, start:self.requestNum]) / (self.requestNum - start + 1)
        return np.around(Cost, 3)