import math
import numpy as np
import random
from collections import deque
import tensorflow.compat.v1 as tf


'''
Tensorflow Setting
'''


tf.disable_eager_execution()
# tf.disable_v2_behavior()
random.seed(6)
np.random.seed(6)
# tf.set_random_seed(6)

class baseline_DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=800,
            batch_size=30,
            e_greedy_increment=0.002,
            # output_graph=False,
    ):
        self.n_actions = n_actions   # if +1: allow to reject requests
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        self.reward_list = []
        # self.reward_batch_list = []

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_initializer = tf.random_normal_initializer(0., 0.3, 5)  # (mean=0.0, stddev=1.0, seed=None)
        # w_initializer = tf.random_normal_initializer(0., 0.3)  # no seed
        b_initializer = tf.constant_initializer(0.1)
        n_l1 = 20  # config of layers

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # --------------------calculate loss---------------------
        self.action_input = tf.placeholder("float", [None, self.n_actions])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')  # for calculating loss
        q_evaluate = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_evaluate))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            print('xxxasdasdasd',self.loss)
            # self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # better than RMSProp

            # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

            # print('w1:', w1, '  b1:', b1, ' w2:', w2, ' b2:', b2)

    def choose_action(self, state):
        pro = np.random.uniform()
        if pro < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
            action = np.argmax(actions_value)
            # print('pro: ', pro, ' q-values:', actions_value, '  best_action:', action)
            # print('  best_action:', action)
        else:
            action = np.random.randint(0, self.n_actions)
            # print('pro: ', pro, '  rand_action:', action)
            # print('  rand_action:', action)
        return action

    def choose_best_action(self, state):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
        action = np.argmax(actions_value)
        return action

    def store_transition(self, s, a, r, s_):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('-------------target_params_replaced------------------')

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        # reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # calculate target q-value batch
        q_next_batch = self.sess.run(self.q_next, feed_dict={self.s_: next_state_batch})
        q_real_batch = []
        for i in range(self.batch_size):
            q_real_batch.append(minibatch[i][2] + self.gamma * np.max(q_next_batch[i]))
        # train eval network
        self.sess.run(self._train_op, feed_dict={
            self.s: state_batch,
            self.action_input: action_batch,
            self.q_target: q_real_batch
        })

        self.reward_list.append(np.mean([data[2] for data in minibatch]))
        # self.reward_batch_list.append(reward_batch)

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1

class baselines:
    def __init__(self, n_actions, oracletypes):
        self.n_actions = n_actions
        self.oracletypes = np.array(oracletypes)  # change list to numpy

        # FWA specific parameters
        self.FWA_sparks = 50  # Number of sparks per explosion
        self.FWA_amplitude = 20  # Amplitude range for generating sparks

    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, request_count):  # round robin policy
        action = (request_count-1) % self.n_actions
        return action

    def early_choose_action(self, idleTs):  # earliest policy
        action = np.argmin(idleTs)
        return action

    def FWA_choose_action(self, reputations):
        # Initialize
        best_action = np.argmax(reputations)
        best_cost = float('inf')
        sparks = []

        # Generate sparks based on current idle times
        for _ in range(self.FWA_sparks):
            for action in range(self.n_actions):
                # Simulate the explosion effect with randomness
                adjusted_idle_time = reputations[action] + np.random.randint(-self.FWA_amplitude, self.FWA_amplitude)
                sparks.append((action, adjusted_idle_time))

        # Evaluate sparks
        for spark in sparks:
            action, cost = spark
            # Assume cost is determined by the adjusted idle time
            if cost < best_cost:
                best_cost = cost
                best_action = action

        return best_action

    def PSG_choose_action(self, rewards, cost):  # semiGreedy policy
        oracles_attrs = np.column_stack((rewards, cost))
        # print(oracles_attrs)

        # choose oracles with reward greater than 0
        condition = oracles_attrs[:, 0] < 0
        candidate_list = {index: row for index, (row, keep) in enumerate(zip(oracles_attrs, ~condition)) if keep}
        # ascend ordering of candidate_list based on the cost of oracles
        # print(candidate_list)
        candidate_array = np.array(list(candidate_list.values()))
        # print(candidate_array)
        original_indices = np.array(list(candidate_list.keys()))
        # print(original_indices)

        sorted_indices = np.argsort(candidate_array[:, 1])
        sorted_array = candidate_array[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices]

        num_oracles = min(7, len(sorted_original_indices))
        # select the first 7 oracles
        # if num_oracles > 0 & num_oracles <= 7
        if num_oracles > 0:
            action = np.random.choice(sorted_original_indices[:num_oracles])
        else:
            # if candidate_array is NULL, randomly select an oracle from oracles_attrs
            action = np.random.choice(range(oracles_attrs.shape[0]))
        return action


class BLOR:

    def __init__(self, success_count, failure_count, cost):
        self.success_count = success_count
        self.failure_count = failure_count
        self.cost = cost
        self.gamma = 0.7
        self.x = 0.7
    def gamma_func(self, z):

        return math.gamma(z)


    def BETA(self, alpha, beta):

        return self.gamma_func(alpha) * self.gamma_func(beta) / self.gamma_func(alpha + beta)
        # return np.exp(gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))

    def beta_pdf(self, x, alpha, beta, a, b):
        if x < a or x > b:
            return 0
        if alpha > 0 and beta > 0:
            num = ((x - a) ** (alpha - 1)) * ((b - x) ** (beta - 1))
            den = self.BETA(alpha, beta) * ((b - a) ** (alpha + beta - 1))
            # if den == 0:
            #     # print(f"Warning: den is zero for alpha = {alpha}, beta = {beta}, num = {num}")
            #     result = float('inf')
            # else:
            result = num / den
            # # Debug prints
            # print(f"beta_pdf: x = {x}, alpha = {alpha}, beta = {beta}, a = {a}, b = {b}")
            # print(f"num = {num}, den = {den}, result = {result}")
            return result
        return 0

    def max_part(self, part):
        maximum = max(part)
        maximum_index = part.index(maximum)
        part_without_max = [x for i, x in enumerate(part) if i != maximum_index]
        maximum_2 = max(part_without_max) if part_without_max else maximum
        maximum_2_index = part.index(maximum_2)
        return maximum, maximum_index, maximum_2, maximum_2_index

    def get_oracles(self, success_count, failure_count, cost):
        oracles_array = np.column_stack((success_count, failure_count, cost))
        oracles = []
        for row in oracles_array:
            success_count = row[0]
            failure_count = row[1]
            cost = row[2]
            oracle = (success_count, failure_count, cost)
            oracles.append(oracle)
        return oracles

    def choose_action(self, oracles):
        number_of_oracles = len(oracles)
        degree_max = 0
        oracle_index_max = 0

        b_t_success = [self.beta_pdf(self.x, o[0] + 1, o[1] + 1, 0, 1) for o in oracles]
        b_t_fail = [self.beta_pdf(self.x, o[0] + 1, o[1] + 1, 0, 1) for o in oracles]
        part_1 = [0] * number_of_oracles
        part_2 = [0] * number_of_oracles
        part_3 = [0] * number_of_oracles
        c_n = [0] * number_of_oracles
        k_g = [0] * number_of_oracles
        degree = [0] * number_of_oracles

        for oracle_index, oracle in enumerate(oracles):
            p_o_success = oracle[0]
            p_o_fail = oracle[1]
            p_teta = p_o_success
            if p_teta == 0:
                p_o_success_teta = 0
                p_o_fail_teta = 0
            else:
                p_o_success_teta = (p_o_success / p_teta) / 2 / oracle[2]
                p_o_fail_teta = (p_o_fail / p_teta) / 2 / oracle[2]

            a = p_o_success
            a_prim = a + 1
            b = p_o_fail
            b_prim = b + 1

            beta_pdf_success = self.beta_pdf(self.x, a_prim, b_prim, 0, 1)
            beta_pdf_fail = self.beta_pdf(self.x, a_prim, b_prim, 0, 1)

            # # Debug prints for each iteration
            # print(f"Oracle {oracle_index}:")
            # print(f"  p_o_success_teta = {p_o_success_teta}")
            # print(f"  b_t_success (before) = {b_t_success[oracle_index]}")
            # print(
            #     f"  gamma * (p_o_success_teta * b_t_success[oracle_index]) = {self.gamma * (p_o_success_teta * b_t_success[oracle_index])}")
            # print(f"  (1 - gamma) * beta_pdf_success = {(1 - self.gamma) * beta_pdf_success}")

            b_t_success[oracle_index] = self.gamma * (p_o_success_teta * b_t_success[oracle_index]) + (
                    (1 - self.gamma) * beta_pdf_success)

            b_t_fail[oracle_index] = self.gamma * (p_o_fail_teta * b_t_fail[oracle_index]) + (
                    (1 - self.gamma) * beta_pdf_fail)

            # # Debug prints after calculation
            # print(f"  b_t_success (after) = {b_t_success[oracle_index]}")
            # print(f"  b_t_fail = {b_t_fail[oracle_index]}")
            # print(f"  beta_pdf_success = {beta_pdf_success}, beta_pdf_fail = {beta_pdf_fail}")

            part_1[oracle_index] = a / (a + b + 1)
            if a == 0 and b == 0:
                part_2[oracle_index] = 0
            else:
                part_2[oracle_index] = a / (a + b)
            part_3[oracle_index] = (a + 1) / (a + b + 1)

            # # Debug prints for part values
            # print(
            #     f"  part_1[{oracle_index}] = {part_1[oracle_index]}, part_2[{oracle_index}] = {part_2[oracle_index]}, part_3[{oracle_index}] = {part_3[oracle_index]}")

        for oracle_index in range(number_of_oracles):
            max_value, max_index, max_value_2, max_index_2 = self.max_part(part_2)

            if max_index != oracle_index:
                c_n[oracle_index] = max_value
            else:
                c_n[oracle_index] = max_value_2

            # print(
            #     f"Oracle {oracle_index}: max_value = {max_value}, max_index = {max_index}, max_value_2 = {max_value_2}, max_index_2 = {max_index_2}")

        for oracle_index in range(number_of_oracles):
            if c_n[oracle_index] < part_3[oracle_index] and c_n[oracle_index] >= part_2[oracle_index]:
                k_g[oracle_index] = b_t_success[oracle_index] * (part_3[oracle_index] - c_n[oracle_index])
            elif c_n[oracle_index] < part_2[oracle_index] and c_n[oracle_index] >= part_1[oracle_index]:
                k_g[oracle_index] = b_t_fail[oracle_index] * (c_n[oracle_index] - part_1[oracle_index])
            else:
                k_g[oracle_index] = 0

            # # Debug prints for k_g calculation
            # print(
            #     f"Oracle {oracle_index}: c_n = {c_n[oracle_index]}, k_g = {k_g[oracle_index]}, part_3[{oracle_index}] = {part_3[oracle_index]}, part_2[{oracle_index}] = {part_2[oracle_index]}, part_1[{oracle_index}] = {part_1[oracle_index]}")

            degree_value = b_t_success[oracle_index] + k_g[oracle_index]
            degree[oracle_index] = degree_value

            # # Debug prints
            # print(f"Oracle {oracle_index}: degree_value = {degree_value}")

            if degree_value > degree_max:
                degree_max = degree_value
                oracle_index_max = oracle_index

        return oracle_index_max