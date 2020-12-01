# -*- coding: utf-8 -*-
import os
# 允许重复加载动态链接库
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
import random
import numpy as np

from collections import deque
from network import Network


class DQN:
    def __init__(self):
        self.model = Network().build_model()
        self.target_model = Network().build_model()
        self.update_target_model()

        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')

        # 经验池
        self.memory_buffer = deque(maxlen=2000)
        # Q_value的discount rate，以便计算未来reward的折扣回报
        self.gamma = 0.95
        # 贪婪选择法的随机选择行为的程度
        self.epsilon = 1.0
        # 上述参数的衰减率
        self.epsilon_decay = 0.995
        # 最小随机探索的概率
        self.epsilon_min = 0.01

        self.env = gym.make('CartPole-v0')

    def update_target_model(self):
        """更新target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def sample(self, state):
        """ε-greedy选择action

        Arguments:
            state: 状态

        Returns:
            action: 动作
        """
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """向经验池添加数据

        Arguments:
            state: 状态
            action: 动作
            reward: 回报
            next_state: 下一个状态
            done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """更新epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """batch数据处理

        Arguments:
            batch: batch size

        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch)
        # 生成Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y
